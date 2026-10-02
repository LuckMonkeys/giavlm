"""Post-commit comparison of DAGER token-score artifacts against private IDs."""
import math
from pathlib import Path
import statistics

import numpy as np

from core.artifacts import read_json, read_tensors
from evaluation.token_recovery import set_metrics


def _summary(values):
    return {"minimum": min(values), "median": statistics.median(values),
            "maximum": max(values)} if values else None


def _distribution(values):
    """Aggregate a reference-control distribution without retaining private IDs."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"minimum": float(array.min()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)), "maximum": float(array.max()),
            "mean": float(array.mean())}


def _hypergeometric_tail(population, positives, draws, observed):
    """Exact P[X >= observed] for sampling ``draws`` IDs without replacement."""
    if not 0 <= positives <= population or not 0 <= draws <= population:
        raise ValueError("Invalid hypergeometric population")
    lower = max(observed, 0, positives - (population - draws))
    upper = min(positives, draws)
    if lower > upper:
        return 0.0
    denominator = math.comb(population, positives)
    return float(sum(math.comb(draws, hits)
                     * math.comb(population - draws, positives - hits)
                     for hits in range(lower, upper + 1)) / denominator)


def _ranking_metrics(ranking, truth, topks):
    rank_by_id = {token_id: index + 1 for index, token_id in enumerate(ranking)}
    ranks = sorted(rank_by_id[token_id] for token_id in truth if token_id in rank_by_id)
    population, positives = len(ranking), len(truth)
    average_precision = (sum((index + 1) / rank for index, rank in enumerate(ranks))
                         / positives if positives else 1.0)
    negatives = population - positives
    wins = sum(population - rank - (positives - index - 1)
               for index, rank in enumerate(ranks))
    roc_auc = wins / (positives * negatives) if positives and negatives else 1.0
    curve = []
    for topk in topks:
        selected = set(ranking[:topk])
        metrics = set_metrics(selected, truth)
        metrics.update({"topk": topk, "hits": len(selected & truth),
                        "random_expected_hits": topk * positives / population})
        curve.append(metrics)
    return {"average_precision": average_precision, "roc_auc": roc_auc,
            "first_hit_rank": ranks[0] if ranks else None,
            "mean_reciprocal_true_rank": (sum(1 / rank for rank in ranks) / positives
                                           if positives else 1.0),
            "true_token_rank": _summary(ranks), "curve": curve}


def _random_null(population, positives, topk, observed, draws, seed):
    rng = np.random.default_rng(seed)
    simulated = rng.hypergeometric(positives, population - positives, topk, size=draws)
    exceedances = int((simulated >= observed).sum())
    return {"null": "uniform random top-k / random score permutation",
            "population_size": population, "reference_count": positives, "topk": topk,
            "observed_hits": observed, "expected_hits": topk * positives / population,
            "expected_recall": topk / population,
            "exact_probability_at_least_observed": _hypergeometric_tail(
                population, positives, topk, observed),
            "simulation_draws": draws, "simulation_seed": seed,
            "simulation_exceedances": exceedances,
            "simulation_p_with_plus_one": (exceedances + 1) / (draws + 1)}


def _wrong_reference_control(selected, correct, references):
    correct_metrics = set_metrics(selected, correct)
    recalls = [set_metrics(selected, reference)["recall"] for reference in references]
    hits = [len(selected & reference) for reference in references]
    return {"reference_count": len(references),
            "correct_hits": len(selected & correct),
            "correct_recall": correct_metrics["recall"],
            "wrong_hits": _distribution(hits), "wrong_recall": _distribution(recalls),
            "empirical_probability_wrong_recall_at_least_correct": (
                (sum(value >= correct_metrics["recall"] for value in recalls) + 1)
                / (len(recalls) + 1)) if recalls else None}


def validate_token_filters(attack_dirs, truth_dir, training, wrong_references=None,
                           exact_length_references=None, wrong_references_by_field=None,
                           exact_length_references_by_field=None,
                           distinct_references_by_field=None,
                           exact_length_distinct_references_by_field=None, topk=50,
                           topks=(1, 5, 10, 20, 50, 100, 200, 256),
                           random_draws=100_000, seed=20261002):
    """Test saved rankings against random and wrong-reference post-hoc controls.

    ``wrong_references`` contains token-ID sets from other texts. Callers must
    construct those sets only in evaluation code after all attacks are committed.
    No individual reference IDs or text are retained in the returned report.
    """
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if type(random_draws) is not int or random_draws <= 0:
        raise ValueError("random_draws must be a positive integer")
    topks = tuple(sorted(set(topks) | {topk}))
    if not topks or topks[0] <= 0:
        raise ValueError("topks must contain positive integers")
    truth_dir = Path(truth_dir)
    references = read_tensors(truth_dir / "text_tokens.safetensors")
    private_fields = ["targets"] + (["questions"] if training.question_private else [])
    report = {"schema_version": 1, "unit": "unique_token_id", "topk": topk,
              "random_draws": random_draws, "random_seed": seed, "modes": {},
              "reference_policy": (
                  "private IDs and wrong-text controls read only after committed attacks; "
                  "report retains aggregate statistics only")}
    common_forbidden = None
    for mode_index, (name, directory) in enumerate(attack_dirs.items()):
        directory = Path(directory)
        if not (directory / "result.json").exists():
            raise ValueError(f"Attack {name} has not committed result.json")
        tensors = read_tensors(directory / "token_candidates.safetensors")
        metadata = read_json(directory / "token_candidates.json")
        if metadata["schema_version"] != 1 or metadata["scope"] != "private_text_union":
            raise ValueError(f"Unsupported token-candidate artifact for {name}")
        forbidden = set(metadata["forbidden_ids"])
        if common_forbidden is not None and forbidden != common_forbidden:
            raise ValueError("Compared attacks use different forbidden token sets")
        common_forbidden = forbidden
        allowed = [token_id for token_id in tensors["token_ids"].tolist()
                   if token_id not in forbidden]
        ranking = sorted(allowed, key=lambda token_id: (float(tensors["scores"][token_id]),
                                                        token_id))
        if len(ranking) < topks[-1]:
            raise ValueError(f"Attack {name} has fewer IDs than the largest top-k")
        by_field = {field: set(references[field].flatten().tolist()) - forbidden
                    for field in private_fields}
        truth = set().union(*by_field.values())
        selected = set(ranking[:topk])
        mode = {"ranking": _ranking_metrics(ranking, truth, topks),
                "recall_by_field_at_topk": {
                    field: set_metrics(selected, ids)["recall"]
                    for field, ids in by_field.items()},
                "random_topk_control": _random_null(
                    len(ranking), len(truth), topk, len(selected & truth),
                    random_draws, seed + mode_index)}
        if wrong_references is not None:
            cleaned = [set(value) - forbidden for value in wrong_references]
            mode["wrong_text_control"] = _wrong_reference_control(selected, truth, cleaned)
        if exact_length_references is not None:
            cleaned = [set(value) - forbidden for value in exact_length_references]
            mode["exact_length_wrong_text_control"] = _wrong_reference_control(
                selected, truth, cleaned)
        if wrong_references_by_field is not None:
            mode["wrong_text_control_by_field"] = {
                field: _wrong_reference_control(
                    selected, by_field[field],
                    [set(value) - forbidden for value in wrong_references_by_field[field]])
                for field in private_fields}
        if exact_length_references_by_field is not None:
            mode["exact_length_wrong_text_control_by_field"] = {
                field: _wrong_reference_control(
                    selected, by_field[field],
                    [set(value) - forbidden
                     for value in exact_length_references_by_field[field]])
                for field in private_fields}
        if distinct_references_by_field is not None:
            mode["distinct_wrong_text_control_by_field"] = {
                field: _wrong_reference_control(
                    selected, by_field[field],
                    [set(value) - forbidden for value in distinct_references_by_field[field]])
                for field in private_fields}
        if exact_length_distinct_references_by_field is not None:
            mode["exact_length_distinct_wrong_text_control_by_field"] = {
                field: _wrong_reference_control(
                    selected, by_field[field],
                    [set(value) - forbidden
                     for value in exact_length_distinct_references_by_field[field]])
                for field in private_fields}
        report["modes"][name] = mode
    return report


def compare_token_filters(attack_dirs, truth_dir, training, topk=50):
    """Compare equal-size score rankings only after every attack is committed."""
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be a positive integer")
    truth_dir = Path(truth_dir)
    reference_path = truth_dir / "text_tokens.safetensors"
    if not reference_path.exists():
        raise ValueError("Exact private token-ID artifact is required")
    references = read_tensors(reference_path)
    rankings, reports = {}, {}
    forbidden_sets = []
    private_fields = ["targets"] + (["questions"] if training.question_private else [])

    for name, directory in attack_dirs.items():
        directory = Path(directory)
        if not (directory / "result.json").exists():
            raise ValueError(f"Attack {name} has not committed result.json")
        tensors = read_tensors(directory / "token_candidates.safetensors")
        metadata = read_json(directory / "token_candidates.json")
        if metadata["schema_version"] != 1 or metadata["scope"] != "private_text_union":
            raise ValueError(f"Unsupported token-candidate artifact for {name}")
        forbidden = set(metadata["forbidden_ids"])
        forbidden_sets.append(forbidden)
        allowed = [token_id for token_id in tensors["token_ids"].tolist()
                   if token_id not in forbidden]
        ranking = sorted(allowed, key=lambda token_id: (float(tensors["scores"][token_id]),
                                                        token_id))
        if len(ranking) < topk:
            raise ValueError(f"Attack {name} has fewer than topk allowed tokens")
        rankings[name] = ranking
        selected = set(ranking[:topk])
        by_field = {field: set(references[field].flatten().tolist()) - forbidden
                    for field in private_fields}
        private_union = set().union(*by_field.values())
        rank_by_id = {token_id: index + 1 for index, token_id in enumerate(ranking)}
        reports[name] = {
            "topk": set_metrics(selected, private_union),
            "ambiguous_in_topk": sum(bool(tensors["ambiguous"][token_id])
                                     for token_id in selected),
            "recall_by_field": {field: set_metrics(selected, ids)["recall"]
                                for field, ids in by_field.items()},
            "private_token_rank": _summary([rank_by_id[token_id]
                                             for token_id in private_union]),
            "private_tokens_outside_topk": len(private_union - selected),
        }

    if any(forbidden != forbidden_sets[0] for forbidden in forbidden_sets[1:]):
        raise ValueError("Compared attacks use different forbidden token sets")
    names = list(rankings)
    pairs = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            a, b = set(rankings[left][:topk]), set(rankings[right][:topk])
            intersection = len(a & b)
            pairs[f"{left}__{right}"] = {
                "intersection": intersection, "union": len(a | b),
                "jaccard": intersection / len(a | b)}
    return {"schema_version": 1, "unit": "unique_token_id", "topk": topk,
            "modes": reports, "pairs": pairs,
            "reference_policy": "private content IDs after committed attack; special IDs excluded"}
