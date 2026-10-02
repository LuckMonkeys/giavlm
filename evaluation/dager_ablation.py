"""Post-commit comparison of DAGER token-score artifacts against private IDs."""
from pathlib import Path
import statistics

from core.artifacts import read_json, read_tensors
from evaluation.token_recovery import set_metrics


def _summary(values):
    return {"minimum": min(values), "median": statistics.median(values),
            "maximum": max(values)} if values else None


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
