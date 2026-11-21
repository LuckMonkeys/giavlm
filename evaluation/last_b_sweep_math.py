"""Cached public features and ranking for the fixed LoRA-B layer sweep."""
import statistics

import numpy as np
import torch


def position_modules():
    result = {}
    for layer in range(1, 33):
        for kind, module in (("down", "mlp.down_proj"), ("o", "self_attn.o_proj")):
            result[f"l{layer:02d}_{kind}"] = (
                f"backend.language_model.model.layers.{layer - 1}.{module}.lora_B.default.weight")
    return result


POSITIONS = position_modules()
METHODS = ("baseline", "lm_prior", "mix_lm_prior", *POSITIONS,
           *("wrong_" + p for p in POSITIONS), *("mix_" + p for p in POSITIONS))


@torch.no_grad()
def public_features(head, gamma, chunk=2048):
    if head.dtype != torch.float64 or gamma.dtype != torch.float64 or gamma.shape != head.shape[1:]:
        raise ValueError("Invalid float64 public vocabulary")
    features, norms = torch.empty_like(head), head.new_empty(len(head))
    for start in range(0, len(head), chunk):
        u = head[start:start + chunk] * gamma
        features[start:start + len(u)] = u
        norms[start:start + len(u)] = u.square().sum(-1)
    if not torch.isfinite(features).all() or (norms <= 0).any():
        raise ValueError("Invalid public output word direction")
    return features, norms


@torch.no_grad()
def cached_scores(gradient, features, norms, chunk=2048):
    if gradient.dtype != torch.float64 or gradient.shape[0] != features.shape[1]:
        raise ValueError("B gradient differs from cached public coordinate")
    energy = gradient.square().sum()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("Zero or nonfinite observed B gradient")
    result = features.new_empty(len(features))
    for start in range(0, len(features), chunk):
        u = features[start:start + chunk]
        result[start:start + len(u)] = -(u @ gradient).square().sum(-1) / (norms[start:start + len(u)] * energy)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite B word score")
    return result


def order_and_ranks(scores, flags, forbidden):
    values = scores.numpy()
    valid = ~flags.numpy().copy()
    valid[list(forbidden)] = False
    allowed = np.flatnonzero(valid)
    order = allowed[np.lexsort((allowed, values[allowed]))].astype(np.int32)
    ranks = np.zeros(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, ranks


def selection(order, ambiguous):
    return set(order[:100].tolist()) | ambiguous


def supplement(base_order, additional_order, ambiguous):
    chosen = set(base_order[:80].tolist())
    for token in additional_order:
        if int(token) not in chosen:
            chosen.add(int(token))
            if len(chosen) == 100:
                break
    if len(chosen) != 100:
        raise ValueError("Insufficient public candidate vocabulary")
    return chosen | ambiguous


def vector_metrics(ranks, chosen, truths, ambiguous):
    result = {}
    for part, truth in truths.items():
        if not truth:
            raise ValueError("Empty private content reference")
        informative = np.asarray(sorted(truth - ambiguous), dtype=np.int64)
        ap = None
        if ranks is not None and len(informative):
            ordered = np.sort(ranks[informative])
            if (ordered <= 0).any():
                raise ValueError("Reference outside valid public vocabulary")
            ap = statistics.fmean((np.arange(1, len(ordered) + 1) / ordered).tolist())
        result.update({f"{part}_recall": len(chosen & truth) / len(truth),
                       f"{part}_ap": ap, f"{part}_full_coverage": truth <= chosen})
    return result


def method_info(method):
    if method in ("baseline", "lm_prior", "mix_lm_prior"):
        return {"layer": None, "module": "", "mode": method}
    mode = "raw"
    if method.startswith("mix_"):
        mode, method = "mix", method[4:]
    elif method.startswith("wrong_"):
        mode, method = "wrong", method[6:]
    layer, module = method.split("_", 1)
    return {"layer": int(layer[1:]), "module": module, "mode": mode}


def layer_ranking(overall):
    lookup = {(r["condition"], r["image"], r["method"]): r for r in overall}
    ranked = []
    for position in POSITIONS:
        noise = lookup["noise", "random", "mix_" + position]
        topk = lookup["sparsify", "random", "mix_" + position]
        info = method_info(position)
        ranked.append({"position": position, **info,
            "noise_answer_recall": noise["answer_recall"], "topk_answer_recall": topk["answer_recall"],
            "mean_answer_recall": .5 * (noise["answer_recall"] + topk["answer_recall"]),
            "worst_answer_recall": min(noise["answer_recall"], topk["answer_recall"]),
            "mean_answer_full_coverage": .5 * (noise["answer_full_coverage"] + topk["answer_full_coverage"])})
    ranked.sort(key=lambda r: (-r["mean_answer_recall"], -r["worst_answer_recall"],
        -r["mean_answer_full_coverage"], -r["layer"], 0 if r["module"] == "down" else 1))
    for rank, row in enumerate(ranked, 1):
        row["development_rank"] = rank
    return ranked
