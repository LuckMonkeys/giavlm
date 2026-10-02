"""Post-commit token-ID evaluation; never imported by an attacker."""
from pathlib import Path

from core.artifacts import read_json, read_tensors


def set_metrics(predicted, reference):
    true_positive = len(predicted & reference)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(reference) if reference else 1.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "selected_count": len(predicted), "reference_count": len(reference)}


def evaluate_token_candidates(reconstruction_dir, truth_dir, training):
    reconstruction_dir, truth_dir = Path(reconstruction_dir), Path(truth_dir)
    if not (reconstruction_dir / "result.json").exists():
        raise ValueError("Token evaluation requires a committed reconstruction")
    candidates_path = reconstruction_dir / "token_candidates.safetensors"
    if not candidates_path.exists():
        return {"status": "unavailable", "reason": "Token filtering did not finish"}
    reference_path = truth_dir / "text_tokens.safetensors"
    if not reference_path.exists():
        return {"status": "unavailable", "reason": "Capture has no exact private token-ID artifact"}
    metadata = read_json(reconstruction_dir / "token_candidates.json")
    if metadata["schema_version"] != 1 or metadata["scope"] != "private_text_union":
        raise ValueError("Unsupported token-candidate artifact schema")
    values = read_tensors(candidates_path)
    references = read_tensors(reference_path)
    forbidden = set(metadata["forbidden_ids"])
    private = ["targets"] + (["questions"] if training.question_private else [])
    by_field = {field: set(references[field].flatten().tolist()) - forbidden for field in private}
    truth = set().union(*by_field.values())
    selected = set(values["selected_ids"].tolist()) - forbidden
    ambiguous = set(values["token_ids"][values["ambiguous"]].tolist()) - forbidden
    return {"status": "completed", "scope": "private_text_union", "unit": "unique_token_id",
            "candidates": set_metrics(selected, truth),
            "informative_detections": set_metrics(selected - ambiguous, truth),
            "ambiguous_reference_count": len(truth & ambiguous),
            "candidate_recall_by_field": {
                field: set_metrics(selected, ids)["recall"] for field, ids in by_field.items()},
            "note": "Candidates include ambiguous public-overlap IDs; neither set recovers token order."}
