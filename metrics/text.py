from collections import Counter
import re
from rouge_score.rouge_scorer import RougeScorer

def words(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def text_metrics(reference, prediction):
    ref, pred = words(reference), words(prediction)
    scores = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False).score(reference, prediction)
    overlap = sum((Counter(ref) & Counter(pred)).values())
    return {"exact_match": float(reference == prediction),
            "normalized_exact_match": float(ref == pred),
            "wer": edit_distance(ref, pred) / max(1, len(ref)),
            "word_recall": overlap / max(1, len(ref)),
            **{name: score.fmeasure for name, score in scores.items()}}


def token_set_f1(reference_ids, prediction_ids):
    """Set F1 over tokenizer IDs supplied by the caller; not word-token F1."""
    ref, pred = set(reference_ids), set(prediction_ids)
    return 2 * len(ref & pred) / (len(ref) + len(pred)) if ref or pred else 1.0


def pii_exact_match_recall(canary_values, prediction):
    """Case-sensitive complete synthetic entity matches; empty canary set is undefined."""
    values = set(canary_values)
    if not values:
        return None
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Canary entities must be nonempty strings")
    return sum(bool(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", prediction))
               for value in values) / len(values)
