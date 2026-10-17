"""Reference-free prototype scores and evaluator-only classification summaries."""
import numpy as np
from scipy.stats import rankdata
import torch


METHODS = ['global', 'centered', 'equal_tensor', 'centered_equal_tensor']


@torch.no_grad()
def score_bank(predictions, targets, device='cpu'):
    """Score a fixed candidate bank against one or more uploads.

Equal weighting uses a common tensor mask for every candidate for each target.
Centering uses the candidate mean only, never any target or truth mean.
"""
    names = sorted(targets[0])
    if not predictions or any(sorted(x) != names for x in [*predictions, *targets]):
        raise ValueError('Nonempty candidates with identical upload parameter names required')
    nc, nt = len(predictions), len(targets)
    totals = torch.zeros(2, nt, nc, 3, dtype=torch.float64, device=device)
    equal = torch.zeros(2, nt, nc, dtype=torch.float64, device=device)
    counts = torch.zeros(2, nt, dtype=torch.long, device=device)
    for name in names:
        shape = targets[0][name].shape
        if any(x[name].shape != shape for x in [*predictions, *targets]):
            raise ValueError('Upload tensor shapes differ')
        p = torch.stack([x[name].detach().to(device=device, dtype=torch.float32).flatten()
                         for x in predictions])
        t = torch.stack([x[name].detach().to(device=device, dtype=torch.float32).flatten()
                         for x in targets])
        if not torch.isfinite(p).all() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite gradient')
        mean = p.mean(0, keepdim=True)
        original_p, original_t = p.square().sum(1), t.square().sum(1)
        for centered in range(2):
            a, b = (p, t) if not centered else (p-mean, t-mean)
            pn, tn = a.square().sum(1), b.square().sum(1)
            dot = b @ a.T
            totals[centered, :, :, 0] += dot.double()
            totals[centered, :, :, 1] += pn.double()[None, :]
            totals[centered, :, :, 2] += tn.double()[:, None]
            pfloor = original_p.double().mul(1e-16).clamp_min(1e-60) if centered else 0
            tfloor = original_t.double().mul(1e-16).clamp_min(1e-60) if centered else 0
            valid = (pn.double() > pfloor).all() & (tn.double() > tfloor)
            denom = (tn[:, None].double()*pn[None, :].double()).sqrt().clamp_min(1e-60)
            cosine = (1-dot.double()/denom).clamp(0, 2)
            equal[centered] += torch.where(valid[:, None], cosine, 0)
            counts[centered] += valid
    totals, equal, counts = totals.cpu().numpy(), equal.cpu().numpy(), counts.cpu().numpy()
    result = []
    for j in range(nt):
        row = {'scores': {}, 'valid_tensors': {}, 'tensor_count': len(names)}
        for centered, global_name, equal_name in [(0, 'global', 'equal_tensor'),
                                                  (1, 'centered', 'centered_equal_tensor')]:
            dot, pn, tn = totals[centered, j].T
            floorp = np.maximum(totals[0, j, :, 1]*1e-16, 1e-60) if centered else 0
            floort = np.maximum(totals[0, j, :, 2]*1e-16, 1e-60) if centered else 0
            valid = np.all(pn > floorp) and np.all(tn > floort)
            row['scores'][global_name] = np.clip(1-dot/np.sqrt(pn*tn), 0, 2).tolist() if valid else None
            n = int(counts[centered, j])
            row['scores'][equal_name] = (equal[centered, j]/n).tolist() if n else None
            row['valid_tensors'][equal_name] = n
        result.append(row)
    return result


def aggregate_classes(losses, bank_labels, classes, top_m=3):
    if losses is None:
        return None
    values = np.asarray(losses, dtype=float)
    if len(values) != len(bank_labels) or not np.isfinite(values).all():
        raise ValueError('Invalid candidate loss vector')
    result = []
    for label in classes:
        subset = values[np.asarray(bank_labels) == label]
        if len(subset) < top_m:
            raise ValueError('Insufficient candidates in class')
        result.append(float(np.sort(subset)[:top_m].mean()))
    return result


def binary_auc(y, score):
    y, score = np.asarray(y, dtype=bool), np.asarray(score)
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(score, method='average')
    return float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative))


def classification_stats(labels, scores, classes):
    """Missing classes stay missing; never present two-class BA as three-class BA."""
    y = np.array([classes.index(x) for x in labels], dtype=int)
    if not len(y):
        return {'n': 0}
    pred = np.array([int(np.argmin(x)) if x is not None else -1 for x in scores])
    matrix = np.zeros((len(classes), len(classes)+1), dtype=int)
    for a, b in zip(y, pred, strict=True):
        matrix[a, b if b >= 0 else len(classes)] += 1
    support = np.bincount(y, minlength=len(classes))
    recalls, f1s, auc = {}, {}, {}
    for c, name in enumerate(classes):
        tp, fp, fn = int(((y == c) & (pred == c)).sum()), int(((y != c) & (pred == c)).sum()), int(((y == c) & (pred != c)).sum())
        recalls[name] = tp/int(support[c]) if support[c] else None
        f1s[name] = 2*tp/(2*tp+fp+fn) if support[c] else None
        if all(s is not None for s in scores):
            # Positive means this class beats its strongest competitor.
            margins = [min(v for k, v in enumerate(s) if k != c)-s[c] for s in scores]
            auc[name] = binary_auc(y == c, margins)
        else:
            auc[name] = None
    supported = [v for v in recalls.values() if v is not None]
    return {'n': len(y), 'invalid': int((pred < 0).sum()), 'correct': int((pred == y).sum()),
            'accuracy': float((pred == y).mean()),
            'balanced_accuracy': float(np.mean(supported)) if (support > 0).all() else None,
            'supported_class_balanced_accuracy': float(np.mean(supported)),
            'macro_f1': float(np.mean(list(f1s.values()))) if (support > 0).all() else None,
            'supported_class_macro_f1': float(np.mean([x for x in f1s.values() if x is not None])),
            'support': dict(zip(classes, support.tolist(), strict=True)), 'recall': recalls,
            'class_auroc': auc, 'confusion': matrix.tolist(),
            'confusion_columns': [*classes, 'invalid']}


def paired_bootstrap(labels, scores, classes, other_scores=None, seed=20261016, repeats=2000):
    """Stratified image bootstrap; no QA or cross-rank pseudo-replication."""
    if not labels:
        return None
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in classes if np.any(labels == c)]
    def credit(values):
        return np.array([s is not None and classes[int(np.argmin(s))] == y
                         for y, s in zip(labels, values, strict=True)], dtype=float)
    values = credit(scores)
    if other_scores is not None:
        values -= credit(other_scores)
    per_class = [values[rng.choice(g, (repeats, len(g)), replace=True)].mean(1) for g in groups]
    accuracy = sum(v*len(g) for v, g in zip(per_class, groups, strict=True))/len(labels)
    values = np.stack([accuracy, np.mean(per_class, axis=0)], axis=1)
    intervals = np.quantile(values, [.025, .975], axis=0).T.tolist()
    return {'accuracy_ci95': intervals[0], 'supported_class_ba_ci95': intervals[1],
            'bootstrap_method': 'stratified image bootstrap', 'repeats': repeats, 'seed': seed}
