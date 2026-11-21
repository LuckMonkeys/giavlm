"""Cross-style weights and domain nuisance removal in per-tensor Gram space."""
import numpy as np

from evaluation.property_geometry import aggregate, centered_blocks, class_matrix, cosine_losses, weights


def style_coefficients(labels, styles):
    labels, styles = np.asarray(labels), np.asarray(styles)
    domains = sorted(set(styles))
    groups = []
    for domain in domains:
        columns = []
        for category in [0, 1]:
            mask = (styles == domain) & (labels == category)
            if not mask.any():
                raise ValueError("Each fit style must contain both classes")
            columns.append(mask / mask.sum())
        groups.append(np.stack(columns, axis=1))
    return groups


def consistency_weights(k, labels, styles, beta):
    coefficients = style_coefficients(labels, styles)
    if len(coefficients) < 2:
        raise ValueError("Cross-style consistency needs two fit styles")
    directions = [a[:, 0] - a[:, 1] for a in coefficients]
    terms = []
    valid_pairs = []
    scale = np.maximum(np.trace(k, axis1=1, axis2=2), 1e-60)
    for i, left in enumerate(directions):
        for right in directions[i + 1:]:
            a = np.einsum("i,tij,j->t", left, k, left)
            b = np.einsum("i,tij,j->t", right, k, right)
            valid = (a > scale * 1e-12) & (b > scale * 1e-12)
            denom = np.sqrt(np.maximum(a * b, 0))
            cross = np.einsum("i,tij,j->t", left, k, right)
            cosine = np.divide(cross, denom, out=np.zeros_like(cross), where=valid)
            terms.append(np.clip(cosine, 0, 1))
            valid_pairs.append(valid)
    valid = np.stack(valid_pairs).any(0)
    score = np.stack(terms).mean(0)
    result = np.zeros(len(k))
    if valid.any():
        result[valid] = (1 - beta) / valid.sum()
        if score[valid].sum() > 0:
            result[valid] += beta * score[valid] / score[valid].sum()
        else:
            result[valid] = 1 / valid.sum()
    else:
        valid = np.trace(k, axis1=1, axis2=2) > 1e-60
        if valid.any():
            result[valid] = 1 / valid.sum()
    return result, score


def domain_coefficients(labels, styles):
    groups = style_coefficients(labels, styles)
    mean = np.stack(groups).mean(0)
    return np.concatenate([a - mean for a in groups], axis=1)


def remove_directions(k, d, qn, coefficients, rank, alpha):
    """Apply (I-alpha UU') without forming a parameter-space projector."""
    nuisance = np.einsum("ni,tnm,mj->tij", coefficients, k, coefficients)
    eigen, vectors = np.linalg.eigh((nuisance + nuisance.transpose(0, 2, 1)) / 2)
    eigen, vectors = np.maximum(eigen[:, ::-1], 0), vectors[:, :, ::-1]
    keep = eigen > np.maximum(eigen[:, :1] * 1e-10, 1e-60)
    inverse = np.divide(1, np.sqrt(eigen), out=np.zeros_like(eigen), where=keep)
    coeff = coefficients @ (vectors * inverse[:, None, :])
    limit = coeff.shape[-1] if rank == "all" else rank
    coeff = coeff[:, :, :limit]
    fit, query = k @ coeff, d @ coeff
    gamma = alpha * (2 - alpha)
    return (k - gamma * (fit @ fit.transpose(0, 2, 1)),
            d - gamma * (query @ fit.transpose(0, 2, 1)),
            np.maximum(qn - gamma * np.square(query).sum(2), 0))


def score_cross_style(grams, fit, query, labels, styles):
    labels = np.asarray(labels)
    k, d, qn, raw, _ = centered_blocks(grams, fit, query)
    raw_kn = np.diagonal(raw, axis1=1, axis2=2)
    raw_qn = grams[:, query, query]
    losses, valid = cosine_losses(k, d, qn, raw_kn, raw_qn)

    def equal(loss, mask, weight=None):
        effective = mask * (np.ones(len(k)) if weight is None else weight)[:, None]
        mass = effective.sum(0)
        return np.divide(np.einsum("tq,tqn->qn", effective, loss), mass[:, None],
                         out=np.full(loss.shape[1:], np.nan), where=mass[:, None] > 0)

    common, _ = consistency_weights(k, labels, styles, 0.)
    common_mask = common > 0
    scores = {"equal": aggregate(equal(losses, valid, common), labels)}
    diagnostics = {}
    for beta in [0., .5, 1.]:
        w, s = consistency_weights(k, labels, styles, beta)
        scores[f"consistency_b{beta:g}"] = aggregate(equal(losses, valid, w), labels)
        old, _ = weights(k, labels, beta)
        old *= common_mask
        scores[f"dispersion_b{beta:g}"] = aggregate(equal(losses, valid, old), labels)
        diagnostics[f"b{beta:g}"] = {"weights": w.tolist(), "consistency": s.tolist(),
                                    "valid_tensor_fraction": float(common_mask.mean()),
                                    "effective_tensors": float(1 / max((w * w).sum(), 1e-60))}
    w, _ = consistency_weights(k, labels, styles, .5)
    scores["permuted_weights"] = aggregate(equal(losses, valid, np.random.default_rng(20261028).permutation(w)), labels)
    permuted = np.random.default_rng(20261028).permutation(labels)
    null, _ = consistency_weights(k, permuted, styles, .5)
    null *= common_mask
    scores["shuffled_labels"] = aggregate(equal(losses, valid, null), labels)
    a = class_matrix(labels)

    def prototype(kk, dd, nn):
        pk = np.einsum("ic,tij,jd->tcd", a, kk, a)
        pd = dd @ a
        loss, mask = cosine_losses(pk, pd, nn, np.diagonal(pk, axis1=1, axis2=2), raw_qn)
        return equal(loss, mask)

    scores["prototype"] = prototype(k, d, qn)
    for kind, coefficients in [("domain", domain_coefficients(labels, styles)),
                               ("within", (np.eye(len(labels)) - a[:, labels].T).T)]:
        for rank in [1, "all"]:
            # Within-class full rank would remove all fit variation; use rank 2 as matched budget.
            actual = 2 if kind == "within" and rank == "all" else rank
            for alpha in [.5, 1.]:
                kk, dd, nn = remove_directions(k, d, qn, coefficients, actual, alpha)
                key = f"{kind}_k{rank}_a{alpha:g}"
                scores[key] = prototype(kk, dd, nn)
                delta = a[:, 0] - a[:, 1]
                before = np.einsum("i,tij,j->t", delta, k, delta).sum()
                after = np.einsum("i,tij,j->t", delta, kk, delta).sum()
                diagnostics[key] = {"between_retained": float(after / max(before, 1e-60))}
    return scores, diagnostics
