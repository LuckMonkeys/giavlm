"""Candidate-label statistics in low-rank Gram geometry; no target labels."""
import numpy as np


BETAS = (0.0, 0.5, 1.0)
SUPPRESSION = ((1, 0.5), (1, 1.0), (4, 0.5), (4, 1.0))
RIDGES = (0.1, 1.0)


def centered_blocks(grams, fit, query):
    fit, query = np.asarray(fit), np.asarray(query)
    raw = grams[:, fit[:, None], fit]
    cross = grams[:, query[:, None], fit]
    mean = raw.mean((1, 2))
    k = raw - raw.mean(1)[:, None, :] - raw.mean(2)[:, :, None] + mean[:, None, None]
    d = cross - cross.mean(2)[:, :, None] - raw.mean(1)[:, None, :] + mean[:, None, None]
    qn = grams[:, query, query] - 2 * cross.mean(2) + mean[:, None]
    return (k + k.transpose(0, 2, 1)) / 2, d, np.maximum(qn, 0), raw, cross


def class_matrix(labels):
    labels = np.asarray(labels)
    if set(labels) != {0, 1}:
        raise ValueError('Two public candidate classes required')
    return np.stack([(labels == c) / (labels == c).sum() for c in range(2)], axis=1)


def weights(k, labels, beta):
    a = class_matrix(labels)
    means = np.einsum('ic,tij,jc->tc', a, k, a)
    counts = np.bincount(labels, minlength=2)
    between = np.maximum((means * counts).sum(1), 0)
    total = np.maximum(np.trace(k, axis1=1, axis2=2), 0)
    ratio = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    ratio = np.clip(ratio, 0, 1)
    # Numeric energy floor is relative to each tensor, never a fixed gradient scale.
    valid = total > np.finfo(float).tiny
    result = np.zeros_like(ratio)
    if valid.any():
        result[valid] = (1 - beta) / valid.sum()
        if ratio[valid].sum() > 0:
            result[valid] += beta * ratio[valid] / ratio[valid].sum()
        else:
            result[valid] = 1 / valid.sum()
    return result, ratio


def cosine_losses(k, d, qn, raw_kn, raw_qn):
    pn = np.maximum(np.diagonal(k, axis1=1, axis2=2), 0)
    valid = ((pn > np.maximum(raw_kn * 1e-16, 1e-60)).all(1)[:, None]
             & (qn > np.maximum(raw_qn * 1e-16, 1e-60)))
    denom = np.sqrt(qn[:, :, None] * pn[:, None, :])
    losses = np.clip(1 - np.divide(d, denom, out=np.zeros_like(d), where=denom > 0), 0, 2)
    return losses, valid


def aggregate(losses, labels, m=3):
    return np.stack([np.sort(losses[:, np.asarray(labels) == c], axis=1)[:, :m].mean(1)
                     for c in range(2)], axis=1)


def score_geometry(grams, fit, query, labels, seed=20261028):
    """Return per-query two-class losses, including diagnostic parameter grids.

    Raw Gram products must be accumulated in float64. Every transformation and
    weight is fitted exclusively on `fit`; query values never fit statistics.
    """
    labels = np.asarray(labels, dtype=int)
    k, d, qn, raw, cross = centered_blocks(grams, fit, query)
    raw_kn = np.diagonal(raw, axis1=1, axis2=2)
    raw_qn = grams[:, query, query]
    lc, valid = cosine_losses(k, d, qn, raw_kn, raw_qn)
    lr, vr = cosine_losses(raw, cross, raw_qn, raw_kn, raw_qn)
    out = {}

    def equal(loss, mask, w=None):
        w = np.ones(len(k)) if w is None else w
        effective = w[:, None] * mask
        mass = effective.sum(0)
        values = np.einsum('tq,tqn->qn', effective, loss)
        return np.divide(values, mass[:, None], out=np.full_like(values, np.nan), where=mass[:, None] > 0)

    def global_score(kk, dd, nn):
        pn = np.maximum(np.diagonal(kk, axis1=1, axis2=2).sum(0), 0)
        tn = np.maximum(nn.sum(0), 0)
        den = np.sqrt(tn[:, None] * pn)
        return np.clip(1 - np.divide(dd.sum(0), den, out=np.zeros_like(dd[0]), where=den > 0), 0, 2)

    out['global'] = aggregate(global_score(raw, cross, raw_qn), labels)
    out['centered'] = aggregate(global_score(k, d, qn), labels)
    out['equal_tensor'] = aggregate(equal(lr, vr), labels)
    out['centered_equal_tensor'] = aggregate(equal(lc, valid), labels)
    w, ratio = weights(k, labels, .5)
    for beta in BETAS:
        wb, _ = weights(k, labels, beta)
        out[f'weighted_b{beta:g}'] = aggregate(equal(lc, valid, wb), labels)
    rng = np.random.default_rng(seed)
    random_labels = rng.permutation(labels)
    wnull, _ = weights(k, random_labels, .5)
    out['shuffled_labels_b0.5'] = aggregate(equal(lc, valid, wnull), labels)
    out['permuted_weights_b0.5'] = aggregate(equal(lc, valid, rng.permutation(w)), labels)

    a = class_matrix(labels)

    def prototype(kk, dd, nn):
        pk = np.einsum('ic,tij,jd->tcd', a, kk, a)
        pd = dd @ a
        pl, pv = cosine_losses(pk, pd, nn,
                                np.maximum(np.diagonal(pk, axis1=1, axis2=2), 0), raw_qn)
        return equal(pl, pv)

    out['prototype_equal'] = prototype(k, d, qn)
    h = np.eye(len(labels)) - a[:, labels].T
    within = h @ k @ h.T
    eig, vec = np.linalg.eigh((within + within.transpose(0, 2, 1)) / 2)
    order = np.arange(len(labels) - 1, -1, -1)
    eig, vec = np.maximum(eig[:, order], 0), vec[:, :, order]
    keep = eig > np.maximum(eig[:, :1] * 1e-10, 1e-60)
    inv = np.divide(1, np.sqrt(eig), out=np.zeros_like(eig), where=keep)
    coeff = h.T @ (vec * inv[:, None, :])
    zfit, zquery = k @ coeff, d @ coeff
    total_between = np.einsum('ic,tij,jc->t', a, k, a)
    diagnostics = {'tensor_ratio': ratio.tolist(), 'weights_b0.5': w.tolist(),
                   'within_rank': keep.sum(1).tolist(), 'suppression': {}}
    for rank, alpha in SUPPRESSION:
        f, q = zfit[:, :, :rank], zquery[:, :, :rank]
        gamma = alpha * (2 - alpha)
        kp = k - gamma * (f @ f.transpose(0, 2, 1))
        dp = d - gamma * (q @ f.transpose(0, 2, 1))
        npq = np.maximum(qn - gamma * (q * q).sum(2), 0)
        name = f'suppress_k{rank}_a{alpha:g}'
        out[name] = prototype(kp, dp, npq)
        post_between = np.einsum('ic,tij,jc->t', a, kp, a)
        diagnostics['suppression'][name] = {
            'between_retained': float(post_between.sum() / max(total_between.sum(), 1e-60)),
            'within_retained': float(np.trace(h @ kp @ h.T, axis1=1, axis2=2).sum()
                                     / max(np.trace(within, axis1=1, axis2=2).sum(), 1e-60)),
            'query_energy_retained': (npq.sum(0) / np.maximum(qn.sum(0), 1e-60)).tolist()}
    # Regularized Fisher in each tensor, followed by an equal-tensor score.
    # Scaling by the class separation gives a dimensionless, unit-invariant margin.
    delta = a[:, 0] - a[:, 1]
    for ridge in RIDGES:
        scale = np.maximum(np.trace(within, axis1=1, axis2=2) / len(labels), 1e-60) * ridge
        ed = np.einsum('tni,n->ti', zfit, delta)
        shrink = eig / (eig + scale[:, None])
        base = d @ delta
        corr = (zquery * ed[:, None, :] * shrink[:, None, :]).sum(2)
        sep = np.einsum('i,tij,j->t', delta, k, delta) - (ed * ed * shrink).sum(1)
        margin = np.divide(base - corr, sep[:, None], out=np.zeros_like(base), where=sep[:, None] > 1e-60)
        finite = sep > np.maximum(np.einsum('i,tij,j->t', delta, k, delta) * 1e-10, 1e-60)
        avg = margin[finite].mean(0) if finite.any() else np.full(len(query), np.nan)
        out[f'fisher_r{ridge:g}'] = np.stack([-avg, avg], axis=1)
    return out, diagnostics
