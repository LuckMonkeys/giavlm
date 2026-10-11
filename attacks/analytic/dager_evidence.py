"""Public-counterfactual calibration and LoRA-B statistical evidence.

B energy is a heuristic feature, not an embedding span-membership certificate.
All fitting labels must come from attacker-generated public examples.
"""
import torch
import torch.nn.functional as F


def negative_moments(values, present, floor_ratio=.1):
    """Token-specific null moments; present-token observations are excluded."""
    mask = (~present).to(values.dtype).unsqueeze(-1)
    count = mask.sum(0)
    mean = (values * mask).sum(0) / count.clamp_min(1)
    variance = ((values - mean).square() * mask).sum(0) / count.clamp_min(1)
    global_mean = (values * mask).sum((0, 1)) / mask.sum().clamp_min(1)
    global_var = ((values - global_mean).square() * mask).sum((0, 1)) / mask.sum().clamp_min(1)
    low = count[:, 0] < 4
    mean[low] = global_mean
    variance[low] = global_var
    std = variance.sqrt().clamp_min(global_var.sqrt() * floor_ratio).clamp_min(1e-6)
    return {'null_mean': mean, 'null_std': std, 'null_count': count[:, 0]}


def evidence_features(raw, calibration):
    z = (raw - calibration['null_mean']) / calibration['null_std']
    return torch.cat([raw, z.clamp(-30, 30)], dim=-1)


def b_energy(features, weight_a, weight_b, gradient_b):
    """Energy in the B-gradient component not determined by the A sketch."""
    q = torch.linalg.qr(weight_b.double(), mode='reduced')[0]
    independent = gradient_b.double() - q @ (q.T @ gradient_b.double())
    covariance = independent.T @ independent
    covariance = covariance / covariance.trace().clamp_min(1e-30)
    compressed = features.double() @ weight_a.double().T
    energy = (compressed @ covariance * compressed).sum(1)
    energy = energy / compressed.square().sum(1).clamp_min(1e-30)
    return energy.clamp_min(1e-12).log(), independent, compressed


def fit_logistic(features, labels, columns, regularization):
    x = features[:, columns].double()
    y = labels.double()
    mean, scale = x.mean(0), x.std(0).clamp_min(.01)
    x = (x - mean) / scale
    parameter = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    weights = torch.where(y.bool(), .5 / y.sum(), .5 / (1-y).sum())
    optimizer = torch.optim.LBFGS([parameter], lr=1, max_iter=300,
                                  tolerance_grad=1e-8, tolerance_change=1e-11,
                                  line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        logits = x @ parameter[:-1] + parameter[-1]
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * weights).sum()
        loss = loss + regularization * parameter[:-1].square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    return {'columns': torch.tensor(columns), 'mean': mean.detach(), 'scale': scale.detach(),
            'parameter': parameter.detach()}


def logistic_score(features, fitted):
    x = (features[:, fitted['columns']] - fitted['mean']) / fitted['scale']
    return -(x @ fitted['parameter'][:-1] + fitted['parameter'][-1])
