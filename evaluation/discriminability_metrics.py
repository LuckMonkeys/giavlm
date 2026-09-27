"""Optimizer-independent, private-reference probes and image-cluster statistics."""
import math
import re

import numpy as np
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from attacks.init import perturb
from attacks.objectives import matching_loss
from core.types import Batch
from metrics.image import image_metrics

LEVELS = [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, .01, .03, .1, .3, 1.]
RADII = [3e-4, 1e-3, 3e-3, .01, .03, .1]
STEPS = [1e-5, 1e-4, 1e-3]
THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, .01, .03, .1, .3, 1., 3., 10.]
PRIMARY = ['cosine', 'relative_l2', 'equal_tensor_cosine', 'centered_cosine']


def candidates(truth, others=()):
    """Yield deterministic CPU fp32 candidates; others already have fixed opaque IDs."""
    truth = truth.detach().float().cpu()
    yield 'truth', {'family': 'truth'}, truth
    for family, levels in [('uniform_mix', LEVELS), ('gaussian', LEVELS[:-1])]:
        for seed in range(3):
            for level in levels:
                yield (f'{family}-{seed}-{level:g}',
                       {'family': family, 'level': level, 'seed': seed},
                       perturb(truth, family, level, seed))
    for sigma in [.5, 1., 2., 4.]:
        yield f'blur-{sigma:g}', {'family': 'blur', 'level': sigma}, perturb(
            truth, 'blur', sigma, 0)
    height, width = truth.shape[-2:]
    for axis in ['right', 'down']:
        for pixels in [1, 4]:
            pad = (pixels, 0, 0, 0) if axis == 'right' else (0, 0, pixels, 0)
            x = F.pad(truth, pad, mode='replicate')[..., :height, :width]
            yield f'shift-{axis}-{pixels}', {'family': 'shift', 'axis': axis,
                                            'level': pixels}, x
    for fraction in [.1, .25, .5]:
        h, w = round(height * math.sqrt(fraction)), round(width * math.sqrt(fraction))
        y, x = (height - h) // 2, (width - w) // 2
        out = truth.clone()
        out[..., y:y+h, x:x+w] = truth.mean(dim=(-2, -1), keepdim=True)
        yield f'occlude-{fraction:g}', {'family': 'occlusion', 'level': fraction}, out
    if height % 7 or width % 7:
        raise ValueError('Patch shuffle requires image dimensions divisible by 7')
    for seed in range(3):
        h, w = height // 7, width // 7
        patches = truth.reshape(1, 3, 7, h, 7, w).permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(49, 3, h, w)
        order = torch.randperm(49, generator=torch.Generator().manual_seed(seed))
        x = patches[order].reshape(1, 7, 7, 3, h, w)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape_as(truth)
        yield f'shuffle-{seed}', {'family': 'shuffle', 'seed': seed}, x
    for value in [0., .5, 1.]:
        yield f'constant-{value:g}', {'family': 'constant', 'level': value}, torch.full_like(
            truth, value)
    for identifier, kind, x in others:
        yield identifier, {'family': kind}, x.detach().float().cpu()


def replay(adapter, images, batch, differentiable=False):
    from core.fl import simulate_update
    candidate = Batch(images.to(adapter.device, adapter.dtype).contiguous(),
                      batch.questions, batch.targets)
    return simulate_update(adapter, candidate, adapter.training_spec, differentiable)


def _moments(x, y):
    x = x.detach().float()
    # Observation tensors are intentionally deserialized on CPU. Match the
    # candidate update's device here, just as the differentiable objective does.
    y = y.detach().to(device=x.device, dtype=torch.float32)
    values = torch.stack([(x * y).sum(), x.square().sum(), y.square().sum(),
                          (x - y).square().sum()])
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite update moments')
    return values.double().cpu().numpy()


def _scores(moment):
    dot, nx, ny, error = (float(v) for v in moment)
    return {'cosine': 1 - dot / math.sqrt(nx * ny) if nx > 0 and ny > 0 else None,
            'relative_l2': error / ny if ny > 0 else None,
            'norm': math.sqrt(nx), 'target_norm': math.sqrt(ny),
            'norm_ratio': math.sqrt(nx / ny) if ny > 0 else None,
            'zero_target': ny == 0, 'zero_candidate': nx == 0}


def update_scores(predicted, target, mean=None, connector=lambda name: False):
    if set(predicted) != set(target):
        raise ValueError('Candidate and target update parameter sets differ')
    totals, groups, tensors = np.zeros(4), {}, {}
    for name in sorted(target):
        if predicted[name].shape != target[name].shape:
            raise ValueError('Candidate update shape differs')
        m = _moments(predicted[name], target[name])
        totals += m
        tensors[name] = _scores(m)
        group = ('connector' if connector(name) else 'lora_A' if 'lora_A' in name
                 else 'lora_B' if 'lora_B' in name else 'other')
        groups[group] = groups.get(group, np.zeros(4)) + m
        layer = re.search(r'(?:layers|blocks)\.(\d+)\.', name)
        if layer:
            key = f'layer_{int(layer[1]):03d}'
            groups[key] = groups.get(key, np.zeros(4)) + m
    for value in tensors.values():
        value['target_norm_share'] = value['target_norm'] ** 2 / totals[2] if totals[2] else None
    global_scores = _scores(totals)
    losses = {k: global_scores[k] for k in ['cosine', 'relative_l2']}
    equal = [v['cosine'] for v in tensors.values() if not v['zero_target']]
    losses['equal_tensor_cosine'] = (float(np.mean(equal)) if equal and
                                   all(v is not None for v in equal) else None)
    losses['centered_cosine'] = None
    if mean is not None:
        centered = sum((_moments(predicted[k].float() - mean[k].to(predicted[k].device),
                                 target[k].float() - mean[k].to(target[k].device))
                        for k in target), np.zeros(4))
        losses['centered_cosine'] = _scores(centered)['cosine']
    grouped = {key: _scores(value) for key, value in groups.items()}
    for key, value in grouped.items():
        value['target_norm_share'] = value['target_norm'] ** 2 / totals[2] if totals[2] else None
        for metric in ['cosine', 'relative_l2']:
            losses[f'{key}/{metric}'] = value[metric]
    return {'losses': losses, 'update': global_scores, 'groups': grouped, 'tensors': tensors}


def image_scores(truth, images, perceptual=None):
    result = image_metrics(truth[0], images[0])
    result['one_minus_ssim'] = 1 - result['ssim']
    if perceptual is not None:
        with torch.no_grad():
            result['lpips'] = float(perceptual(truth.float() * 2 - 1,
                                             images.float() * 2 - 1).item())
    return result


def objective(predicted, target, kind):
    if kind == 'cosine':
        return matching_loss(predicted, target, 'cosine')
    squared = matching_loss(predicted, target, 'l2')
    denom = sum(t.float().square().sum().to(squared.device) for t in target.values())
    if denom.detach().item() == 0:
        raise ValueError('Zero target update has no relative L2 objective')
    return squared / denom


def alignment(direction, delta):
    denom = direction.norm() * delta.norm()
    return float((direction.flatten() @ delta.flatten() / denom).item()) if denom > 0 else None


def direction_probe(adapter, batch, target, alpha, kind, perceptual=None):
    truth = batch.images.detach().float().cpu()
    start = perturb(truth, 'uniform_mix', alpha, 0).to(adapter.device).requires_grad_(True)
    loss = objective(replay(adapter, start, batch, True), target, kind)
    gradient, = torch.autograd.grad(loss, start)
    if not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite image derivative')
    baseline = image_scores(truth, start.detach().cpu(), perceptual)
    record = {'alpha': alpha, 'objective': kind, 'loss': float(loss.detach()),
              'image': baseline, 'zero_derivative': not bool(gradient.any()), 'steps': []}
    del loss
    for name, vector in [('gradient', -gradient), ('sign', -gradient.sign())]:
        vector = vector.detach()
        rms = vector.square().mean().sqrt()
        record[f'{name}_alignment'] = alignment(vector, truth.to(start.device) - start.detach())
        if rms == 0:
            continue
        vector = vector / rms
        for step in STEPS:
            moved = (start.detach() + step * vector).clamp(0, 1)
            actual = moved - start.detach()
            value = objective(replay(adapter, moved, batch), target, kind)
            metrics = image_scores(truth, moved.cpu(), perceptual)
            record['steps'].append({
                'direction': name, 'step_rms': step, 'actual_rms': float(actual.square().mean().sqrt()),
                'projected_alignment': alignment(actual, truth.to(start.device) - start.detach()),
                'quantized_unchanged': bool(torch.equal(moved.to(adapter.dtype),
                                                       start.detach().to(adapter.dtype))),
                'loss': float(value.detach()), 'image': metrics,
                'loss_decreased': float(value.detach()) < record['loss'],
                'mse_decreased': metrics['mse'] < baseline['mse']})
    return record


def rank_correlation(x, y):
    if len(x) < 3:
        return {'value': None, 'reason': 'fewer_than_three_candidates'}
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {'value': None, 'reason': 'constant_values'}
    return {'value': float(spearmanr(x, y).statistic), 'reason': None}


def candidate_statistics(rows, metric):
    valid = [r for r in rows if r['losses'].get(metric) is not None]
    result = {'count': len(valid), 'excluded': len(rows) - len(valid), 'spearman': {}}
    for image_metric in ['mse', 'one_minus_ssim', 'lpips']:
        selected = [r for r in valid if image_metric in r['image']]
        result['spearman'][image_metric] = rank_correlation(
            [r['losses'][metric] for r in selected], [r['image'][image_metric] for r in selected])
    near = [r['losses'][metric] for r in valid if r['image']['mse'] <= 1e-3]
    far = [r['losses'][metric] for r in valid if r['image']['mse'] >= 1e-2]
    pairs = np.subtract.outer(near, far)
    result['near_win'] = {'value': float(np.mean((pairs < 0) + .5 * (pairs == 0)))
                          if pairs.size else None, 'pairs': int(pairs.size),
                          'reason': None if pairs.size else 'missing_near_or_far'}

    def accepted(items):
        distances = [r['image']['mse'] for r in items]
        return {'count': len(items), 'far_fraction': float(np.mean(np.array(distances) >= 1e-2))
                if items else None, 'mse_median': float(np.median(distances)) if items else None,
                'mse_p90': float(np.quantile(distances, .9)) if items else None,
                'reason': None if items else 'no_candidates'}

    result['thresholds'] = {str(t): accepted([r for r in valid if r['losses'][metric] <= t])
                            for t in THRESHOLDS}
    result['bottom_fraction'] = {}
    ordered = sorted(valid, key=lambda r: r['losses'][metric])
    for fraction in [.01, .05, .1, .2]:
        # Include all boundary ties; never break quantization ties using image truth.
        boundary = ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]['losses'][metric] \
            if ordered else None
        selected = [r for r in ordered if r['losses'][metric] <= boundary]
        result['bottom_fraction'][str(fraction)] = accepted(selected)
    return result


def bootstrap(values, seed=42, repeats=2000):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {'mean': None, 'ci95': None, 'images': 0}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'images': len(values)}
