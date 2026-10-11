"""Public-only filter ablations followed by separately committed evaluation.

The score phase never opens private references. This is a development study,
not a claim of independently validated improvement or original DAGER fidelity.
"""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import time

import numpy as np
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids, gradient_groups
from attacks.analytic.dager_filter_refinement import (
    adaptive_candidates, fuse_rankings, projected_system, sparse_support, spectral_scores,
)
from attacks.analytic.dager_subspace import SpanFilter, row_basis
from core.adapters.llava import LlavaTextView
from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.config import DAGEROptions
from core.fl import load_observation, restore_model


TOPKS = (6, 10, 20, 50, 100, 200, 500, 1000)
GPU5 = "GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d"


def _plan(args, options):
    return {"schema_version": 1, "source_sha256": source_fingerprint(),
            "options": asdict(options), "random_seeds": [0, 1, 2],
            "hard_ranks": [6, 8, 12, 16], "soft_relative_scales": [0.001, 0.01, 0.05],
            "sparse_pool": 2000, "sparse_lambda_ratios": [0.005, 0.02, 0.1],
            "sparse_iterations": 400, "topks": TOPKS, "gpu_uuid": GPU5,
            "sample_count": args.samples,
            "reference_policy": "All scores committed before private-reference evaluation.",
            "cohort": "existing ten-sample development cohort"}


@torch.no_grad()
def score_condition(adapter, obs, features, options, seed, directory, baseline_path=None):
    start = time.monotonic()
    images = obs.public_images if seed is None else torch.rand(
        1, 3, obs.model.image_size, obs.model.image_size,
        generator=torch.Generator(device="cpu").manual_seed(seed))
    if images is None:
        raise ValueError("Declared image missing")
    view = LlavaTextView(adapter, replace(obs, public_images=images))
    names = gradient_groups(adapter, obs, "qkv")[0]
    gradients = [obs.tensors[name] for name in names]
    projected, blocks, matrix, singular, vectors, ratio, ambiguous = projected_system(
        gradients, view.public_layer0, features, options)
    forbidden = forbidden_token_ids(adapter)
    allowed = [i for i in range(len(features)) if i not in forbidden and not ambiguous[i]]
    scores = {"baseline": spectral_scores(projected, vectors)}
    direct, original_ambiguous = SpanFilter.build(
        gradients, view.public_layer0, options).score(features)
    torch.testing.assert_close(ambiguous, original_ambiguous)
    delta = float((scores["baseline"][~ambiguous] - direct[~ambiguous]).abs().max())
    if delta > 1e-6:
        raise ValueError("Spectral implementation differs from the original span score")
    saved_delta = None
    if baseline_path is not None:
        prior = read_tensors(baseline_path)
        if not torch.equal(prior['ambiguous'], ambiguous.cpu()):
            raise ValueError("Baseline ambiguity flags changed")
        saved_delta = float((direct.cpu() - prior['scores']).abs().max())
        if saved_delta > 1e-6:
            raise ValueError(f"Baseline reproduction differs by {saved_delta}")
    for rank in (6, 8, 12, 16):
        scores[f"rank_{rank}"] = spectral_scores(projected, vectors[:rank])
    public_length = sum(obs.public_question_lengths) + sum(obs.public_target_lengths)
    scores['rank_public_length'] = spectral_scores(projected, vectors[:public_length])
    ratios = singular[:-1] / singular[1:].clamp_min(1e-30)
    gap_rank = int(ratios[1:].argmax()) + 2 if len(ratios) > 1 else len(singular)
    if len(ratios) < 2 or ratios[gap_rank - 1] < 3:
        gap_rank = len(singular)
    scores['rank_spectral_gap'] = spectral_scores(projected, vectors[:gap_rank])
    for scale in (0.001, 0.01, 0.05):
        weights = singular.square() / (singular.square() + (scale * singular[0]).square())
        scores[f"soft_{str(scale).replace('.', 'p')}"] = spectral_scores(projected, vectors, weights)
    scores['original_norm_residual'] = scores['baseline'] * ratio
    separate = []
    for name, block in zip('qkv', blocks, strict=True):
        basis, _ = row_basis(block, options.rank_rtol, options.rank_atol)
        scores[f'projection_{name}'] = spectral_scores(projected, basis)
        separate.append(scores[f'projection_{name}'])
    for name, value in fuse_rankings([scores['baseline'], *separate], allowed).items():
        scores[f'projection_fusion_{name}'] = value
    sparse_diagnostics = {}
    if seed in (None, 0):
        pool = sorted(allowed, key=lambda i: (float(scores['baseline'][i]), i))[:2000]
        for penalty, (strength, diagnostic) in sparse_support(projected, matrix, pool).items():
            label = f'sparse_{str(penalty).replace(".", "p")}'
            value = 1 + scores['baseline'].clone()
            value[pool] = -strength
            scores[label] = value
            sparse_diagnostics[label] = diagnostic
    # The gap policy is intentionally tested separately from each fixed-K score.
    gap_ids, gap_count = adaptive_candidates(scores['baseline'], ambiguous, forbidden)
    tensors = {f'scores__{name}': values.cpu() for name, values in scores.items()}
    tensors.update(ambiguous=ambiguous.cpu(), retained_norm_ratio=ratio.cpu())
    write_tensors(directory / 'scores.safetensors', tensors)
    write_tensors(directory / 'system.safetensors', {
        'public_inputs': view.public_layer0, 'gradient_q': gradients[0],
        'gradient_k': gradients[1], 'gradient_v': gradients[2],
        'singular_values': singular})
    metadata = {'seed': seed, 'forbidden_ids': sorted(forbidden), 'variants': list(scores),
                'score_sha256': file_hash(directory / 'scores.safetensors'),
                'baseline_formula_max_error': delta, 'saved_baseline_max_error': saved_delta,
                'public_length': public_length, 'spectral_gap_rank': gap_rank,
                'ambiguous_count': int(ambiguous.sum()), 'sparse_diagnostics': sparse_diagnostics,
                'adaptive_candidates': gap_ids, 'adaptive_informative_count': gap_count,
                'seconds': time.monotonic() - start}
    write_json(directory / 'complete.json', metadata)
    print(f'committed {directory.parent.name}/{directory.name} '
          f'{metadata["seconds"]:.1f}s; baseline delta={saved_delta}', flush=True)


def score(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5:
        raise ValueError('Expose physical GPU 5 UUID only')
    if args.device != 'cuda:0' or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible device cuda:0 required')
    output, study = Path(args.output), Path(args.study)
    if output.exists():
        raise FileExistsError('Use a fresh output directory; no silent retry')
    options = DAGEROptions(**read_json(study / 'configs/known_residual.json')['attack']['dager'])
    plan = _plan(args, options)
    plan['model_hashes'] = {name: file_hash(Path(args.model) / name)
                            for name in ['model.json', 'model.safetensors']}
    plan['public_observation_hashes'] = [
        {kind: {name: file_hash(study / f'run-{i:05d}' / 'captures' / kind / 'public' / name)
                for name in ['observation.json', 'update.safetensors']}
         for kind in ['known', 'private']} for i in range(args.samples)]
    write_json(output / 'plan.json', plan)
    try:
        adapter = restore_model(args.model, args.device)
        features = adapter.backend.language_model.model.layers[0].input_layernorm(
            adapter.embedding().weight).detach()
        write_tensors(output / 'public_vocabulary_features.safetensors', {'features': features})
        for index in range(args.samples):
            run = study / f'run-{index:05d}'
            for name, seed in [('known', None), ('random0', 0), ('random1', 1), ('random2', 2)]:
                write_json(output / 'progress.json', {'run': index, 'condition': name})
                capture = run / 'captures' / ('known' if seed is None else 'private') / 'public'
                obs = load_observation(capture, args.device)
                if obs.model_fingerprint != adapter.fingerprint():
                    raise ValueError('Model fingerprint mismatch')
                adapter.set_training_spec(obs.training)
                baseline = None
                if seed in (None, 0):
                    old = 'known_residual' if seed is None else 'random_residual'
                    baseline = run / 'attacks' / old / 'token_candidates.safetensors'
                score_condition(adapter, obs, features, options, seed,
                                output / f'run-{index:05d}' / name, baseline)
        write_json(output / 'scores_complete.json', {'sample_count': args.samples,
                                                    'source_sha256': plan['source_sha256']})
    except Exception as error:
        write_json(output / 'failed.json', {'error_type': type(error).__name__, 'reason': str(error)})
        raise


def _metrics(scores, ambiguous, forbidden, truth):
    values = scores.detach().cpu().numpy()
    flags = ambiguous.detach().cpu().numpy()
    valid = np.ones(len(values), dtype=bool)
    valid[list(forbidden)] = False
    allowed = np.flatnonzero(valid & ~flags)
    order = allowed[np.lexsort((allowed, values[allowed]))].tolist()
    uncertain = set(np.flatnonzero(valid & flags).tolist())
    curve = []
    for k in TOPKS:
        selected = set(order[:k]) | uncertain
        curve.append({'k': k, 'count': len(selected), 'recall': len(selected & truth) / len(truth),
                      'precision': len(selected & truth) / len(selected), 'complete': truth <= selected})
    positions = {i: k + 1 for k, i in enumerate(order)}
    worst = max([positions.get(i, len(scores)) for i in truth - uncertain], default=0)
    return {'curve': curve, 'worst_true_rank': worst}


def commit_ensembles(output, count):
    """Commit every surrogate fusion score before evaluation reads references."""
    for index in range(count):
        root = output / f'run-{index:05d}'
        conditions = {name: read_tensors(root / name / 'scores.safetensors')
                      for name in ['random0', 'random1', 'random2']}
        forbidden = set(read_json(root / 'random0/complete.json')['forbidden_ids'])
        flags = conditions['random0']['ambiguous']
        if not all(torch.equal(flags, conditions[name]['ambiguous'])
                   for name in ['random1', 'random2']):
            raise ValueError('Surrogate ambiguity sets differ; explicit union policy required')
        allowed = [i for i in range(len(flags)) if i not in forbidden and not flags[i]]
        tensors = {'ambiguous': flags}
        for variant in ['baseline', 'rank_public_length', 'soft_0p001', 'soft_0p01']:
            vectors = [conditions[c][f'scores__{variant}'] for c in ['random0', 'random1', 'random2']]
            for fusion, values in fuse_rankings(vectors, allowed).items():
                tensors[f'scores__{variant}_{fusion}'] = values
        write_tensors(root / 'random_ensemble/scores.safetensors', tensors)
        write_json(root / 'random_ensemble/complete.json', {
            'forbidden_ids': sorted(forbidden),
            'score_sha256': file_hash(root / 'random_ensemble/scores.safetensors')})
    write_json(output / 'ensemble_scores_complete.json', {'sample_count': count,
                                                        'reference_access': False})


def evaluate(args):
    """Join references only after all planned score vectors are committed."""
    output, study = Path(args.output), Path(args.study)
    count = read_json(output / 'scores_complete.json')['sample_count']
    for index in range(count):
        for condition in ['known', 'random0', 'random1', 'random2']:
            directory = output / f'run-{index:05d}' / condition
            meta = read_json(directory / 'complete.json')
            if file_hash(directory / 'scores.safetensors') != meta['score_sha256']:
                raise ValueError('Score artifact hash mismatch')
    commit_ensembles(output, count)
    rows = []
    for index in range(count):
        run = study / f'run-{index:05d}'
        references = read_tensors(run / 'captures/known/private/text_tokens.safetensors')
        conditions = {}
        for condition in ['known', 'random0', 'random1', 'random2']:
            directory = output / f'run-{index:05d}' / condition
            meta = read_json(directory / 'complete.json')
            tensors = read_tensors(directory / 'scores.safetensors')
            forbidden = set(meta['forbidden_ids'])
            truth = set(references['questions'].flatten().tolist()
                        + references['targets'].flatten().tolist()) - forbidden
            conditions[condition] = (tensors, meta)
            for variant in meta['variants']:
                metrics = _metrics(tensors[f'scores__{variant}'], tensors['ambiguous'], forbidden, truth)
                rows.append({'run': index, 'condition': condition, 'variant': variant, **metrics})
            selected = set(meta['adaptive_candidates'])
            rows.append({'run': index, 'condition': condition, 'variant': 'adaptive_gap',
                         'adaptive': {'count': len(selected), 'complete': truth <= selected,
                                      'recall': len(selected & truth) / len(truth),
                                      'precision': len(selected & truth) / len(selected)}})
        ensemble = read_tensors(output / f'run-{index:05d}/random_ensemble/scores.safetensors')
        for name, values in ensemble.items():
            if name.startswith('scores__'):
                rows.append({'run': index, 'condition': 'random_ensemble',
                             'variant': name.removeprefix('scores__'),
                             **_metrics(values, ensemble['ambiguous'], forbidden, truth)})
    aggregate = []
    for condition, variant in sorted({(r['condition'], r['variant']) for r in rows}):
        samples = [r for r in rows if (r['condition'], r['variant']) == (condition, variant)]
        record = {'condition': condition, 'variant': variant, 'sample_count': len(samples)}
        if variant == 'adaptive_gap':
            record['adaptive'] = {'complete': sum(r['adaptive']['complete'] for r in samples),
                                  'mean_count': float(np.mean([r['adaptive']['count'] for r in samples])),
                                  'mean_recall': float(np.mean([r['adaptive']['recall'] for r in samples]))}
        else:
            record['curve'] = [{'k': k, 'complete': sum(r['curve'][j]['complete'] for r in samples),
                                'mean_recall': float(np.mean([r['curve'][j]['recall'] for r in samples])),
                                'mean_precision': float(np.mean([r['curve'][j]['precision'] for r in samples]))}
                               for j, k in enumerate(TOPKS)]
            record['worst_true_rank_max'] = max(r['worst_true_rank'] for r in samples)
        aggregate.append(record)
    write_json(output / 'report.json', {'schema_version': 1, 'aggregate': aggregate, 'rows': rows,
                                       'evaluation_source_sha256': source_fingerprint(),
                                       'scope': 'Development ablation; no independent validation yet.'})
    write_json(output / 'complete.json', {'report_sha256': file_hash(output / 'report.json')})
    print('Evaluation complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['score', 'evaluate'], required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--model')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'score':
        score(args)
    else:
        try:
            evaluate(args)
        except Exception as error:
            write_json(Path(args.output) / 'failed.json', {
                'stage': 'evaluation', 'error_type': type(error).__name__, 'reason': str(error)})
            raise


if __name__ == '__main__':
    main()
