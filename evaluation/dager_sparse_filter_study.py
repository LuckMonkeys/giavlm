"""Longer, float64 sparse-filter ablation on already committed public systems."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from attacks.analytic.dager_filter_refinement import projected_system, sparse_support
from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.config import DAGEROptions
from evaluation.dager_filter_study import GPU5, TOPKS, _metrics


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('GPU 5 only')
    source, study, output = Path(args.scores), Path(args.study), Path(args.output)
    source_plan = read_json(source / 'plan.json')
    count = read_json(source / 'scores_complete.json')['sample_count']
    if output.exists():
        raise FileExistsError('Use a fresh sparse output')
    options = DAGEROptions(**source_plan['options'])
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'source_plan_sha256': file_hash(source / 'plan.json'),
        'lambda_ratios': [0.0001, 0.001, 0.005], 'iterations': 4000,
        'analysis_dtype': 'float64', 'victim_dtype_unchanged': True, 'pool_size': 2000,
        'rule': 'Active support first, then original baseline order for inactive and out-of-pool IDs.'})
    try:
        features = read_tensors(source / 'public_vocabulary_features.safetensors', 'cuda:0')['features']
        for index in range(count):
            for condition in ['known', 'random0']:
                write_json(output / 'progress.json', {'run': index, 'condition': condition})
                root = source / f'run-{index:05d}' / condition
                system = read_tensors(root / 'system.safetensors', 'cuda:0')
                saved = read_tensors(root / 'scores.safetensors')
                meta = read_json(root / 'complete.json')
                gradients = [system[f'gradient_{name}'] for name in 'qkv']
                projected, _, matrix, _, _, _, ambiguous = projected_system(
                    gradients, system['public_inputs'], features, options)
                forbidden = set(meta['forbidden_ids'])
                flags = ambiguous.cpu()
                allowed = [i for i in range(len(features)) if i not in forbidden and not flags[i]]
                baseline = saved['scores__baseline']
                pool = sorted(allowed, key=lambda i: (float(baseline[i]), i))[:2000]
                tensors, diagnostics = {'ambiguous': flags}, {}
                for penalty, (strength, diagnostic) in sparse_support(
                        projected, matrix, pool, lambda_ratios=(0.0001, 0.001, 0.005),
                        iterations=4000, analysis_dtype=torch.float64).items():
                    name = f'sparse64_{str(penalty).replace(".", "p")}'
                    value = 1 + baseline.clone()
                    strength = strength.cpu()
                    pool_tensor = torch.tensor(pool)
                    active = strength > 1e-7
                    value[pool_tensor[active]] = -strength[active]
                    tensors[f'scores__{name}'] = value
                    diagnostics[name] = diagnostic
                target = output / f'run-{index:05d}' / condition
                write_tensors(target / 'scores.safetensors', tensors)
                write_json(target / 'complete.json', {
                    'forbidden_ids': sorted(forbidden), 'diagnostics': diagnostics,
                    'score_sha256': file_hash(target / 'scores.safetensors')})
                print(f'committed {index}/{condition}: '
                      f'{[(k, v["steps"], v["converged"]) for k, v in diagnostics.items()]}', flush=True)
        write_json(output / 'scores_complete.json', {'sample_count': count})
        del features, projected, matrix, system
        torch.cuda.empty_cache()
        rows = []
        for index in range(count):
            refs = read_tensors(study / f'run-{index:05d}/captures/known/private/text_tokens.safetensors')
            for condition in ['known', 'random0']:
                root = output / f'run-{index:05d}' / condition
                meta = read_json(root / 'complete.json')
                tensors = read_tensors(root / 'scores.safetensors')
                forbidden = set(meta['forbidden_ids'])
                truth = set(refs['questions'].flatten().tolist() + refs['targets'].flatten().tolist()) - forbidden
                for key, values in tensors.items():
                    if key.startswith('scores__'):
                        rows.append({'run': index, 'condition': condition,
                                     'variant': key.removeprefix('scores__'),
                                     **_metrics(values, tensors['ambiguous'], forbidden, truth)})
        aggregate = []
        for condition, variant in sorted({(r['condition'], r['variant']) for r in rows}):
            selected = [r for r in rows if (r['condition'], r['variant']) == (condition, variant)]
            aggregate.append({'condition': condition, 'variant': variant,
                              'worst_true_rank_max': max(r['worst_true_rank'] for r in selected),
                              'curve': [{'k': k, 'complete': sum(r['curve'][j]['complete'] for r in selected),
                                         'mean_recall': float(np.mean([r['curve'][j]['recall'] for r in selected]))}
                                        for j, k in enumerate(TOPKS)]})
        write_json(output / 'report.json', {'aggregate': aggregate, 'rows': rows})
        write_json(output / 'complete.json', {'report_sha256': file_hash(output / 'report.json')})
    except Exception as error:
        write_json(output / 'failed.json', {'error_type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
