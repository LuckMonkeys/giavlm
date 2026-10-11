"""Frozen sparse-filter validation on independent eval image groups.

Capture owns private batches; public_score receives only an Observation. All
candidate scores commit before this module joins private reference token IDs.
"""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path

import numpy as np
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids, gradient_groups
from attacks.analytic.dager_filter_refinement import projected_system, sparse_support, spectral_scores
from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.config import DAGEROptions, load_config
from core.data import load_batch, read_manifest
from core.fl import capture, restore_model, save_observation
from evaluation.dager_filter_study import GPU5, TOPKS, _metrics


@torch.no_grad()
def public_score(adapter, observation, features, output, options, random_image):
    from core.adapters.llava import LlavaTextView
    image = (torch.rand(1, 3, observation.model.image_size, observation.model.image_size,
                        generator=torch.Generator(device='cpu').manual_seed(0))
             if random_image else observation.public_images)
    if image is None:
        raise ValueError('Missing public image')
    view = LlavaTextView(adapter, replace(observation, public_images=image))
    gradients = [observation.tensors[n] for n in gradient_groups(adapter, observation, 'qkv')[0]]
    projected, _, matrix, _, vectors, _, ambiguous = projected_system(
        gradients, view.public_layer0, features, options)
    baseline = spectral_scores(projected, vectors).cpu()
    flags = ambiguous.cpu()
    forbidden = forbidden_token_ids(adapter)
    allowed = [i for i in range(len(features)) if i not in forbidden and not flags[i]]
    pool = sorted(allowed, key=lambda i: (float(baseline[i]), i))[:2000]
    strength, diagnostic = sparse_support(
        projected, matrix, pool, lambda_ratios=(0.0001,), iterations=4000,
        analysis_dtype=torch.float64)[0.0001]
    strength = strength.cpu()
    sparse = 1 + baseline.clone()
    active = strength > 1e-7
    sparse[torch.tensor(pool)[active]] = -strength[active]
    write_tensors(output / 'scores.safetensors', {
        'baseline': baseline, 'sparse64_0p0001': sparse, 'ambiguous': flags})
    write_json(output / 'complete.json', {'forbidden_ids': sorted(forbidden),
                                        'sparse_diagnostic': diagnostic,
                                        'score_sha256': file_hash(output / 'scores.safetensors')})


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('GPU 5 only')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Validation output exists')
    cfg = load_config(args.config)
    options = DAGEROptions(**read_json(args.config)['attack']['dager'])
    plan = {'schema_version': 1, 'source_sha256': source_fingerprint(),
            'sample_count': args.samples, 'split': 'eval', 'selection': 'first unique image groups, client 0',
            'data_sha256': file_hash(args.data),
            'model_sha256': file_hash(Path(args.model) / 'model.safetensors'),
            'options': asdict(options), 'lambda_ratio': 0.0001, 'sparse_pool': 2000,
            'iterations': 4000, 'analysis_dtype': 'float64', 'random_seed': 0,
            'primary_informative_k': [50, 100],
            'policy': 'Frozen from development; no validation-based retuning. Commit all scores before evaluation.'}
    write_json(output / 'plan.json', plan)
    try:
        adapter = restore_model(args.model, 'cuda:0')
        if asdict(adapter.spec) != asdict(replace(cfg.model, device='cuda:0')):
            raise ValueError('Victim model specification changed')
        adapter.set_training_spec(replace(cfg.training, server_round=adapter.training_spec.server_round))
        with torch.no_grad():
            features = adapter.backend.language_model.model.layers[0].input_layernorm(
                adapter.embedding().weight).detach()
        rows = read_manifest(args.data, cfg.training.task, 'eval', 0, unique_images=True)[:args.samples]
        tune_ids = {row['image_id'] for row in read_manifest(args.data, cfg.training.task, 'tune',
                                                           unique_images=True)}
        if len(rows) != args.samples or any(row['image_id'] in tune_ids for row in rows):
            raise ValueError('Insufficient or overlapping validation image groups')
        for index, row in enumerate(rows):
            write_json(output / 'progress.json', {'run': index, 'stage': 'capture_and_score'})
            root = output / f'run-{index:05d}'
            adapter.set_training_spec(replace(cfg.training, server_round=adapter.training_spec.server_round))
            batch = load_batch([row], adapter)
            obs = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
            observation_id = save_observation(root / 'capture/public', obs)
            private_obs = replace(obs, training=replace(obs.training, knowledge='private'), public_images=None)
            save_observation(root / 'private_image_capture/public', private_obs)
            write_tensors(root / 'capture/private/text_tokens.safetensors', {
                'questions': batch.questions, 'targets': batch.targets})
            write_tensors(root / 'capture/private/images.safetensors', {'images': batch.images})
            # Private source metadata remains confined to the private directory.
            write_json(root / 'capture/private/truth.json', {
                'observation_id': observation_id, 'training': asdict(obs.training),
                'samples': [{**row, 'model_question': adapter.decode(batch.questions)[0],
                             'model_target': adapter.decode(batch.targets)[0]}]})
            del batch
            public_score(adapter, obs, features, root / 'known', options, False)
            public_score(adapter, private_obs, features, root / 'random0', options, True)
            print(f'committed validation run {index}', flush=True)
        write_json(output / 'scores_complete.json', {'sample_count': len(rows)})
        del adapter, features, obs, private_obs
        torch.cuda.empty_cache()
        records = []
        for index in range(len(rows)):
            root = output / f'run-{index:05d}'
            references = read_tensors(root / 'capture/private/text_tokens.safetensors')
            for condition in ['known', 'random0']:
                meta = read_json(root / condition / 'complete.json')
                if file_hash(root / condition / 'scores.safetensors') != meta['score_sha256']:
                    raise ValueError('Validation score integrity failure')
                scores = read_tensors(root / condition / 'scores.safetensors')
                forbidden = set(meta['forbidden_ids'])
                by_field = {field: set(references[field].flatten().tolist()) - forbidden
                            for field in ['questions', 'targets']}
                truth = by_field['questions'] | by_field['targets']
                for variant in ['baseline', 'sparse64_0p0001']:
                    records.append({'run': index, 'condition': condition, 'variant': variant,
                                    **_metrics(scores[variant], scores['ambiguous'], forbidden, truth),
                                    'by_field': {field: _metrics(scores[variant], scores['ambiguous'], forbidden, ids)
                                                 for field, ids in by_field.items()}})
        aggregate = []
        for condition in ['known', 'random0']:
            for variant in ['baseline', 'sparse64_0p0001']:
                selected = [r for r in records if (r['condition'], r['variant']) == (condition, variant)]
                aggregate.append({'condition': condition, 'variant': variant,
                                  'worst_true_rank_max': max(r['worst_true_rank'] for r in selected),
                                  'curve': [{'k': k, 'complete': sum(r['curve'][j]['complete'] for r in selected),
                                             'mean_recall': float(np.mean([r['curve'][j]['recall'] for r in selected])),
                                             'mean_precision': float(np.mean([r['curve'][j]['precision'] for r in selected]))}
                                            for j, k in enumerate(TOPKS)]})
        write_json(output / 'report.json', {'sample_count': len(rows), 'aggregate': aggregate,
                                           'rows': records, 'frozen_plan_sha256': file_hash(output / 'plan.json')})
        write_json(output / 'complete.json', {'report_sha256': file_hash(output / 'report.json')})
        print('Frozen independent validation completed', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'error_type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
