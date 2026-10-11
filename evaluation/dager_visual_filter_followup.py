"""Extended public-visual ridge range and frozen, fresh-cohort validation."""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path

import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import DAGEROptions, load_config
from core.data import load_batch, read_manifest
from core.fl import capture, restore_model, save_observation
from evaluation.dager_filter_study import GPU5
from evaluation.dager_visual_filter_study import evaluate, observation_system, score_system, variants


def extended_variants():
    result = [variants()[0]]
    for rank in (64, 256, 512):
        for ridge in (1e-5, 1e-7, 0.):
            result.append({'name': f'visual_r{rank}_tau{ridge}', 'context': 'visual',
                           'rank': rank, 'ridge': ridge, 'sparse': False})
    for rank, ridge in ((256, 1e-7), (512, 1e-7), (512, 0.)):
        for penalty in (.0001, .001, .01):
            result.append({'name': f'joint_visual_r{rank}_tau{ridge}_lambda{penalty}',
                           'context': 'visual', 'rank': rank, 'ridge': ridge,
                           'sparse': True, 'penalty': penalty})
    return result


def refine(args, output, options):
    bank_root, source = Path(args.bank), Path(args.source)
    specs = extended_variants()
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'variants': specs,
        'bank_sha256': file_hash(bank_root / 'public_visual_bank.safetensors'),
        'reason': 'Public covariance eigenvalue ratios at ranks 64/256/512 are 4.42e-4/1.94e-5/3.68e-6; extend ridge below those scales.',
        'cohort': 'Same development cohort; previous development results already inspected',
        'policy': 'All extended scores commit before evaluation. Fresh validation remains unused.',
        'pool': 'Original random-image top 2000; fixed across all joint fits',
        'iterations': 4000, 'analysis_dtype': 'float64'})
    bank = read_tensors(bank_root / 'public_visual_bank.safetensors', 'cuda:0')
    features = read_tensors(source / 'public_vocabulary_features.safetensors', 'cuda:0')['features']
    for index in range(10):
        write_json(output / 'progress.json', {'stage': 'extended_development', 'run': index})
        old = source / f'run-{index:05d}/random0'
        score_system(features, read_tensors(old / 'system.safetensors', 'cuda:0'), bank, options,
                     set(read_json(old / 'complete.json')['forbidden_ids']), specs,
                     output / f'run-{index:05d}',
                     read_tensors(old / 'scores.safetensors')['scores__baseline'])
    write_json(output / 'scores_complete.json', {'sample_count': 10, 'reference_access': False})
    evaluate(output, [Path(args.study) / f'run-{i:05d}/captures/known/private' for i in range(10)])


def selection_key(record):
    curves = {r['k']: r for r in record['curve']}
    return (curves[50]['complete'], curves[50]['mean_recall'],
            curves[100]['complete'], curves[100]['mean_recall'])


def select_frozen(args):
    all_specs = {s['name']: s for s in variants() + extended_variants()}
    reports = [read_json(Path(root) / 'report.json') for root in (args.bank, args.refinement)]
    records = {r['variant']: r for report in reports for r in report['aggregate']}
    selected = [all_specs['random_baseline']]
    # Include the independently tuned random surrogate as a control even when
    # the public visual bank loses. No parameters are chosen on validation.
    for context in ('random', 'visual'):
        names = [n for n in records if all_specs[n]['context'] == context and n != 'random_baseline']
        winner = max(sorted(names), key=lambda n: selection_key(records[n]))
        selected.append(all_specs[winner])
    return selected, {s['name']: records[s['name']] for s in selected}


def validate(args, output, options):
    specs, development = select_frozen(args)
    config = Path(args.study) / 'configs/known_residual.json'
    cfg = load_config(config)
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'variants': specs, 'development_selection': development,
        'bank_sha256': file_hash(Path(args.bank) / 'public_visual_bank.safetensors'),
        'development_report_hashes': [file_hash(Path(r) / 'report.json') for r in (args.bank, args.refinement)],
        'model_sha256': file_hash(Path(args.model) / 'model.safetensors'),
        'data_sha256': file_hash(args.data), 'sample_count': 10, 'split': 'eval', 'client': 0,
        'offset': 10, 'selection': 'Next ten unique image groups, excluded first ten used in previous filter validation',
        'iterations': 4000, 'analysis_dtype': 'float64', 'primary_k': [50, 100],
        'policy': 'Frozen before new captures. All 30 score conditions commit before private-token evaluation.'})
    rows = read_manifest(args.data, cfg.training.task, 'eval', 0, unique_images=True)
    selected_rows = rows[10:20]
    tune_ids = {r['image_id'] for r in read_manifest(args.data, cfg.training.task, 'tune', unique_images=True)}
    prior_ids = {r['image_id'] for r in rows[:10]}
    if len(selected_rows) != 10 or any(r['image_id'] in tune_ids | prior_ids for r in selected_rows):
        raise ValueError('Insufficient or overlapping fresh validation groups')
    adapter = restore_model(args.model, 'cuda:0')
    if asdict(adapter.spec) != asdict(replace(cfg.model, device='cuda:0')):
        raise ValueError('Victim model configuration changed')
    training = replace(cfg.training, knowledge='private', server_round=adapter.training_spec.server_round)
    adapter.set_training_spec(training)
    with torch.no_grad():
        features = adapter.backend.language_model.model.layers[0].input_layernorm(
            adapter.embedding().weight).detach()
    bank = read_tensors(Path(args.bank) / 'public_visual_bank.safetensors', 'cuda:0')
    forbidden = forbidden_token_ids(adapter)
    for index, row in enumerate(selected_rows):
        write_json(output / 'progress.json', {'stage': 'fresh_validation', 'run': index})
        root = output / f'run-{index:05d}'
        batch = load_batch([row], adapter)
        obs = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
        save_observation(root / 'capture/public', obs)
        write_tensors(root / 'capture/private/text_tokens.safetensors', {
            'questions': batch.questions, 'targets': batch.targets})
        write_json(root / 'capture/private/truth.json', {'samples': [row], 'training': asdict(training)})
        del batch
        system = observation_system(adapter, obs)
        write_tensors(root / 'system.safetensors', system)
        score_system(features, system, bank, options, forbidden, specs, root)
        del obs, system
    write_json(output / 'scores_complete.json', {'sample_count': 10, 'reference_access': False})
    evaluate(output, [output / f'run-{i:05d}/capture/private' for i in range(10)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('refine', 'validate'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bank', default='outputs/diagnostics/dager_visual_filter_gpu5_20261005')
    parser.add_argument('--refinement', default='outputs/diagnostics/dager_visual_filter_extended_gpu5_20261005')
    parser.add_argument('--source', default='outputs/diagnostics/dager_filter_gpu5_20261005_v2')
    parser.add_argument('--study', default='outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
    parser.add_argument('--model', default='outputs/federation/slake_llava_f_l_fedavg/round-0010')
    parser.add_argument('--data', default='data/slake/samples.jsonl')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('GPU 5 only')
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh output path')
    options = DAGEROptions(**read_json(Path(args.source) / 'plan.json')['options'])
    try:
        (refine if args.phase == 'refine' else validate)(args, output, options)
        print(f'{args.phase} completed', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
