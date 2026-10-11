"""Frozen token-evidence families on twenty unused private-image groups."""
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
from evaluation.dager_evidence_study import (commit_shuffled, names_for, public_features,
                                             save_scores, scores_from)
from evaluation.dager_filter_study import GPU5
from evaluation.dager_visual_filter_study import evaluate


def freeze_families(source):
    report = read_json(source / 'report.json')
    rows = {r['variant']: r for r in report['aggregate']}

    def key(name):
        curve = {c['k']: c for c in rows[name]['curve']}
        return (curve[50]['complete'], curve[50]['mean_recall'],
                curve[100]['complete'], curve[100]['mean_recall'])

    families = {
        'A_calibration': [n for n in rows if n.startswith(('null_', 'A_'))],
        'AB_calibration': [n for n in rows if n.startswith('AB_')],
        'token_groups': [n for n in rows if n.startswith('group_')]}
    winners = {family: max(sorted(names), key=key) for family, names in families.items()}
    return winners, {name: rows[name] for name in ['baseline', *winners.values()]}


def run(args):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh validation output')
    if not (source / 'complete.json').exists():
        raise ValueError('Development must complete before freezing validation')
    winners, development = freeze_families(source)
    selected = {'baseline', 'public_frequency', 'A_0.01', 'AB_0.01', 'B_0.01', *winners.values()}
    config = Path(args.study) / 'configs/known_residual.json'
    cfg = load_config(config)
    options = DAGEROptions(**read_json(config)['attack']['dager'])
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'winners': winners,
        'development_selection': development, 'variants': sorted(selected),
        'public_calibration_plan_sha256': file_hash(source / 'public_calibration_plan.json'),
        'calibration_sha256': file_hash(source / 'calibration.safetensors'),
        'development_report_sha256': file_hash(source / 'report.json'),
        'model_sha256': file_hash(Path(args.model) / 'model.safetensors'),
        'data_sha256': file_hash(args.data), 'split': 'eval', 'client': 0,
        'offset': args.offset, 'sample_count': 20, 'primary_k': [50, 100],
        'shuffle_seed': args.shuffle_seed,
        'cohort_selection': 'Unique image groups after offset; optional fixed permutation, no text-based selection.',
        'policy': 'Family winners frozen before captures, all scores and shuffled controls saved before evaluation; no retuning.'})
    try:
        rows = read_manifest(args.data, cfg.training.task, 'eval', 0, unique_images=True)
        available = rows[args.offset:]
        if args.shuffle_seed is not None:
            order = torch.randperm(len(available), generator=torch.Generator().manual_seed(args.shuffle_seed)).tolist()
            available = [available[i] for i in order]
        selected_rows = available[:20]
        excluded = {r['image_id'] for r in rows[:args.offset]}
        excluded |= {r['image_id'] for r in read_manifest(args.data, cfg.training.task, 'tune', unique_images=True)}
        if len(selected_rows) != 20 or any(r['image_id'] in excluded for r in selected_rows):
            raise ValueError('Overlapping or insufficient validation image groups')
        adapter = restore_model(args.model, 'cuda:0')
        if asdict(adapter.spec) != asdict(replace(cfg.model, device='cuda:0')):
            raise ValueError('Victim configuration mismatch')
        training = replace(cfg.training, knowledge='private', server_round=adapter.training_spec.server_round)
        adapter.set_training_spec(training)
        with torch.no_grad():
            features = adapter.backend.language_model.model.layers[0].input_layernorm(adapter.embedding().weight).detach()
        calibration = read_tensors(source / 'calibration.safetensors')
        models = {p.stem.removeprefix('model_'): read_tensors(p) for p in sorted(source.glob('model_*.safetensors'))}
        groups = read_tensors(source / 'groups.safetensors')['groups']
        names, forbidden = names_for(adapter), forbidden_token_ids(adapter)
        for index, row in enumerate(selected_rows):
            write_json(output / 'progress.json', {'run': index, 'stage': 'capture_and_public_score'})
            root = output / f'run-{index:05d}'
            batch = load_batch([row], adapter)
            obs = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
            save_observation(root / 'capture/public', obs)
            write_tensors(root / 'capture/private/text_tokens.safetensors', {
                'questions': batch.questions, 'targets': batch.targets})
            write_json(root / 'capture/private/truth.json', {'samples': [row], 'training': asdict(training)})
            del batch
            values, audit = public_features(adapter, obs, features, options, names)
            full = scores_from(values['raw'], values['baseline'], values['ambiguous'], calibration,
                               models, groups, forbidden)
            write_tensors(root / 'features.safetensors', values)
            save_scores(root, {n: full[n] for n in selected}, values['ambiguous'], forbidden, audit)
            del obs, full
            print(f'fresh validation {index+1}/20 saved', flush=True)
        commit_shuffled(output, 20)
        write_json(output / 'scores_complete.json', {'sample_count': 20})
        evaluate(output, [output / f'run-{i:05d}/capture/private' for i in range(20)])
        print('Frozen evidence validation complete', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='outputs/diagnostics/dager_evidence_gpu5_20261005_v5')
    parser.add_argument('--study', default='outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
    parser.add_argument('--model', default='outputs/federation/slake_llava_f_l_fedavg/round-0010')
    parser.add_argument('--data', default='data/slake/samples.jsonl')
    parser.add_argument('--offset', type=int, default=20)
    parser.add_argument('--shuffle-seed', type=int)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('GPU 5 only')
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
