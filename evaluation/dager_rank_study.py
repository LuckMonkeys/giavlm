"""Paired LoRA-rank study using the unchanged DAGER reconstruction attack.

Train native F-L snapshots first, then capture and reconstruct with one loaded
adapter per rank. Only evaluation joins committed reconstructions with truth.
"""
import argparse
from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from attacks.factory import create_attacker
from core.artifacts import (environment, file_hash, read_json,
                            source_fingerprint, write_json, write_tensors)
from core.config import ModelSpec, TrainingSpec, digest, load_config
from core.data import load_batch, read_manifest
from core.fl import capture, load_observation, restore_model, save_observation
from core.knowledge import AdversaryKnowledge
from evaluation.dager_reconstruction_study import CONDITIONS, _protocol, evaluate_committed


GPU5 = 'GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d'
BASELINE = Path('outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
BASE_MODEL = Path('outputs/federation/slake_llava_f_l_fedavg')
DATA = Path('data/slake/samples.jsonl')


def rank_configs(base, model_metadata, rank):
    """Preserve alpha/rank=2 and every other train/capture setting."""
    if rank not in (16, 32):
        raise ValueError('This frozen comparison supports ranks 16 and 32')
    training = TrainingSpec(**model_metadata['training'])
    if (training.lora_rank, training.lora_alpha) != (8, 16):
        raise ValueError('Unexpected rank-8 reference scaling')
    training = replace(training, lora_rank=rank, lora_alpha=2*rank, server_round=0)
    model = replace(ModelSpec(**model_metadata['model']), device='cuda:0')
    trained = replace(base, model=model, training=training)
    attacked = replace(base, model=model, training=replace(
        base.training, lora_rank=rank, lora_alpha=2*rank))
    return trained, attacked


def prepare(output):
    if output.exists():
        raise FileExistsError('Use a fresh rank-study output')
    base = load_config(BASELINE/'configs/known_residual.json')
    metadata = read_json(BASE_MODEL/'round-0010/model.json')
    reference = read_json(BASELINE/'plan.json')
    if file_hash(DATA) != reference['data_sha256']:
        raise ValueError('Dataset differs from the rank-8 study')
    for name, expected in reference['model_sha256'].items():
        if file_hash(BASE_MODEL/'round-0010'/name) != expected:
            raise ValueError('Rank-8 snapshot differs from the baseline study')
    for rank in (16, 32):
        trained, attacked = rank_configs(base, metadata, rank)
        write_json(output/'configs'/f'train-rank{rank}.json', trained)
        write_json(output/'configs'/f'attack-rank{rank}.json', attacked)
    write_json(output/'plan.json', {
        'schema_version': 1, 'ranks': [8, 16, 32], 'new_ranks': [16, 32],
        'alpha_per_rank': 2, 'samples': 10, 'split': 'tune', 'offsets': list(range(10)),
        'conditions': CONDITIONS, 'candidates': 100, 'beam_width': 16,
        'prefix_batch_size': 32, 'max_evaluations': 200000, 'seconds': 3600.,
        'source_sha256': source_fingerprint(), 'data_sha256': file_hash(DATA),
        'baseline_summary_sha256': file_hash(BASELINE/'summary.json'),
        'model_policy': 'Native fresh initialization per rank, model seed 17; alpha/rank=2; same training seed/data/steps.',
        'reference_policy': 'No reference reads for attack; evaluate after every new-rank attack commits.',
        'config_sha256': {p.name: file_hash(p) for p in sorted((output/'configs').glob('*.json'))}})


def capture_pair(adapter, row, known_training, root, model):
    """Victim-side capture; private data never enter the attack interface."""
    adapter.set_training_spec(known_training)
    batch = load_batch([row], adapter)
    known = capture(adapter, batch, [row['question']], [row['target']])
    private = replace(known, training=replace(known.training, knowledge='private'), public_images=None)
    private.validate()
    record = dict(row, model_question=adapter.decode(batch.questions)[0],
                  model_target=adapter.decode(batch.targets)[0])
    for name, obs in [('known', known), ('private', private)]:
        folder = root/'captures'/name
        observation_id = save_observation(folder/'public', obs)
        write_json(folder/'public/model_ref.json', {'path': str(model.resolve())})
        write_tensors(folder/'private/images.safetensors', {'images': batch.images})
        write_tensors(folder/'private/text_tokens.safetensors', {
            'questions': batch.questions, 'targets': batch.targets})
        write_json(folder/'private/truth.json', {'observation_id': observation_id,
                    'training': asdict(obs.training), 'samples': [record]})
    if read_json(root/'captures/known/public/observation.json')['update_sha256'] != read_json(
            root/'captures/private/public/observation.json')['update_sha256']:
        raise ValueError('Paired uploads differ')


def attack_public(adapter, config, public, output, fingerprint):
    """Accept public observation only; call the registered, unchanged attacker."""
    obs = load_observation(public, adapter.spec.device)
    if obs.model_fingerprint != fingerprint:
        raise ValueError('Observed model differs from loaded snapshot')
    adapter.set_training_spec(obs.training)
    meta = read_json(public/'observation.json')
    versions = {name: parameter._version for name, parameter in adapter.named_parameters()}
    result = create_attacker(adapter, config.attack).attack(
        obs.tensors, obs, AdversaryKnowledge(obs.training.knowledge, True),
        directory=output, resume=False, upload_metadata={})
    if result.images is not None:
        raise ValueError('Text-only DAGER must not return a private image')
    info = {k: v for k, v in asdict(result).items() if k != 'images'}
    info.update(observation_id=meta['observation_id'],
                run_signature=digest({'observation_id': meta['observation_id'],
                    'attack': asdict(config.attack), 'source_sha256': source_fingerprint()}),
                condition={**asdict(obs.training), 'model': obs.model.name,
                    'revision': obs.model.revision, 'model_state': fingerprint,
                    'method': config.attack.method, 'seed': config.attack.seed,
                    'source_sha256': source_fingerprint(), 'init_source': 'random',
                    'init_text_source': 'random', 'attack_image_dtype': config.attack.image_dtype},
                environment=environment())
    write_json(output/'result.json', info)
    if any(parameter._version != versions[name] for name, parameter in adapter.named_parameters()):
        raise ValueError('Read-only text attack mutated victim model')
    return result.status, result.costs


def reconstruct(output, rank):
    root = output/f'rank-{rank}'
    model = output/'models'/f'rank-{rank}'/'round-0010'
    base = load_config(output/'configs'/f'attack-rank{rank}.json')
    configs = {name: _protocol(base, *settings, 100, 16, 200000, 3600., 0)
               for name, settings in CONDITIONS.items()}
    write_json(root/'plan.json', {'source_sha256': source_fingerprint(),
               'rank': rank, 'model_sha256': file_hash(model/'model.safetensors'),
               'parent_plan_sha256': file_hash(output/'plan.json')})
    for name, config in configs.items():
        write_json(root/'configs'/f'{name}.json', config)
    adapter = restore_model(model, 'cuda:0')
    fingerprint = adapter.fingerprint()
    training = replace(configs['known_residual'].training, server_round=10)
    rows = read_manifest(DATA, training.task, 'tune', 0, unique_images=True)[:10]
    if len(rows) != 10:
        raise ValueError('Missing paired evaluation samples')
    for index, row in enumerate(rows):
        write_json(output/'progress.json', {'stage': 'capture', 'rank': rank, 'run': index})
        capture_pair(adapter, row, training, root/f'run-{index:05d}', model)
        print(f'rank {rank}: capture {index+1}/10 saved', flush=True)
    del rows, row
    gc.collect()
    torch.cuda.empty_cache()
    for index in range(10):
        for name, (knowledge, _, _) in CONDITIONS.items():
            write_json(output/'progress.json', {'stage': 'attack', 'rank': rank,
                                                'run': index, 'condition': name})
            run_dir = root/f'run-{index:05d}'
            public = run_dir/'captures'/('known' if knowledge == 'image_known' else 'private')/'public'
            status, costs = attack_public(adapter, configs[name], public,
                                         run_dir/'attacks'/name, fingerprint)
            print(f'rank {rank}: run {index+1}/10 {name}: {status}; {costs}', flush=True)
            gc.collect()
    if adapter.fingerprint() != fingerprint:
        raise ValueError('Model state changed during reconstruction phase')
    write_json(root/'attacks_complete.json', {'rank': rank, 'attacks': 40})


def evaluate_all(output):
    for rank in (16, 32):
        if not (output/f'rank-{rank}'/'attacks_complete.json').exists():
            raise ValueError('Every rank must finish before evaluation')
    summaries = {'8': read_json(BASELINE/'summary.json')}
    paired = []
    for rank in (16, 32):
        root = output/f'rank-{rank}'
        for index in range(10):
            for filename in ('images.safetensors', 'text_tokens.safetensors'):
                relative = Path(f'run-{index:05d}')/'captures/known/private'/filename
                if file_hash(root/relative) != file_hash(BASELINE/relative):
                    raise ValueError('Rank comparison selected different samples')
        paths = {name: root/'configs'/f'{name}.json' for name in CONDITIONS}
        summaries[str(rank)] = evaluate_committed(root, paths, 100)
        write_json(root/'complete.json', {'summary_sha256': file_hash(root/'summary.json')})
        paired.append({'rank': rank, 'samples': 10, 'images_and_text_identical_to_rank8': True})
    # Aggregate metrics only: no true text, sample identifiers, or generated text.
    clean = {rank: {name: {k: v for k, v in row.items() if k != 'reconstructions'}
                    for name, row in summary['conditions'].items()}
             for rank, summary in summaries.items()}
    write_json(output/'comparison.json', {'sample_count': 10, 'conditions_by_rank': clean,
               'paired_reference_audit': paired, 'source_sha256': source_fingerprint()})


def run(output):
    prepare(output)
    started = time.monotonic()
    try:
        for rank in (16, 32):
            write_json(output/'progress.json', {'stage': 'train', 'rank': rank})
            print(f'Training rank {rank} from native round zero', flush=True)
            subprocess.run([sys.executable, '-m', 'core.commands', 'train', '--data', str(DATA),
                            '--config', str(output/'configs'/f'train-rank{rank}.json'),
                            '--output', str(output/'models'/f'rank-{rank}')], check=True)
        write_json(output/'models_complete.json', {'ranks': [16, 32]})
        for rank in (16, 32):
            subprocess.run([sys.executable, '-m', 'evaluation.dager_rank_study',
                            '--output', str(output), '--stage', 'reconstruct',
                            '--rank', str(rank)], check=True)
        write_json(output/'progress.json', {'stage': 'evaluation'})
        evaluate_all(output)
        write_json(output/'complete.json', {'seconds': time.monotonic()-started,
                    'comparison_sha256': file_hash(output/'comparison.json')})
        write_json(output/'progress.json', {'stage': 'completed'})
    except Exception as error:
        write_json(output/'failed.json', {'error_type': type(error).__name__, 'reason': str(error),
                    'progress': read_json(output/'progress.json')})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('all', 'reconstruct'), default='all')
    parser.add_argument('--rank', type=int, choices=(16, 32))
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Expose only authorized GPU5')
    torch.set_num_threads(1)
    output = Path(args.output)
    if args.stage == 'all':
        run(output)
    else:
        reconstruct(output, args.rank)


if __name__ == '__main__':
    main()
