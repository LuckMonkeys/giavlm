"""Image-ordering diagnostic with already committed LM-prior reconstructions.

The only image score is gradient matching. The prior never enters this score.
Truth constructs diagnostic image banks and oracle text controls in evaluation.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import torch

from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.fl import load_observation, restore_model, simulate_update
from core.types import Batch
from evaluation.dager_lm_prior_study import study_inputs
from evaluation.dager_rank_study import GPU5
from evaluation.discriminability_metrics import bootstrap
from evaluation.reconstructed_text_discriminability import FIELDS, evaluate, text_audit
from evaluation.text_unknown_discriminability import candidate_bank, load_spec
from evaluation.token_image_ordering import GradientScorer


PRIOR32 = Path('outputs/diagnostics/dager_lm_prior_gpu5_20261011')
PRIOR8 = Path('outputs/diagnostics/dager_lm_prior_rank8_gpu5_20261012')
OLD = Path('outputs/diagnostics/reconstructed_text_gpu5_20261011_v2')


def inputs_for(rank, phase):
    if rank not in (8, 32) or phase not in ('development', 'validation'):
        raise ValueError('Unsupported cohort')
    if rank == 8 and phase == 'validation':
        raise ValueError('No committed rank-8 validation reconstructions')
    original, model = study_inputs(rank)
    prior = (PRIOR8 if rank == 8 else PRIOR32)/phase
    return (original if phase == 'development' else prior), model, prior


def validate_result(result, observation, rank, weight):
    options = result['provenance']['options']
    condition = result['condition']
    if (result['status'] != 'completed' or result['observation_id'] != observation['observation_id']
            or options['lm_prior_weight'] != weight or options['mode'] != 'public_residual'
            or options['image_source'] != 'random' or condition['lora_rank'] != rank
            or condition['knowledge'] != 'private' or not condition['token_lengths_known']):
        raise ValueError('Reconstruction source, prior weight, rank or knowledge differs')


def text_conditions(reference, baseline, prior):
    return [('oracle', reference), ('baseline', baseline), ('lm_prior', prior),
            ('prior_question', {'questions': prior['questions'], 'targets': reference['targets']}),
            ('prior_answer', {'questions': reference['questions'], 'targets': prior['targets']})]


def input_files(src, attack_root):
    paths = {f'{name}_{kind}': attack_root/'attacks'/weight/filename
             for name, weight in [('baseline', 'weight-0p0'), ('lm_prior', 'weight-0p01')]
             for kind, filename in [('result', 'result.json'), ('tokens', 'text_tokens.safetensors')]}
    paths.update({'observation': src/'captures/private/public/observation.json',
                  'update': src/'captures/private/public/update.safetensors',
                  'reference_tokens': src/'captures/known/private/text_tokens.safetensors',
                  'reference_image': src/'captures/known/public/public_images.safetensors'})
    return paths


def finish_report(output):
    evaluate(output)
    report = read_json(output/'report.json')
    paired = []
    for group in ('all', 'lora_A', 'lora_B', 'first_qkv_A'):
        for loss in ('cosine', 'relative_l2'):
            rows = [r for r in report['rows'] if r['group'] == group and r['loss'] == loss]
            prior = sorted([r for r in rows if r['family'] == 'lm_prior'], key=lambda r: r['run'])
            baseline = sorted([r for r in rows if r['family'] == 'baseline'], key=lambda r: r['run'])
            paired.append({'group': group, 'loss': loss, **{
                metric: bootstrap([p[metric]-b[metric] for p, b in zip(prior, baseline)], seed=20261012)
                for metric in ('rho', 'near_win', 'truth_percentile')}})
    write_json(output/'paired_prior_gain.json', {'paired': paired})
    write_json(output/'complete.json', {'report_sha256': file_hash(output/'report.json'),
               'paired_sha256': file_hash(output/'paired_prior_gain.json')})


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic output; no automatic retry')
    started = time.monotonic()
    try:
        source, model, prior_root = inputs_for(args.rank, args.phase)
        prior_plan = read_json(prior_root/'plan.json')
        completion = read_json(prior_root/'complete.json')
        if completion['summary_sha256'] != file_hash(prior_root/'summary.json'):
            raise ValueError('Committed reconstruction summary changed')
        if prior_plan['model_sha256'] != file_hash(model/'model.safetensors'):
            raise ValueError('Prior experiment used a different model')
        spec = load_spec('configs/diagnostics/slake_llava_text_unknown_discriminability.yaml')
        signatures = {}
        for i in range(10):
            files = input_files(source/f'run-{i:05d}', prior_root/f'run-{i:05d}')
            obs = read_json(files['observation'])
            for name, weight in [('baseline', 0.), ('lm_prior', .01)]:
                validate_result(read_json(files[f'{name}_result']), obs, args.rank, weight)
            signatures[str(i)] = {label: file_hash(path) for label, path in files.items()}
        fingerprint = source_fingerprint()
        write_json(output/'plan.json', {
            'schema_version': 1, 'uses_private_reference': True, 'not_an_attack_benchmark': True,
            'rank': args.rank, 'phase': args.phase, 'samples': 10,
            'source_study': str(source), 'reconstruction_study': str(prior_root),
            'source_sha256': fingerprint, 'prior_plan_sha256': file_hash(prior_root/'plan.json'),
            'model_sha256': {f: file_hash(model/f) for f in ('model.json', 'model.safetensors')},
            'input_sha256': signatures, 'prior_weights': [0., .01],
            'candidate_ids': spec['candidate_ids'], 'physical_gpu': 5,
            'conditions': ['oracle', 'baseline', 'lm_prior', 'prior_question', 'prior_answer'],
            'primary': 'All uploaded gradients cosine loss; Spearman and near-win on 24 nontruth images',
            'secondary': 'Gradient blocks, relative L2, truth percentile, text correctness subgroups',
            'selection': 'All ten committed runs; no selection by text quality or image score',
            'prior_policy': 'Only previously reconstructed text changes; no LM prior added to image score',
            'reference_policy': 'Oracle text and image bank are private-reference diagnostic controls',
            'validation_round_note': 'Keep original observation metadata; model hash and exact replay verify actual snapshot'})
        adapter = restore_model(model, 'cuda:0')
        model_fingerprint = adapter.fingerprint()
        torch.cuda.reset_peak_memory_stats()
        evaluations, qa, legacy = 0, [], []
        for i in range(10):
            if source_fingerprint() != fingerprint:
                raise ValueError('Diagnostic code changed during execution')
            root = output/f'run-{i:05d}'
            src, attack_root = source/f'run-{i:05d}', prior_root/f'run-{i:05d}'
            files = input_files(src, attack_root)
            if signatures[str(i)] != {label: file_hash(path) for label, path in files.items()}:
                raise ValueError('Frozen source artifacts changed')
            obs = load_observation(src/'captures/private/public', 'cuda:0')
            if obs.model_fingerprint != model_fingerprint or obs.sample_count != 1 or obs.training.algorithm != 'fedsgd':
                raise ValueError('Observation incompatible with model')
            adapter.set_training_spec(obs.training)
            truth = read_tensors(files['reference_image'])['images'].float()
            reference = read_tensors(files['reference_tokens'], 'cuda:0')
            baseline = read_tensors(files['baseline_tokens'], 'cuda:0')
            prior = read_tensors(files['lm_prior_tokens'], 'cuda:0')
            lengths = dict(zip(FIELDS, [obs.public_question_lengths[0], obs.public_target_lengths[0]]))
            qa.append(tuple(tuple(reference[f][0, :lengths[f]].tolist()) for f in FIELDS))
            cases = text_conditions(reference, baseline, prior)
            write_json(root/'text_audit.json', {name: text_audit(reference, text, lengths) for name, text in cases})
            write_tensors(root/'private/conditions.safetensors', {
                f'{name}.{field}': text[field] for name, text in cases for field in FIELDS})
            bank = candidate_bank(spec, truth)
            distances = {name: float((image-truth).square().mean()) for name, _, image in bank}
            scorer = GradientScorer(obs.tensors, 'cuda:0')
            replay = simulate_update(adapter, Batch(truth.to('cuda:0', adapter.dtype),
                                     reference['questions'], reference['targets']), obs.training)
            error = scorer.score(replay)['all']['relative_l2']
            del replay
            if error > 1e-12:
                raise ValueError('Victim gradient replay differs')
            write_json(root/'replay.json', {'relative_l2': error, 'capture_round': obs.training.server_round})
            cache, mapping = {}, []
            for j, (name, text) in enumerate(cases):
                write_json(output/'progress.json', {'run': i, 'case': name, 'evaluations': evaluations})
                key = tuple(tuple(text[f].flatten().tolist()) for f in FIELDS)
                if key not in cache:
                    rows = []
                    for identifier, meta, image in bank:
                        batch = Batch(image.to('cuda:0', adapter.dtype), text['questions'], text['targets'])
                        predicted = simulate_update(adapter, batch, obs.training)
                        rows.append({'id': identifier, **meta, 'mse': distances[identifier],
                                     'losses': scorer.score(predicted)})
                        del predicted, batch
                        evaluations += 1
                    filename = f'scores/{name}.json'
                    write_json(root/filename, {'images': rows})
                    cache[key] = filename
                mapping.append({'name': name, 'family': name, 'file': cache[key]})
                print(f'rank {args.rank} {args.phase} run {i+1}/10 case {j+1}/5 committed; {time.monotonic()-started:.1f}s', flush=True)
            write_json(root/'score_index.json', {'cases': mapping, 'hashes': {
                f: file_hash(root/f) for f in [*cache.values(), 'text_audit.json', 'replay.json', 'private/conditions.safetensors']}})
            if args.phase == 'development':
                for name, old_name in [('oracle', 'oracle'), ('baseline', 'reconstructed')]:
                    new_rows = read_json(root/next(c['file'] for c in mapping if c['name'] == name))['images']
                    old_rows = read_json(OLD/f'rank-{args.rank}/run-{i:05d}/scores/{old_name}.json')['images']
                    if [(r['id'], r['mse']) for r in new_rows] != [(r['id'], r['mse']) for r in old_rows]:
                        raise ValueError('Legacy image bank differs')
                    difference = max(abs(a['losses'][g][metric]-b['losses'][g][metric])
                                     for a, b in zip(new_rows, old_rows)
                                     for g in a['losses'] for metric in a['losses'][g])
                    if difference > 1e-6:
                        raise ValueError('Legacy gradient losses differ beyond 1e-6')
                    legacy.append({'run': i, 'condition': name, 'max_absolute_difference': difference})
            del obs, scorer, reference, baseline, prior, cases, bank, text
        write_json(output/'legacy_audit.json', {'rows': legacy})
        write_json(output/'scores_complete.json', {'rank': args.rank, 'phase': args.phase,
                   'samples': 10, 'evaluations': evaluations, 'unique_qa': len(set(qa)),
                   'qa_multiplicities': sorted(Counter(qa).values(), reverse=True),
                   'seconds': time.monotonic()-started,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
        finish_report(output)
    except Exception as error:
        write_json(output/'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rank', type=int, choices=(8, 32), required=True)
    parser.add_argument('--phase', choices=('development', 'validation'), default='development')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Expose only authorized GPU 5')
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
