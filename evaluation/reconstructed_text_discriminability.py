"""Private-reference image ordering with committed Residual DAGER text.

No attack is run here. Oracle text permutations and oracle image banks are
diagnostic controls; actual reconstructed tokens are consumed without alteration.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
import torch

from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.fl import load_observation, restore_model, simulate_update
from core.types import Batch
from evaluation.dager_rank_study import BASELINE, GPU5
from evaluation.discriminability_metrics import bootstrap
from evaluation.text_unknown_discriminability import candidate_bank, load_spec
from evaluation.token_image_ordering import GradientScorer, guesses, ordering_metrics


RANK_STUDY = Path('outputs/diagnostics/dager_rank_gpu5_20261010_v2')
FIELDS = ('questions', 'targets')
SEEDS = (20261011, 20261012)


def shuffled(tokens, length, seed):
    """Permute content token IDs only, keeping EOS/padding and multiplicities.

    If a random permutation is identical, rotate once to ensure a changed
    sequence whenever the content contains at least two distinct token IDs.
    """
    result = tokens.clone()
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(length, generator=rng).to(tokens.device)
    values = tokens[0, :length]
    permuted = values[order]
    if torch.equal(values, permuted) and len(set(values.tolist())) > 1:
        permuted = values.roll(1)
    result[0, :length] = permuted
    return result


def text_audit(reference, text, lengths):
    result = {}
    for field in FIELDS:
        n = lengths[field]
        a, b = reference[field][0, :n].tolist(), text[field][0, :n].tolist()
        overlap = sum((Counter(a) & Counter(b)).values())
        result[field] = {'length': n, 'bag_recall': overlap/max(n, 1),
                         'position_accuracy': sum(x == y for x, y in zip(a, b))/max(n, 1),
                         'exact': a == b, 'same_bag': Counter(a) == Counter(b),
                         'pure_order_error': a != b and Counter(a) == Counter(b)}
    return result


def conditions(adapter, reference, reconstructed, lengths, index):
    cases = [('oracle', 'oracle', reference),
             ('reconstructed', 'reconstructed', reconstructed),
             ('reconstructed_question', 'reconstructed_question',
              {'questions': reconstructed['questions'], 'targets': reference['targets']}),
             ('reconstructed_answer', 'reconstructed_answer',
              {'questions': reference['questions'], 'targets': reconstructed['targets']})]
    for seed in SEEDS:
        permuted = {field: shuffled(reference[field], lengths[field], seed+100*index+j)
                    for j, field in enumerate(FIELDS)}
        for role, fields in [('question', ('questions',)), ('answer', ('targets',)),
                             ('both', FIELDS)]:
            text = {f: permuted[f] if f in fields else reference[f] for f in FIELDS}
            cases.append((f'shuffle_{role}_{seed}', f'shuffle_{role}', text))
    for words in (False, True):
        name = 'unknown_words' if words else 'unknown_template'
        cases.append((name, name, guesses(adapter, reference, lengths, 7301+index, words)))
    return cases


def bank_statistics(images, group, loss):
    values = np.array([r['losses'][group][loss] for r in images])
    ranks = (rankdata(values, method='average')-1)/(len(values)-1)
    keep = [i for i, r in enumerate(images) if r['id'] != 'truth']
    stats = ordering_metrics([images[i]['mse'] for i in keep], values[keep])
    stats['truth_percentile'] = float(ranks[[r['id'] for r in images].index('truth')])
    return stats, ranks


def evaluate(output):
    complete = read_json(output/'scores_complete.json')
    rows, audit = [], []
    for index in range(complete['samples']):
        root = output/f'run-{index:05d}'
        saved = read_json(root/'score_index.json')
        for filename, checksum in saved['hashes'].items():
            if file_hash(root/filename) != checksum:
                raise ValueError('Committed scores changed')
        audit.append(read_json(root/'text_audit.json'))
        for family in sorted({c['family'] for c in saved['cases']}):
            selected = [c for c in saved['cases'] if c['family'] == family]
            for group in ('all', 'lora_A', 'lora_B', 'first_qkv_A'):
                for loss in ('cosine', 'relative_l2'):
                    stats, ranks = [], []
                    for case in selected:
                        images = read_json(root/case['file'])['images']
                        s, r = bank_statistics(images, group, loss)
                        stats.append(s)
                        ranks.append(r)
                    mean_ranks = np.mean(ranks, axis=0)
                    keep = [i for i, r in enumerate(images) if r['id'] != 'truth']
                    aggregate = ordering_metrics([images[i]['mse'] for i in keep], mean_ranks[keep])
                    rows.append({'run': index, 'family': family, 'group': group, 'loss': loss,
                                 'rho': aggregate['rho'], 'near_win': aggregate['near_win'],
                                 'conditional_rho': float(np.mean([s['rho'] for s in stats])),
                                 'truth_percentile': float(np.mean([s['truth_percentile'] for s in stats]))})
    aggregate, paired = [], []
    dimensions = ('family', 'group', 'loss')
    for key in sorted({tuple(r[k] for k in dimensions) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in dimensions) == key]
        aggregate.append({**dict(zip(dimensions, key)), **{
            metric: bootstrap([r[metric] for r in selected], seed=20261011)
            for metric in ('rho', 'near_win', 'conditional_rho', 'truth_percentile')}})
        oracle = [next(r for r in rows if r['run'] == s['run'] and r['family'] == 'oracle'
                       and r['group'] == key[1] and r['loss'] == key[2]) for s in selected]
        paired.append({**dict(zip(dimensions, key)), 'rho_delta_from_oracle': bootstrap(
            [s['rho']-o['rho'] for s, o in zip(selected, oracle)], seed=20261011)})
    report = {'uses_private_reference': True, 'plan_sha256': file_hash(output/'plan.json'),
              'aggregate': aggregate, 'paired': paired, 'rows': rows, 'text_audits': audit,
              'completion': complete}
    write_json(output/'report.json', report)
    write_json(output/'complete.json', {'report_sha256': file_hash(output/'report.json')})


def run(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh output; no automatic retry')
    started = time.monotonic()
    try:
        source = BASELINE if args.rank == 8 else RANK_STUDY/'rank-32'
        model = (Path('outputs/federation/slake_llava_f_l_fedavg/round-0010') if args.rank == 8
                 else RANK_STUDY/'models/rank-32/round-0010')
        spec = load_spec('configs/diagnostics/slake_llava_text_unknown_discriminability.yaml')
        inputs = {}
        for index in range(args.samples):
            src = source/f'run-{index:05d}'
            attack = src/'attacks/random_residual'
            if read_json(attack/'result.json')['status'] != 'completed':
                raise ValueError('Source reconstruction is not completed')
            paths = {'reconstructed_tokens': attack/'text_tokens.safetensors',
                     'result': attack/'result.json',
                     'observation': src/'captures/private/public/observation.json',
                     'update': src/'captures/private/public/update.safetensors',
                     'reference_tokens': src/'captures/known/private/text_tokens.safetensors',
                     'reference_image': src/'captures/known/public/public_images.safetensors'}
            inputs[str(index)] = {label: file_hash(path) for label, path in paths.items()}
        write_json(output/'plan.json', {
            'schema_version': 1, 'uses_private_reference': True, 'not_an_attack_benchmark': True,
            'source_sha256': source_fingerprint(), 'rank': args.rank, 'samples': args.samples,
            'source_study': str(source), 'method': 'dager_adapted', 'filter': 'residual',
            'source_image_context': 'random surrogate, private image, seed 0',
            'model_sha256': {f: file_hash(model/f) for f in ('model.json', 'model.safetensors')},
            'input_sha256': inputs, 'candidate_ids': spec['candidate_ids'], 'shuffle_seeds': SEEDS,
            'primary': 'all cosine, Spearman and near-win over 24 nontruth images; mean ranks across permutations',
            'controls': ['oracle', 'question-only reconstruction', 'answer-only reconstruction',
                         'question/answer/both token permutations', 'length-matched template and words'],
            'shuffle_policy': 'Within each question/answer content segment; preserve token multiset, EOS, padding; force change if possible',
            'selection': 'First ten existing paired DAGER samples, no filtering by text quality',
            'secondary': 'LoRA-A, LoRA-B, first QKV-A and relative L2; individual permutation rho',
            'privacy': 'Private references stay inside evaluation; no decoded text in reports',
            'physical_gpu': 5})
        adapter = restore_model(model, 'cuda:0')
        fingerprint = adapter.fingerprint()
        torch.cuda.reset_peak_memory_stats()
        qa = []
        evaluations = 0
        for index in range(args.samples):
            root, src = output/f'run-{index:05d}', source/f'run-{index:05d}'
            obs = load_observation(src/'captures/private/public', 'cuda:0')
            if obs.model_fingerprint != fingerprint or obs.sample_count != 1 or obs.training.algorithm != 'fedsgd':
                raise ValueError('Observation incompatible with model or protocol')
            adapter.set_training_spec(obs.training)
            truth = read_tensors(src/'captures/known/public/public_images.safetensors')['images'].float()
            reference = read_tensors(src/'captures/known/private/text_tokens.safetensors', 'cuda:0')
            reconstructed = read_tensors(src/'attacks/random_residual/text_tokens.safetensors', 'cuda:0')
            lengths = dict(zip(FIELDS, (obs.public_question_lengths[0], obs.public_target_lengths[0])))
            qa.append(tuple(tuple(reference[f][0, :lengths[f]].tolist()) for f in FIELDS))
            cases = conditions(adapter, reference, reconstructed, lengths, index)
            write_json(root/'text_audit.json', {name: text_audit(reference, text, lengths)
                                               for name, _, text in cases})
            write_tensors(root/'private/conditions.safetensors', {
                f'{name}.{field}': text[field] for name, _, text in cases for field in FIELDS})
            bank = candidate_bank(spec, truth)
            distances = {name: float((image-truth).square().mean()) for name, _, image in bank}
            scorer = GradientScorer(obs.tensors, 'cuda:0')
            replay = simulate_update(adapter, Batch(truth.to('cuda:0', adapter.dtype),
                                     reference['questions'], reference['targets']), obs.training)
            replay_loss = scorer.score(replay)['all']['relative_l2']
            del replay
            if replay_loss > 1e-12:
                raise ValueError('Captured gradient replay differs')
            write_json(root/'replay.json', {'relative_l2': replay_loss})
            cache, mapping = {}, []
            for case_index, (name, family, text) in enumerate(cases):
                write_json(output/'progress.json', {'run': index, 'case': case_index, 'cases': len(cases),
                                                   'evaluations': evaluations, 'seconds': time.monotonic()-started})
                key = tuple(tuple(text[f].flatten().tolist()) for f in FIELDS)
                if key not in cache:
                    records = []
                    for identifier, metadata, image in bank:
                        candidate = Batch(image.to('cuda:0', adapter.dtype), text['questions'], text['targets'])
                        predicted = simulate_update(adapter, candidate, obs.training)
                        losses = scorer.score(predicted)
                        records.append({'id': identifier, **metadata, 'mse': distances[identifier], 'losses': losses})
                        del predicted, candidate
                        evaluations += 1
                    filename = f'scores/{name}.json'
                    write_json(root/filename, {'images': records})
                    cache[key] = filename
                mapping.append({'name': name, 'family': family, 'file': cache[key]})
                print(f'rank {args.rank} run {index+1}/{args.samples} case {case_index+1}/{len(cases)} committed; {time.monotonic()-started:.1f}s', flush=True)
            write_json(root/'score_index.json', {'cases': mapping, 'hashes': {
                f: file_hash(root/f) for f in [*cache.values(), 'text_audit.json', 'replay.json']}})
            del obs, scorer, reference, reconstructed, cases, bank, text
        write_json(output/'scores_complete.json', {'samples': args.samples, 'rank': args.rank,
                   'evaluations': evaluations, 'unique_qa': len(set(qa)),
                   'qa_multiplicities': sorted(Counter(qa).values(), reverse=True),
                   'seconds': time.monotonic()-started,
                   'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
        evaluate(output)
    except Exception as error:
        write_json(output/'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rank', type=int, choices=(8, 32), required=True)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Expose only authorized GPU 5')
    if not 1 <= args.samples <= 10:
        raise ValueError('Require 1 to 10 paired samples')
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
