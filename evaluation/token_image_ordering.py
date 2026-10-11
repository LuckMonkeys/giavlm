"""Oracle-bank diagnostic: can image ordering discriminate candidate tokens?

True images define distance-controlled image banks. True text defines candidate
labels, probe positions and explicitly oracle/partial backgrounds. This is not
an attack under image-private knowledge. All operations stay in evaluation.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from attacks.init import perturb
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json
from core.fl import load_observation, restore_model, simulate_update
from core.types import Batch
from evaluation.dager_filter_study import GPU5
from evaluation.discriminability_metrics import bootstrap


LEVELS = (.03, .1, .3, .8)
CONTEXTS = ('oracle_rest', 'half_rest', 'unknown_template', 'unknown_words')
KINDS = ('correct_slot', 'present_elsewhere', 'absent_hard', 'absent_neighbor', 'absent_random')
WORDS = 'object shape area surface detail color view space region contains appears shows visible small large left right upper lower light dark one two several'.split()


def image_bank(truth):
    bank = [('truth', truth.detach().float().cpu(), {'family': 'truth', 'level': 0.})]
    for family, seed in [('uniform_mix', 0), ('gaussian', 1)]:
        for level in LEVELS:
            bank.append((f'{family}_{level}', perturb(truth, family, level, seed),
                         {'family': family, 'level': level, 'seed': seed}))
    return bank


def ordering_metrics(distances, losses):
    """Primary metrics exclude the exact-truth anchor before calling here."""
    x, y = np.asarray(distances), np.asarray(losses)
    if len(x) < 3 or not np.isfinite(y).all():
        raise ValueError('Need at least three finite image losses')
    rho = float(spearmanr(x, y).statistic) if len(set(y)) > 1 else 0.
    differences = y[:, None] - y[None, :]
    ordered = x[:, None] < x[None, :]
    concordance = float(np.mean((differences[ordered] < 0) + .5*(differences[ordered] == 0)))
    near, far = y[x <= 1e-3], y[x >= 1e-2]
    if not len(near) or not len(far):
        raise ValueError('Image bank lacks near or far candidates')
    cross = near[:, None] - far[None, :]
    return {'rho': rho, 'concordance': concordance,
            'near_win': float(np.mean((cross < 0) + .5*(cross == 0))),
            'near_far_gap': float(far.mean()-near.mean()),
            'relative_gap': float((far.mean()-near.mean())/(abs(far.mean())+abs(near.mean())+1e-12)),
            'mean_loss': float(y.mean()), 'loss_std': float(y.std())}


def token_comparison(scores, correct, negatives, tolerance=1e-10):
    """Tie-aware ranking among one position-correct and three absent tokens."""
    values = np.array([scores[correct], *[scores[i] for i in negatives]])
    better = int((values > values[0]+tolerance).sum())
    tied = int((np.abs(values-values[0]) <= tolerance).sum())
    negative = values[1:]
    return {'rank': 1+better+(tied-1)/2,
            'top1_credit': 1/tied if better == 0 else 0.,
            'pairwise_win': float(np.mean((values[0] > negative+tolerance)
                                         + .5*(np.abs(values[0]-negative) <= tolerance))),
            'margin_over_absent_mean': float(values[0]-negative.mean())}


def background(reference, guess, lengths, role, position, context, seed):
    result = {f: reference[f].clone() if context == 'oracle_rest' else guess[f].clone()
              for f in reference}
    slots = [(f, i) for f in reference for i in range(lengths[f]) if (f, i) != (role, position)]
    if context == 'half_rest':
        order = torch.randperm(len(slots), generator=torch.Generator().manual_seed(seed))
        for j in order[:len(slots)//2].tolist():
            f, i = slots[j]
            result[f][0, i] = reference[f][0, i]
    correct_count = sum(int(result[f][0, i] == reference[f][0, i]) for f, i in slots)
    return result, correct_count / max(len(slots), 1)


class GradientScorer:
    """Global and predeclared block scores; reduce on GPU before transferring."""
    def __init__(self, target, device):
        all_names = sorted(target)
        self.names = {
            'all': all_names,
            'lora_A': [n for n in all_names if '.lora_A.' in n],
            'lora_B': [n for n in all_names if '.lora_B.' in n],
            'first_qkv_A': [n for n in all_names if '.layers.0.self_attn.' in n
                            and '.lora_A.' in n and any(f'.{p}_proj.' in n for p in 'qkv')]}
        self.targets = {group: torch.cat([target[n].detach().to(device=device, dtype=torch.float32).flatten()
                                         for n in names]) for group, names in self.names.items()}
        self.norm2 = {g: t.square().sum() for g, t in self.targets.items()}

    @torch.no_grad()
    def score(self, predicted):
        result = {}
        for group, names in self.names.items():
            candidate = torch.cat([predicted[n].detach().float().flatten() for n in names])
            target = self.targets[group]
            norm2 = candidate.square().sum()
            cosine = 1-(candidate*target).sum()/(norm2*self.norm2[group]).sqrt().clamp_min(1e-30)
            relative = (candidate-target).square().sum()/self.norm2[group].clamp_min(1e-30)
            values = torch.stack([cosine, relative]).double().cpu().tolist()
            if not all(np.isfinite(values)):
                raise ValueError('Nonfinite gradient loss')
            result[group] = dict(zip(('cosine', 'relative_l2'), values, strict=True))
        return result


def guesses(adapter, reference, lengths, seed, words=False):
    result = {f: v.clone() for f, v in reference.items()}
    rng = torch.Generator().manual_seed(seed)
    public_words = [t for word in WORDS for t in adapter.tokenizer.encode(' '+word, add_special_tokens=False)]
    phrases = {'questions': 'Describe the main object visible in this picture.',
               'targets': 'An object with a simple shape.'}
    for field in result:
        tokens = adapter.tokenizer.encode(phrases[field], add_special_tokens=False)
        if words:
            tokens = [public_words[i] for i in torch.randint(len(public_words), (lengths[field],), generator=rng).tolist()]
        else:
            tokens = (tokens*((lengths[field]+len(tokens)-1)//len(tokens)))[:lengths[field]]
        result[field][0, :lengths[field]] = torch.tensor(tokens, device=result[field].device)
    return result


def make_conditions(adapter, reference, lengths, scores, seed):
    forbidden = forbidden_token_ids(adapter)
    truth = set(reference['questions'].flatten().tolist()+reference['targets'].flatten().tolist())-forbidden
    allowed = [t for t in range(adapter.vocab_size) if t not in forbidden and not scores['ambiguous'][t]]
    absent = set(allowed)-truth
    order = sorted(absent, key=lambda t: (float(scores['scores__baseline'][t]), t))
    embedding = adapter.embedding().weight.detach().float()
    rng = torch.Generator().manual_seed(seed)
    cases = []
    for role in ('questions', 'targets'):
        eligible = [i for i in range(lengths[role])
                    if any(c.isalpha() for c in (adapter.tokenizer.convert_ids_to_tokens(int(reference[role][0, i])) or ''))]
        position = eligible[-1] if eligible else lengths[role]-1
        correct = int(reference[role][0, position])
        hard = order[0]
        neighbors = [t for t in order[:2000] if t != hard]
        similarity = torch.nn.functional.cosine_similarity(embedding[neighbors], embedding[correct][None], dim=1)
        neighbor = neighbors[int(similarity.argmax())]
        alphabetic = [t for t in sorted(absent-{hard, neighbor})
                      if (adapter.tokenizer.convert_ids_to_tokens(t) or '').lstrip('▁').isalpha()]
        random_id = alphabetic[int(torch.randint(len(alphabetic), (), generator=rng))]
        present_elsewhere = min(truth-{correct})
        candidates = dict(zip(KINDS, [correct, present_elsewhere, hard, neighbor, random_id], strict=True))
        for context in CONTEXTS:
            guessed = guesses(adapter, reference, lengths, seed+37, words=context == 'unknown_words')
            text, actual_fraction = background(reference, guessed, lengths, role, position, context, seed+71)
            for kind, token in candidates.items():
                modified = {f: x.clone() for f, x in text.items()}
                modified[role][0, position] = token
                cases.append({'id': f'case-{len(cases):03d}', 'role': role, 'position': position,
                              'context': context, 'kind': kind, 'token_id': token,
                              'actual_other_token_correct_fraction': actual_fraction,
                              'text': modified})
    return cases


def evaluate(output, count):
    if not (output / 'scores_complete.json').exists():
        raise ValueError('Commit every score before evaluating token labels')
    rows = []
    for index in range(count):
        root = output / f'run-{index:05d}'
        index_data = read_json(root / 'score_index.json')
        if any(file_hash(root / filename) != checksum for filename, checksum in index_data['hashes'].items()):
            raise ValueError('Committed image-score integrity failure')
        definitions = read_json(root / 'private/conditions.json')
        definitions = {c['id']: c for c in definitions}
        grouped = {}
        for case_id, filename in index_data['cases'].items():
            saved = read_json(root / filename)
            case = definitions[case_id]
            perturbations = [r for r in saved['images'] if r['id'] != 'truth']
            for group in ('all', 'lora_A', 'lora_B', 'first_qkv_A'):
                for loss in ('cosine', 'relative_l2'):
                    stats = ordering_metrics([r['mse'] for r in perturbations],
                                             [r['losses'][group][loss] for r in perturbations])
                    record = {'run': index, 'role': case['role'], 'context': case['context'],
                              'kind': case['kind'], 'group': group, 'loss': loss,
                              'other_token_correct_fraction': case['actual_other_token_correct_fraction'],
                              **stats}
                    rows.append(record)
                    grouped.setdefault((case['role'], case['context'], group, loss), {})[case['kind']] = record
        comparisons = []
        for (role, context, group, loss), by_kind in grouped.items():
            for metric in ('rho', 'concordance', 'near_win', 'near_far_gap', 'relative_gap'):
                values = {k: r[metric] for k, r in by_kind.items()}
                comparison = token_comparison(values, 'correct_slot', [k for k in KINDS if k.startswith('absent_')])
                comparisons.append({'run': index, 'role': role, 'context': context, 'group': group,
                                     'loss': loss, 'metric': metric, **comparison,
                                     'correct_value': values['correct_slot'],
                                     'present_elsewhere_value': values['present_elsewhere'],
                                     'absent_mean': float(np.mean([values[k] for k in KINDS if k.startswith('absent_')])),
                                     'delta_from_random_baseline': {k: v-values['absent_random'] for k, v in values.items()}})
        write_json(root / 'evaluated.json', {'comparisons': comparisons})
    comparisons = [r for i in range(count) for r in read_json(output / f'run-{i:05d}/evaluated.json')['comparisons']]
    aggregate = []
    dimensions = ('role', 'context', 'group', 'loss', 'metric')
    for key in sorted({tuple(r[k] for k in dimensions) for r in comparisons}):
        selected = [r for r in comparisons if tuple(r[k] for k in dimensions) == key]
        aggregate.append({**dict(zip(dimensions, key, strict=True)), 'images': len(selected),
                          **{name: bootstrap([r[name] for r in selected], seed=7301)
                             for name in ('top1_credit', 'pairwise_win', 'margin_over_absent_mean',
                                          'correct_value', 'present_elsewhere_value', 'absent_mean')}})
    report = {'uses_private_reference': True, 'sample_count': count,
              'primary': 'all/cosine/rho, exact-truth image excluded; correct-slot vs three absent tokens',
              'chance': {'top1_credit': .25, 'pairwise_win': .5},
              'aggregate': aggregate, 'comparisons': comparisons, 'rows': rows,
              'plan_sha256': file_hash(output / 'plan.json')}
    write_json(output / 'report.json', report)
    write_json(output / 'complete.json', {'report_sha256': file_hash(output / 'report.json')})


def run(args):
    output, study = Path(args.output), Path(args.study)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic output')
    selected = torch.randperm(10, generator=torch.Generator().manual_seed(20261006))[:args.samples].tolist()
    write_json(output / 'plan.json', {
        'schema_version': 1, 'uses_private_reference': True, 'not_an_attack_benchmark': True,
        'source_sha256': source_fingerprint(), 'source_runs': selected, 'sample_count': args.samples,
        'model_sha256': file_hash(Path(args.model) / 'model.safetensors'),
        'contexts': CONTEXTS, 'candidate_kinds': KINDS, 'image_levels': LEVELS,
        'image_families': {'uniform_mix': 0, 'gaussian': 1}, 'image_count': 9,
        'probe_policy': 'Last alphabetic question token and last alphabetic answer token; positions use private text.',
        'negative_policy': 'Absent anywhere in truth: best baseline token, nearest embedding in top2000, uniform alphabetic token.',
        'primary': 'All uploaded gradients, cosine loss, image-order Spearman excluding exact truth image.',
        'controls': ['Truth-image exact replay', 'Correct token at known slot', 'Present token at wrong slot',
                     'Oracle/half/unknown rest', 'Two unknown text backgrounds', 'Relative L2 and gradient blocks'],
        'scope': 'Ground-truth images, text and positions construct diagnostic inputs. No recovery claim.',
        'seed': 20261006})
    started = time.monotonic()
    try:
        adapter = restore_model(args.model, 'cuda:0')
        model_fingerprint = adapter.fingerprint()
        private_sequences = []
        for index, source_run in enumerate(selected):
            root = output / f'run-{index:05d}'
            source = study / f'run-{source_run:05d}'
            obs = load_observation(source / 'captures/private/public', 'cuda:0')
            if obs.model_fingerprint != model_fingerprint:
                raise ValueError('Observation and restored model differ')
            adapter.set_training_spec(obs.training)
            if obs.training.algorithm != 'fedsgd' or obs.sample_count != 1:
                raise ValueError('Only captured single-sample FedSGD is supported')
            known = load_observation(source / 'captures/known/public', 'cuda:0')
            truth = known.public_images.detach().float().cpu()
            references = read_tensors(source / 'captures/known/private/text_tokens.safetensors', 'cuda:0')
            lengths = {'questions': obs.public_question_lengths[0], 'targets': obs.public_target_lengths[0]}
            private_sequences.append(tuple(tuple(references[f][0, :lengths[f]].tolist()) for f in ('questions', 'targets')))
            del known
            baseline = read_tensors(Path(args.filter_scores) / f'run-{source_run:05d}/random0/scores.safetensors')
            cases = make_conditions(adapter, references, lengths, baseline, seed=7310+index)
            write_json(root / 'private/conditions.json', [
                {**{k: v for k, v in c.items() if k != 'text'},
                 'text': {f: x.cpu().tolist() for f, x in c['text'].items()}} for c in cases])
            bank = image_bank(truth)
            distances = {name: float((image-truth).square().mean()) for name, image, _ in bank}
            scorer = GradientScorer(obs.tensors, 'cuda:0')
            true_batch = Batch(truth.to('cuda:0', adapter.dtype), references['questions'], references['targets'])
            replay = simulate_update(adapter, true_batch, obs.training)
            replay_scores = scorer.score(replay)
            if replay_scores['all']['relative_l2'] > 1e-12:
                raise ValueError('Victim gradient replay differs from captured update')
            write_json(root / 'replay.json', {'relative_l2': replay_scores['all']['relative_l2']})
            del replay, true_batch
            cache, mapping = {}, {}
            for j, case in enumerate(cases):
                write_json(output / 'progress.json', {'run': index, 'case': j, 'cases': len(cases)})
                key = tuple(tuple(case['text'][f].flatten().tolist()) for f in ('questions', 'targets'))
                if key in cache:
                    mapping[case['id']] = cache[key]
                    continue
                records = []
                for name, image, metadata in bank:
                    candidate = Batch(image.to('cuda:0', adapter.dtype), case['text']['questions'], case['text']['targets'])
                    predicted = simulate_update(adapter, candidate, obs.training)
                    losses = scorer.score(predicted)
                    records.append({'id': name, **metadata, 'mse': distances[name], 'losses': losses})
                    del predicted, candidate
                filename = f'scores/{case["id"]}.json'
                write_json(root / filename, {'images': records})
                cache[key] = filename
                mapping[case['id']] = filename
                print(f'run {index+1}/{args.samples} case {j+1}/{len(cases)} saved; elapsed {time.monotonic()-started:.1f}s', flush=True)
            write_json(root / 'score_index.json', {'cases': mapping, 'unique_texts': len(cache),
                                                  'hashes': {p: file_hash(root/p) for p in cache.values()}})
            del obs, references, scorer, cases, bank
        write_json(output / 'scores_complete.json', {'sample_count': len(selected),
                    'unique_qa_sequences': len(set(private_sequences)),
                    'qa_multiplicities': sorted(Counter(private_sequences).values(), reverse=True),
                    'seconds': time.monotonic()-started})
        evaluate(output, len(selected))
        print('Token/image ordering diagnostic complete', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--study', default='outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
    parser.add_argument('--filter-scores', default='outputs/diagnostics/dager_filter_gpu5_20261005_v2')
    parser.add_argument('--model', default='outputs/federation/slake_llava_f_l_fedavg/round-0010')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Expose only authorized GPU 5')
    if not 1 <= args.samples <= 10:
        raise ValueError('Sample count must be between 1 and 10')
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
