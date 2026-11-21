"""Public-observation pilot of LM candidate supplementation under Top-k uploads.

Experimental search only; production attack defaults and registry are unchanged.
All reconstructions commit before references are opened by the evaluation phase.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import statistics
import time
import traceback

import torch

from attacks.analytic.dager_adapted import DAGERSearch, SearchBudgetExhausted, forbidden_token_ids
from attacks.analytic.dager_search import retain_best
from attacks.analytic.dager_subspace import select_tokens
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import AttackSpec, DAGEROptions, EvalSpec
from core.fl import load_observation, restore_model
from core.types import Reconstruction
from evaluation.reconstruction import evaluate
from metrics.text import text_metrics


VARIANTS = ("gradient100", "gradient116", "dynamic16", "lm_only")


class TextOnlyView(LlavaTextView):
    """Only public length scalars and the model enter the LM-only control."""

    def __init__(self, adapter, question_length, target_length):
        self.adapter = adapter
        self.device = adapter.embedding().weight.device
        self.question_content_length = question_length
        self.question_is_public = False
        self.question_ids = self._slots(adapter.spec.question_length, adapter.eos, adapter.pad, question_length)
        self.target_ids = self._slots(adapter.spec.target_length, adapter.eos, adapter.pad, target_length)
        self.slots = ([('questions', i, i) for i in range(question_length)] +
                      [('targets', i, question_length + i) for i in range(target_length)])


def allowed_lm_top(nll, forbidden, count):
    order = torch.argsort(nll, stable=True).cpu().tolist()
    return [i for i in order if i not in forbidden][:count]


def supplement(base, expanded, nll, forbidden, count=16):
    """Union with LM Top-M, then gradient-fill to the expanded control's width.

    All lists are public-score selected. Shared ambiguous extras are already in
    base. Filling overlaps preserves the exact same branch count as Top-(K+M).
    """
    result = set(base) | set(allowed_lm_top(nll, forbidden, count))
    if len(result) > len(expanded):
        raise ValueError("LM supplementation exceeded the prespecified branch cap")
    for token in expanded:
        if len(result) == len(expanded):
            break
        result.add(token)
    if len(result) != len(expanded):
        raise ValueError("Cannot match expanded control's branch count")
    return sorted(result)


@torch.no_grad()
def search(view, span, base, expanded, forbidden, spec, variant, directory, progress=None):
    if variant not in VARIANTS or (variant == "lm_only") != (span is None):
        raise ValueError("LM-only must not receive a gradient span")
    started = time.monotonic()
    costs = dict(prefix_candidates_evaluated=0, prefix_forward_batches=0,
                 lm_prior_prefixes=0, lm_prior_forward_batches=0, lm_prior_input_tokens=0,
                 full_gradient_replays=0, ambiguous_prefix_checks=0)
    beam, history = [{"tokens": [], "score": 0., "span_score": 0., "lm_nll": 0.}], []

    def budget(requested):
        used = costs['prefix_candidates_evaluated'] + costs['lm_prior_prefixes']
        if time.monotonic() - started >= spec.seconds or (
                spec.max_evaluations is not None and used + requested > spec.max_evaluations):
            raise SearchBudgetExhausted()

    try:
        for depth in range(len(view.slots)):
            budget(len(beam))
            nll, input_tokens = view.text_prior_nll([row['tokens'] for row in beam],
                                                   list(range(view.adapter.vocab_size)))
            nll = nll.cpu()
            costs['lm_prior_prefixes'] += len(beam)
            costs['lm_prior_forward_batches'] += 1
            costs['lm_prior_input_tokens'] += input_tokens
            if variant == 'dynamic16':
                pools = [supplement(base, expanded, values, forbidden) for values in nll]
            elif variant == 'lm_only':
                pools = [sorted(allowed_lm_top(values, forbidden, 16)) for values in nll]
            else:
                pools = [base if variant == 'gradient100' else expanded for _ in beam]
            children = [(parent, token) for parent, pool in enumerate(pools) for token in pool]
            next_beam = []
            for start in range(0, len(children), spec.dager.prefix_batch_size):
                part = children[start:start + spec.dager.prefix_batch_size]
                budget(len(part))
                additions = [{'tokens': beam[parent]['tokens'] + [token]} for parent, token in part]
                if span is None:
                    distances = [0.] * len(part)
                else:
                    features = view.prefix_features([row['tokens'] for row in additions])
                    distances, ambiguous = span.score(features)
                    distances = distances.cpu().tolist()
                    costs['prefix_forward_batches'] += 1
                    costs['ambiguous_prefix_checks'] += int(ambiguous.sum())
                costs['prefix_candidates_evaluated'] += len(part)
                for row, (parent, token), distance in zip(additions, part, distances, strict=True):
                    row['span_score'] = beam[parent]['span_score'] + distance
                    row['lm_nll'] = beam[parent]['lm_nll'] + float(nll[parent, token])
                    row['score'] = (row['lm_nll'] if span is None else
                                    row['span_score'] + spec.dager.lm_prior_weight * row['lm_nll'])
                next_beam = retain_best(next_beam, additions, spec.dager.beam_width)
            # Complete-depth checkpoints also retain proposed edges for a
            # strictly post-commit pruning audit. No private prefixes enter here.
            write_json(directory / 'checkpoints' / f'depth-{depth + 1:03d}.json', {
                'schema': 1, 'depth': depth + 1, 'parents': beam, 'pools': pools,
                'beam': next_beam, 'costs': costs, 'variant': variant})
            beam = next_beam
            history.append({'private_tokens_filled': depth + 1, 'beam_size': len(beam),
                            'best_span_score': beam[0]['span_score'], 'best_search_score': beam[0]['score'],
                            'min_branch_count': min(map(len, pools)), 'max_branch_count': max(map(len, pools))})
            if progress:
                progress(depth + 1, len(view.slots), costs)
        winner = beam[0]
        ids = view.token_ids(winner['tokens'])
        write_tensors(directory / 'text_tokens.safetensors', ids)
        costs['seconds'] = time.monotonic() - started
        return Reconstruction('completed', questions=view.adapter.decode(ids['questions']),
                              targets=view.adapter.decode(ids['targets']), costs=costs, history=history,
                              provenance={'reconstructed_fields': ['question', 'target'],
                                          'variant': variant, 'final_score': winner['score'],
                                          'final_score_type': 'text_only_nll' if span is None else
                                                              'span_distance_with_text_prior',
                                          'final_beam': beam, 'uses_private_reference': False})
    except BaseException:
        write_json(directory / 'failed.json', {'status': 'failed', 'variant': variant,
                   'depth': len(history), 'costs': costs, 'traceback': traceback.format_exc()})
        raise


def prepare(output, source):
    if (output / 'plan.json').exists():
        raise FileExistsError('Use a new experiment directory')
    parent = read_json(source / 'plan.json')
    jobs = []
    for length in ('1-16', '17-32'):
        selected = [j for j in parent['jobs'] if j['condition'] == 'sparsify' and
                    j['length_bins'] == [length]][:2]
        if len(selected) != 2:
            raise ValueError('Expected the original two short and two medium uploads')
        jobs.extend(selected)
    sources = {}
    for job in jobs:
        for image in ('gt', 'random'):
            public = source / 'runs' / job['id'] / 'captures' / image / 'public'
            meta = read_json(public / 'observation.json')
            sources[f"{job['id']}:{image}"] = {
                'metadata_sha256': file_hash(public / 'observation.json'),
                'observation_id': meta['observation_id'], 'update_sha256': meta['update_sha256'],
                'filter_sha256': file_hash(source / 'runs' / job['id'] / 'attacks' / image /
                                           'token_candidates.safetensors'),
                'baseline_result_sha256': file_hash(source / 'runs' / job['id'] / 'attacks' / image / 'result.json')}
    write_json(output / 'plan.json', {
        'schema_version': 1, 'source_sha256': source_fingerprint(), 'source': str(source.resolve()),
        'source_plan_sha256': file_hash(source / 'plan.json'), 'jobs': jobs, 'sources': sources,
        'model': parent['model'], 'model_fingerprint': parent['model_fingerprint'],
        'model_metadata_sha256': parent['model_metadata_sha256'], 'cohort_n': 4,
        'variants': list(VARIANTS), 'gradient_attacks': 24, 'lm_only_attacks': 4, 'total_attacks': 28,
        'attack_specs': parent['attack_specs']['1'], 'supplement_size': 16,
        'candidate_policy': 'gradient100; gradient116; gradient100 union LM Top16, gradient-fill to same width',
        'selection': 'first two short and first two medium uploads in original frozen order; no outcomes',
        'score_before_truth': True, 'no_new_capture_or_noise': True,
        'lm_only_inputs': 'public question/answer content lengths and public model only; no image or gradient'})


def verify(output):
    plan = read_json(output / 'plan.json')
    if source_fingerprint() != plan['source_sha256']:
        raise ValueError('Frozen source changed')
    if file_hash(Path(plan['model']) / 'model.json') != plan['model_metadata_sha256']:
        raise ValueError('Model metadata changed')
    return plan


def commit_result(directory, result, observation_id, condition):
    write_json(directory / 'result.json', {
        **{k: v for k, v in asdict(result).items() if k != 'images'},
        'observation_id': observation_id, 'condition': condition})
    write_json(directory / 'complete.json', {
        'status': result.status, 'result_sha256': file_hash(directory / 'result.json'),
        'tokens_sha256': file_hash(directory / 'text_tokens.safetensors')})


@torch.no_grad()
def attack_all(output):
    plan = verify(output)
    source = Path(plan['source'])
    adapter = restore_model(Path(plan['model']), 'cuda:0')
    if adapter.fingerprint() != plan['model_fingerprint']:
        raise ValueError('Model differs from original uploads')
    versions = {n: p._version for n, p in adapter.named_parameters()}
    forbidden, completed = forbidden_token_ids(adapter), 0
    for job in plan['jobs']:
        for image in ('gt', 'random'):
            public = source / 'runs' / job['id'] / 'captures' / image / 'public'
            expected = plan['sources'][f"{job['id']}:{image}"]
            if file_hash(public / 'observation.json') != expected['metadata_sha256']:
                raise ValueError('Observation metadata changed')
            obs = load_observation(public, 'cuda:0')
            if obs.model_fingerprint != plan['model_fingerprint'] or (
                    image == 'random' and obs.public_images is not None):
                raise ValueError('Invalid public observation')
            adapter.set_training_spec(obs.training)
            raw = plan['attack_specs'][image]
            spec = AttackSpec(**{**raw, 'dager': DAGEROptions(**raw['dager'])})
            helper = DAGERSearch(adapter, obs, spec, None)
            helper.prepare()
            helper.filter_tokens()
            original_dir = source / 'runs' / job['id'] / 'attacks' / image
            if file_hash(original_dir / 'token_candidates.safetensors') != expected['filter_sha256']:
                raise ValueError('Original filter changed')
            original = read_tensors(original_dir / 'token_candidates.safetensors')
            delta = float((helper.scores - original['scores']).abs().max())
            base = helper.state['candidates']
            if set(base) != set(original['selected_ids'].tolist()) or delta > 1e-6:
                raise ValueError('Original token filter not reproduced')
            expanded, _ = select_tokens(helper.scores, helper.ambiguous, forbidden,
                                         replace(spec.dager, max_candidates=116))
            # Fill LM overlaps by the original gradient order, not token ID.
            expanded_ranked = sorted(expanded, key=lambda i, scores=helper.scores: (float(scores[i]), i))
            for variant in VARIANTS[:3]:
                directory = output / 'runs' / job['id'] / image / variant
                if directory.exists():
                    raise FileExistsError('No implicit retry or resume')
                write_tensors(directory / 'token_candidates.safetensors', {
                    'scores': helper.scores, 'ambiguous': helper.ambiguous,
                    'selected_ids': torch.tensor(base if variant != 'gradient116' else expanded),
                    'gradient100_ids': torch.tensor(base), 'gradient116_ids': torch.tensor(expanded)})
                write_json(directory / 'token_candidates.json', {
                    'forbidden_ids': sorted(forbidden), 'baseline_candidates_exact': True,
                    'baseline_max_score_delta': delta, 'variant': variant,
                    'static_selected_ids_are_base_only': variant == 'dynamic16'})
                torch.cuda.reset_peak_memory_stats()
                def progress(depth, total, costs):
                    write_json(output / 'progress-attack.json', {
                        'status': 'running', 'completed': completed, 'total_attacks': 28,
                        'job': job['id'], 'image': image, 'variant': variant,
                        'depth': depth, 'total_depth': total, 'costs': costs})
                result = search(helper.view, helper.spans[1], base, expanded_ranked, forbidden,
                                spec, variant, directory, progress)
                if variant == 'gradient100':
                    if file_hash(original_dir / 'result.json') != expected['baseline_result_sha256']:
                        raise ValueError('Original baseline result changed')
                    tokens = read_tensors(original_dir / 'text_tokens.safetensors')
                    current = read_tensors(directory / 'text_tokens.safetensors')
                    if any(not torch.equal(tokens[k], current[k]) for k in tokens):
                        raise ValueError('Original static Top100 winner not reproduced')
                    result.provenance['original_winner_exact'] = True
                result.provenance.update(baseline_candidates_exact=True, baseline_max_score_delta=delta,
                                         peak_allocated_bytes=torch.cuda.max_memory_allocated())
                condition = {**asdict(obs.training), 'method': 'dynamic_candidate_diagnostic',
                             'variant': variant, 'defense': job['defense'], 'source_sha256': plan['source_sha256']}
                commit_result(directory, result, expected['observation_id'], condition)
                completed += 1
                print('committed', job['id'], image, variant, result.status, flush=True)
            del helper, obs
            torch.cuda.empty_cache()
        # LM-only never receives an Observation, upload tensors, filter, image,
        # original results or sample contents; just declared length scalars.
        meta = read_json(source / 'runs' / job['id'] / 'captures/random/public/observation.json')
        lm_view = TextOnlyView(adapter, meta['public_question_lengths'][0], meta['public_target_lengths'][0])
        directory = output / 'runs' / job['id'] / 'none/lm_only'
        if directory.exists():
            raise FileExistsError('No implicit LM-only retry')
        result = search(lm_view, None, [], [], forbidden, spec, 'lm_only', directory)
        result.provenance['no_image_or_gradient_inputs'] = True
        commit_result(directory, result, plan['sources'][f"{job['id']}:random"]['observation_id'],
                      {**meta['training'], 'method': 'lm_only_diagnostic', 'variant': 'lm_only',
                       'source_sha256': plan['source_sha256']})
        completed += 1
        print('committed', job['id'], 'none', 'lm_only', result.status, flush=True)
    if any(p._version != versions[n] for n, p in adapter.named_parameters()):
        raise ValueError('Read-only search mutated the model')
    write_json(output / 'attacks-complete.json', {
        'status': 'completed', 'attacks': completed, 'time': time.time(),
        'source_sha256': plan['source_sha256'], 'initial_model_immutable': True,
        'private_reference_read': False})
    write_json(output / 'progress-attack.json', {'status': 'completed', 'completed': completed, 'total_attacks': 28})


def reconstruction_dirs(output, plan):
    return [(job, image, variant, output / 'runs' / job['id'] / image / variant)
            for job in plan['jobs'] for image, variants in
            [('gt', VARIANTS[:3]), ('random', VARIANTS[:3]), ('none', ('lm_only',))]
            for variant in variants]


def evaluate_all(output):
    plan = verify(output)
    completion = read_json(output / 'attacks-complete.json')
    entries = reconstruction_dirs(output, plan)
    if completion['attacks'] != len(entries) or len(entries) != 28:
        raise ValueError('Incomplete experiment')
    # Preflight every commit before any reference is opened.
    for _, _, _, directory in entries:
        meta = read_json(directory / 'complete.json')
        if meta['status'] != 'completed' or any(file_hash(directory / name) != meta[key] for name, key in
                [('result.json', 'result_sha256'), ('text_tokens.safetensors', 'tokens_sha256')]):
            raise ValueError('Incomplete or changed reconstruction')
    write_json(output / 'evaluation-barrier.json', {'attacks_verified': 28, 'time': time.time(),
                                                  'private_reference_read': False})
    rows = []
    for job, image, variant, directory in entries:
        truth_image = 'random' if image == 'none' else image
        truth = Path(plan['source']) / 'runs' / job['id'] / 'captures' / truth_image / 'private'
        report = evaluate(directory, truth, EvalSpec())
        write_json(directory / 'evaluation.json', report)
        result = read_json(directory / 'result.json')
        ref = read_json(truth / 'truth.json')['samples'][0]
        m = report['samples'][0]['metrics']
        full = text_metrics(ref['model_question'] + ' ' + ref['model_target'],
                            result['questions'][0] + ' ' + result['targets'][0])
        refs = read_tensors(truth / 'text_tokens.safetensors')
        public = read_json(Path(plan['source']) / 'runs' / job['id'] / 'captures' / truth_image /
                           'public/observation.json')
        q = refs['questions'][0, :public['public_question_lengths'][0]].tolist()
        a = refs['targets'][0, :public['public_target_lengths'][0]].tolist()
        reference, first_drop, rescued = q + a, None, 0
        base = (set(read_tensors(directory / 'token_candidates.safetensors')['gradient100_ids'].tolist())
                if variant != 'lm_only' else set())
        for depth in range(1, len(reference) + 1):
            state = read_json(directory / 'checkpoints' / f'depth-{depth:03d}.json')
            parent = next((i for i, row in enumerate(state['parents'])
                           if row['tokens'] == reference[:depth - 1]), None)
            offered = parent is not None and reference[depth - 1] in state['pools'][parent]
            if variant == 'dynamic16' and offered and reference[depth - 1] not in base:
                rescued += 1
            survives = any(row['tokens'] == reference[:depth] for row in state['beam'])
            if first_drop is None and not survives:
                first_drop = {'depth': depth, 'reason': 'scored_out' if offered else 'not_proposed'}
        rows.append({'job': job['id'], 'image': image, 'variant': variant,
                     'length_bin': job['length_bins'][0],
                     'question_em': m['question_exact_match'], 'answer_em': m['target_exact_match'],
                     'joint_em': m['question_exact_match'] * m['target_exact_match'],
                     **{k: full[k] for k in ('rouge1', 'rouge2', 'rougeL', 'word_recall')},
                     'seconds': result['costs']['seconds'], 'first_drop': first_drop,
                     'rescued_true_prefix_tokens': rescued,
                     'final_beam_joint_hit': any(r['tokens'] == reference for r in result['provenance']['final_beam']),
                     'final_beam_question_hit': any(r['tokens'][:len(q)] == q for r in result['provenance']['final_beam']),
                     'peak_allocated_bytes': result['provenance'].get('peak_allocated_bytes')})
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['image'], row['variant'])].append(row)
    fields = ('question_em', 'answer_em', 'joint_em', 'rouge1', 'rouge2', 'rougeL', 'word_recall',
              'seconds', 'rescued_true_prefix_tokens', 'final_beam_joint_hit', 'final_beam_question_hit')
    aggregate = [{'image': image, 'variant': variant, 'n': len(items),
                  **{k: statistics.fmean(r[k] for r in items) for k in fields}}
                 for (image, variant), items in buckets.items()]
    write_json(output / 'summary.json', {
        'status': 'completed', 'aggregate': aggregate, 'rows': rows, 'cohort_n': 4, 'attacks': 28,
        'source_sha256': plan['source_sha256'], 'all_reconstructions_committed_before_truth': True,
        'contains_private_tokens': False, 'uses_private_reference': False,
        'note': 'Fixed pilot, two short and two medium QA. No outcomes selected, no independent validation.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'attack', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f'state-{args.phase}.json'
    if state.exists():
        raise FileExistsError('No implicit retry')
    write_json(state, {'status': 'running'})
    try:
        if args.phase == 'prepare':
            if args.source is None:
                raise ValueError('Source is required')
            prepare(args.output, args.source)
        elif args.phase == 'attack':
            attack_all(args.output)
        else:
            evaluate_all(args.output)
        write_json(state, {'status': 'completed'})
    except BaseException:
        write_json(state, {'status': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
