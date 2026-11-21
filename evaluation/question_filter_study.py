"""Frozen six-family Stage-1 question study; every score precedes target truth."""
import argparse
from dataclasses import replace
from pathlib import Path
import shutil
import statistics
import time
import traceback

import torch
from safetensors import safe_open

from attacks.analytic.dager_adapted import forbidden_token_ids
from attacks.analytic.dager_subspace import SpanFilter, residual, row_basis
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.config import DAGEROptions, ModelSpec, TrainingSpec
from core.fl import restore_model
from core.types import Observation
from evaluation.last_b_token_readout import prepare as prepare_base
from evaluation.question_filter_math import (blend, energy, metrics, rank, rank_consensus,
                                             support_energy)
from evaluation.question_filter_public import (CONTEXT_MODULES, capture_public, defended,
    first_pool, public_manifest, question_prior, shallow_inputs)
from evaluation.soft_residual_filter_study import verify_plan


CONDITIONS = ('fedsgd-b1', 'noise', 'sparsify')
IMAGES = ('random', 'gt')
PREFIX = 'backend.language_model.model.'
FIRST_A = {p: PREFIX + f'layers.0.self_attn.{p}_proj.lora_A.default.weight' for p in 'qkv'}
VB = PREFIX + 'layers.0.self_attn.v_proj.lora_B.default.weight'
VA = FIRST_A['v']
CONTEXT_A = {k: PREFIX + v + '.lora_A.default.weight' for k, v in CONTEXT_MODULES.items()}
RHO = (.01, .1, 1.)


def prepare(root, source, previous):
    prepare_base(root, source)
    plan = read_json(root / 'plan.json')
    old = read_json(previous / 'plan.json')
    if old['model_fingerprint'] != plan['model_fingerprint']:
        raise ValueError('Historical B model mismatch')
    required = set(FIRST_A.values()) | {VB} | set(CONTEXT_A.values())
    for condition in CONDITIONS:
        for index in range(30):
            path = source / 'runs' / f'{condition}-{index:02d}' / 'captures/random/public/observation.json'
            if not required <= set(read_json(path)['parameter_names']):
                raise ValueError('An experimental module was not uploaded')
    files = {}
    for condition in CONDITIONS:
        for index in range(30):
            job = f'{condition}-{index:02d}'
            path = previous / 'scores' / job / 'scores.safetensors'
            files[job] = file_hash(path)
    shutil.copyfile(previous / 'prior.safetensors', root / 'historical-prior.safetensors')
    rows = public_manifest()
    write_json(root / 'public-plan.json', rows)
    plan.update(previous=str(previous.resolve()), previous_scores=files,
        required_modules=sorted(required), context_modules=CONTEXT_A, rho_grid=list(RHO),
        public_plan_sha256=file_hash(root / 'public-plan.json'),
        image_order=list(IMAGES), conditions=list(CONDITIONS), no_target_truth_until_all_candidates=True,
        confirmation_gate={'exclusive_gain': .05, 'answer_loss_tolerance': .02},
        score_protocol='six independent families; fixed model and original uploads',
        public_prior_hash=file_hash(root / 'historical-prior.safetensors'))
    write_json(root / 'plan.json', plan)


def observation(plan, condition, index, image, device='cuda:0'):
    job = f'{condition}-{index:02d}'
    folder = Path(plan['source']) / 'runs' / job / 'captures' / image / 'public'
    if file_hash(folder / 'observation.json') != plan['sources'][job + ':' + image]['metadata_sha256']:
        raise ValueError('Observation metadata changed')
    meta = read_json(folder / 'observation.json')
    with safe_open(str(folder / 'update.safetensors'), framework='pt', device='cpu') as stream:
        tensors = {name: stream.get_tensor(name) for name in plan['required_modules']}
    images = read_tensors(folder / 'public_images.safetensors')['images'].to(device) if image == 'gt' else None
    obs = Observation(model=ModelSpec(**meta['model']), training=TrainingSpec(**meta['training']),
        tensors=tensors, model_fingerprint=meta['model_fingerprint'], public_images=images,
        public_question_lengths=meta['public_question_lengths'], public_target_lengths=meta['public_target_lengths'])
    obs.validate()
    if obs.model_fingerprint != plan['model_fingerprint']:
        raise ValueError('Observation model differs')
    return obs


def baseline(root, plan, condition, index, image):
    job = f'{condition}-{index:02d}'
    path = Path(plan['source']) / 'runs' / job / 'attacks' / image / 'token_candidates.safetensors'
    if file_hash(path) != plan['baselines'][job + ':' + image]:
        raise ValueError('Historical baseline changed')
    return read_tensors(path)


def b_scores(root, plan, condition, index):
    job = f'{condition}-{index:02d}'
    path = Path(plan['previous']) / 'scores' / job / 'scores.safetensors'
    if file_hash(path) != plan['previous_scores'][job]:
        raise ValueError('Historical B score changed')
    with safe_open(str(path), framework='pt', device='cpu') as stream:
        return {name: stream.get_tensor(name) for name in ('l31_down', 'l32_o')}


def view_for(adapter, obs, image, seed):
    adapter.set_training_spec(obs.training)
    pixels = (obs.public_images if image == 'gt' else torch.rand(
        1, 3, obs.model.image_size, obs.model.image_size, generator=torch.Generator().manual_seed(seed)))
    return LlavaTextView(adapter, replace(obs, public_images=pixels))


def save_bundle(path, values, **meta):
    if path.exists():
        raise FileExistsError('No implicit retry: ' + str(path))
    write_tensors(path, values)
    write_json(path.with_suffix('.json'), dict(meta, sha256=file_hash(path), time=time.time()))


def load_verified(path):
    if file_hash(path) != read_json(path.with_suffix('.json'))['sha256']:
        raise ValueError('Changed score or candidate commitment')
    return read_tensors(path)


def model(root):
    plan = verify_plan(root)
    adapter = restore_model(Path(plan['model']), 'cuda:0')
    if adapter.fingerprint() != plan['model_fingerprint']:
        raise ValueError('Model differs from original capture')
    return plan, adapter, DAGEROptions(**plan['options']), forbidden_token_ids(adapter)


def public_stage(root):
    plan, adapter, options, forbidden = model(root)
    obs = observation(plan, 'fedsgd-b1', 0, 'random')
    adapter.set_training_spec(obs.training)
    rows = read_json(root / 'public-plan.json')
    if file_hash(root / 'public-plan.json') != plan['public_plan_sha256']:
        raise ValueError('Public authored inputs changed')
    capture_public(adapter, rows, {p: FIRST_A[p] for p in 'kv'}, root / 'public/captures')
    with torch.no_grad():
        x = adapter.backend.language_model.model.layers[0].input_layernorm(adapter.embedding().weight).double()
        view = view_for(adapter, obs, 'random', plan['seed'])
        basis, _ = row_basis(view.public_layer0.double(), options.rank_rtol, options.rank_atol)
        projected = residual(x, basis)
        flags = projected.norm(dim=1) <= options.zero_tolerance * x.norm(dim=1)
        amb = set(torch.nonzero(flags).flatten().tolist()) - forbidden
        results = []
        for index, row in enumerate(rows):
            source = read_tensors(root / 'public/captures' / f'{index:02d}.safetensors')
            for condition in CONDITIONS:
                save_bundle(root / 'public/defended' / condition / f'{index:02d}.safetensors',
                            defended(source, condition, 8123 + index), source_public=True)
            g = load_verified(root / 'public/defended/sparsify' / f'{index:02d}.safetensors')
            scores = {}
            diagnostics = {}
            for rho in RHO:
                scores[str(rho)] = sum(support_energy(g[p].to(x), x, basis, rho=rho)[0]
                                        for p in 'kv') / 2
            scores['row'] = sum(energy(residual(g[p].to(x), basis), projected, rowwise=True)
                                 for p in 'kv') / 2
            save_bundle(root / 'public/scores' / f'{index:02d}.safetensors', scores,
                        split=row['split'], diagnostics=diagnostics)
            # Only attacker-authored public labels are used here, never target references.
            q = set(adapter.tokenizer.encode(row['question'], add_special_tokens=False)) - forbidden
            a = set(adapter.tokenizer.encode(row['answer'], add_special_tokens=False)) - forbidden
            for method, value in scores.items():
                chosen = blend([rank(value, flags, forbidden)], [100], amb)
                results.append(dict(split=row['split'], method=method, **metrics(chosen, q, a)))
            print('public rho calibration', index + 1, '/48', flush=True)
        def average(method, split, field):
            values = [r[field] for r in results if r['method'] == method and r['split'] == split
                      and r[field] is not None]
            if not values:
                raise ValueError('No applicable public calibration examples')
            return statistics.fmean(values)
        selected = max(RHO, key=lambda rho: (average(str(rho), 'fit', 'exclusive_recall'),
            average(str(rho), 'fit', 'question_full_coverage'), rho))
        passed = (average(str(selected), 'audit', 'exclusive_recall') >=
                  average('row', 'audit', 'exclusive_recall') and
                  average(str(selected), 'audit', 'answer_recall') >=
                  average('row', 'audit', 'answer_recall') - .02)
        write_json(root / 'calibration.json', {'rho': selected, 'audit_passed': passed,
            'rows': results, 'private_reference_used': False, 'fit_families': 8, 'audit_families': 4,
            'selection': 'public fit exclusive recall, question coverage, larger rho tie-break'})
    if adapter.fingerprint() != plan['model_fingerprint']:
        raise ValueError('Public captures mutated model')


@torch.no_grad()
def prior_stage(root):
    plan, adapter, _, forbidden = model(root)
    blind, traces = question_prior(adapter, ['unknown'])
    save_bundle(root / 'priors/blind.safetensors', {'scores': blind}, traces=traces)
    pool, logits = first_pool(adapter, forbidden)
    save_bundle(root / 'first/pool.safetensors', {'pool': pool, 'logits': logits})
    # Fixed signed permutation is an observable coordinate control, never selected by truth.
    for condition in CONDITIONS:
        for index in range(30):
            scores = b_scores(root, plan, condition, index)['l31_down']
            flags = baseline(root, plan, condition, index, 'random')['ambiguous']
            ids = rank(scores, flags, forbidden)[:5]
            pieces = [adapter.tokenizer.decode([i], skip_special_tokens=False) for i in ids]
            value, traces = question_prior(adapter, pieces)
            save_bundle(root / 'priors' / f'{condition}-{index:02d}.safetensors', {'scores': value},
                        hint_ids=ids, traces=traces)
            write_json(root / 'progress.json', {'phase': 'E1', 'condition': condition,
                                              'index': index, 'time': time.time()})
            print('prior', condition, index, flush=True)


@torch.no_grad()
def matrix_stage(root):
    plan, adapter, options, forbidden = model(root)
    rho = read_json(root / 'calibration.json')['rho']
    decoder = adapter.backend.language_model.model
    x = decoder.layers[0].input_layernorm(adapter.embedding().weight).detach().double()
    params = dict(adapter.named_parameters())
    generator = torch.Generator().manual_seed(20261105)
    permutation = torch.randperm(4096, generator=generator)
    sign = (torch.randint(2, (4096,), generator=generator) * 2 - 1).to(x)
    # Effective weights include the trained LoRA contribution.
    module = decoder.layers[0].self_attn.v_proj
    w = module.base_layer.weight.detach().double() + module.scaling['default'] * (
        module.lora_B.default.weight.detach().double() @ module.lora_A.default.weight.detach().double())
    forward = x @ w.T
    permuted = forward[:, permutation.to(x.device)] * sign
    compressed = x @ params[VA].detach().double().T
    head = adapter.backend.language_model.lm_head.weight.detach().double() * decoder.norm.weight.detach().double()
    required = sorted(set(FIRST_A.values()) | {VB})
    for image in IMAGES:
        for index in range(30):
            obs0 = observation(plan, 'fedsgd-b1', index, image)
            view = view_for(adapter, obs0, image, plan['seed'])
            public, _ = row_basis(view.public_layer0.double(), options.rank_rtol, options.rank_atol)
            projected = residual(x, public)
            flags = (projected.norm(dim=1) <= options.zero_tolerance * x.norm(dim=1)).cpu()
            for condition in CONDITIONS:
                started = time.monotonic()
                obs = obs0 if condition == 'fedsgd-b1' else observation(plan, condition, index, image)
                g = {n: obs.tensors[n].to(x) for n in required}
                original = baseline(root, plan, condition, index, image)
                check = SpanFilter.build([g[FIRST_A[p]] for p in 'qkv'], view.public_layer0, options)
                computed, computed_flags = check.score(x)
                delta = float((computed.cpu() - original['scores']).abs().max())
                if (delta > 1e-6 or not torch.equal(flags, original['ambiguous']) or
                        not torch.equal(computed_flags.cpu(), flags)):
                    raise ValueError(f'Historical baseline differs: {delta}')
                values, diagnostics = {}, {'baseline_max_delta': delta, 'rho': rho}
                for p in 'qkv':
                    values['energy_' + p] = energy(residual(g[FIRST_A[p]], public), projected)
                values['energy_kv'] = (values['energy_k'] + values['energy_v']) / 2
                values['energy_qkv'] = sum(values['energy_' + p] for p in 'qkv') / 3
                values['row_kv'] = sum(energy(residual(g[FIRST_A[p]], public), projected,
                    rowwise=True) for p in 'kv') / 2
                donor = observation(plan, condition, (index + 1) % 30, image)
                for p in 'qkv':
                    values['wrong_energy_' + p] = energy(residual(donor.tensors[FIRST_A[p]].to(x),
                                                                  public), projected)
                values['wrong_energy_kv'] = (values['wrong_energy_k'] + values['wrong_energy_v']) / 2
                values['wrong_energy_qkv'] = sum(values['wrong_energy_' + p] for p in 'qkv') / 3
                if condition == 'sparsify':
                    for label, amount in (('postmask', None), ('ridge', rho)):
                        scored = [support_energy(g[FIRST_A[p]], x, public, rho=amount,
                                                  projected=projected) for p in 'kv']
                        values[label] = (scored[0][0] + scored[1][0]) / 2
                        diagnostics[label] = {p: item[1] for p, item in zip('kv', scored, strict=True)}
                for label, feature in (('b_forward', forward), ('b_permuted', permuted),
                                        ('b_head', head)):
                    values[label] = energy(g[VB].T, feature)
                values['b_input'] = energy(g[VB], compressed)
                save_bundle(root / 'scores' / image / f'{condition}-{index:02d}.safetensors',
                    {k: v.cpu() for k, v in values.items()}, forbidden_ids=sorted(forbidden),
                    diagnostics=diagnostics, seconds=time.monotonic() - started)
                write_json(root / 'progress.json', {'phase': 'E2-E3-E6', 'image': image,
                    'index': index, 'condition': condition, 'time': time.time()})
                print('matrix', image, condition, index, 'baseline delta', delta, flush=True)
                del donor
            del obs0, obs, view, g, projected
            torch.cuda.empty_cache()


@torch.no_grad()
def context_stage(root):
    plan, adapter, options, _ = model(root)
    pool = load_verified(root / 'first/pool.safetensors')['pool']
    for image in IMAGES:
        cached = None
        cached_prefix = None
        for index in range(30):
            obs0 = observation(plan, 'fedsgd-b1', index, image)
            view = view_for(adapter, obs0, image, plan['seed'])
            if image == 'random' and cached is not None:
                if not torch.equal(view.embeddings[:, :view.slots[0][2]], cached_prefix):
                    raise ValueError('Cached Random prefix differs')
                public, features = cached
            else:
                public, features = shallow_inputs(view, pool)
                if image == 'random':
                    cached = public, features
                    cached_prefix = view.embeddings[:, :view.slots[0][2]].clone()
            bases = {key: row_basis(value, options.rank_rtol, options.rank_atol)[0]
                     for key, value in public.items()}
            projected = {k: residual(v, bases[k]) for k, v in features.items()}
            for condition in CONDITIONS:
                obs = obs0 if condition == 'fedsgd-b1' else observation(plan, condition, index, image)
                values = {key: energy(residual(obs.tensors[name].to(features[key]), bases[key]), projected[key])
                          for key, name in CONTEXT_A.items()}
                donor = observation(plan, condition, (index + 1) % 30, image)
                values.update({'wrong_' + key: energy(residual(donor.tensors[name].to(features[key]),
                    bases[key]), projected[key]) for key, name in CONTEXT_A.items()})
                save_bundle(root / 'first' / image / f'{condition}-{index:02d}.safetensors',
                            {**values, 'pool': pool}, public_rank={k: len(b) for k, b in bases.items()})
                write_json(root / 'progress.json', {'phase': 'E4', 'image': image,
                    'index': index, 'condition': condition, 'time': time.time()})
                print('context', image, condition, index, flush=True)
            del obs, obs0, view, donor
            torch.cuda.empty_cache()


def candidates_stage(root):
    plan = verify_plan(root)
    blind = load_verified(root / 'priors/blind.safetensors')['scores']
    historical = read_tensors(root / 'historical-prior.safetensors')['lm_prior']
    total = 0
    for condition in CONDITIONS:
        for index in range(30):
            job, donor_job = f'{condition}-{index:02d}', f'{condition}-{(index+1)%30:02d}'
            b = b_scores(root, plan, condition, index)
            matched = load_verified(root / 'priors' / (job + '.safetensors'))['scores']
            wrong = load_verified(root / 'priors' / (donor_job + '.safetensors'))['scores']
            for image in IMAGES:
                original = baseline(root, plan, condition, index, image)
                flags = original['ambiguous']
                meta = read_json((root / 'scores' / image / (job + '.safetensors')).with_suffix('.json'))
                forbidden = set(meta['forbidden_ids'])
                ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
                if len(ambiguous) != 9:
                    raise ValueError('Common ambiguity budget changed')
                base_order = rank(original['scores'], flags, forbidden)
                b_order = rank(b['l31_down'], flags, forbidden)
                values = {'baseline': original['scores'], 'q_blind': blind, 'q_matched': matched,
                          'q_wrong': wrong, 'historical_prior': historical}
                matrices = load_verified(root / 'scores' / image / (job + '.safetensors'))
                values.update(matrices)
                donor = load_verified(root / 'scores' / image / (donor_job + '.safetensors'))
                values.update({'wrong_' + k: v for k, v in donor.items()
                               if not k.startswith('wrong_') and 'wrong_' + k not in values})
                orders = {k: rank(v, flags, forbidden) for k, v in values.items()}
                chosen = {k: blend([order], [100], ambiguous) for k, order in orders.items()}
                if chosen['baseline'] != set(original['selected_ids'].tolist()):
                    raise ValueError('Historical candidate set differs')
                chosen['b_anchor'] = blend([base_order, b_order], [80, 20], ambiguous)
                for name in ('q_blind', 'q_matched', 'q_wrong'):
                    chosen['bridge_' + name] = blend([base_order, b_order, rank(values[name], flags, forbidden)],
                                                     [60, 20, 20], ambiguous)
                mix_names = set(matrices) | {'wrong_' + name for name in matrices if not name.startswith('wrong_')}
                for name in sorted(mix_names):
                    chosen['mix_' + name] = blend([rank(values[name], flags, forbidden), b_order],
                                                  [80, 20], ambiguous)
                consensus = rank_consensus(b['l31_down'], b['l32_o'], flags, forbidden)
                for kind, score in (('single', b['l31_down']), ('consensus', consensus)):
                    for policy in ('8', '20', 'dynamic'):
                        length = read_json(Path(plan['source']) / 'runs' / job /
                            'captures' / image / 'public/observation.json')['public_target_lengths'][0]
                        quota = min(18, max(8, 3 * length)) if policy == 'dynamic' else int(policy)
                        chosen[f'quota_{kind}_{policy}'] = blend([base_order, rank(score, flags, forbidden)],
                                                               [100-quota, quota], ambiguous)
                if {len(v) for v in chosen.values()} != {109}:
                    raise ValueError('Unequal candidate budgets')
                save_bundle(root / 'candidates' / image / (job + '.safetensors'),
                    {k: torch.tensor(sorted(v), dtype=torch.long) for k, v in chosen.items()},
                    forbidden_ids=sorted(forbidden), methods=sorted(chosen))
                positions = {}
                for key, order in orders.items():
                    ranks = torch.zeros(len(flags), dtype=torch.int32)
                    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.int32)
                    positions[key] = ranks
                save_bundle(root / 'ranks' / image / (job + '.safetensors'), positions)
                total += len(chosen)
    write_json(root / 'all-candidates-complete.json', {'groups': 180, 'candidate_conditions': total,
        'status': 'completed', 'all_scores_before_target_truth': True, 'time': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'public', 'prior', 'matrix', 'context', 'candidates'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / ('state-' + args.phase + '.json')
    if state.exists():
        raise FileExistsError('No implicit phase retry')
    started = time.time()
    write_json(state, {'status': 'running', 'started': started})
    try:
        if args.phase == 'prepare':
            prepare(args.output, args.source, args.previous)
        else:
            globals()[args.phase + '_stage'](args.output)
        write_json(state, {'status': 'completed', 'started': started, 'finished': time.time()})
    except BaseException:
        write_json(state, {'status': 'failed', 'started': started, 'finished': time.time(),
                           'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
