"""Image-private token evidence from public counterfactuals and LoRA A/B.

Public calibration uses installed example pictures and authored generic texts,
with same-length token replacements. Private references enter only evaluation.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from attacks.analytic.dager_evidence import (b_energy, evidence_features, fit_logistic,
                                            logistic_score, negative_moments)
from attacks.analytic.dager_filter_refinement import projected_system, rank_vector, spectral_scores
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import DAGEROptions
from core.fl import load_observation, restore_model
from evaluation.dager_filter_study import GPU5, TOPKS, _metrics
from evaluation.dager_visual_filter_study import PICTURES, evaluate


QUESTIONS = ('What objects can be seen in the photograph?', 'Describe the colors of the main object.',
             'Where is the light coming from in this scene?', 'How many separate objects are visible?',
             'What material covers the surface of the object?', 'Is the scene indoors or outdoors?',
             'Which part of the photograph is in focus?', 'Describe the background behind the object.',
             'What shapes appear near the center?', 'How does the texture of the surface look?',
             'Which direction does the object face?', 'What is different between the foreground and background?')
ANSWERS = ('A person wearing a helmet.', 'A camera with a dark body.', 'A cup beside a plate.',
           'A cat with pointed ears.', 'A rocket near a tall structure.', 'Several circular coins.',
           'A bright curved surface.', 'Many small points of light.', 'A motorcycle on a road.',
           'Green plants cover the ground.', 'Red rectangular blocks.', 'Small stones with rough surfaces.')


def names_for(adapter):
    params = dict(adapter.named_parameters())
    names = {}
    for projection in 'qkv':
        for factor in 'AB':
            prefix = f'backend.language_model.model.layers.0.self_attn.{projection}_proj.lora_{factor}.'
            matches = [n for n in params if n.startswith(prefix) and n.endswith('.weight')]
            if len(matches) != 1:
                raise ValueError('Expected one first-layer LoRA factor')
            names[projection + factor] = matches[0]
    return names


@torch.no_grad()
def public_features(adapter, obs, features, options, names):
    if obs.public_images is not None or obs.public_questions or obs.public_targets:
        raise ValueError('Scoring requires image/text-private observation')
    image = torch.rand(1, 3, obs.model.image_size, obs.model.image_size,
                       generator=torch.Generator(device='cpu').manual_seed(0))
    view = LlavaTextView(adapter, replace(obs, public_images=image))
    gradients = [obs.tensors[names[p+'A']] for p in 'qkv']
    projected, _, _, _, basis, _, ambiguous = projected_system(
        gradients, view.public_layer0, features, options)
    baseline = spectral_scores(projected, basis)
    raw = [baseline.clamp_min(1e-7).log()]
    params = dict(adapter.named_parameters())
    audit = {}
    for p in 'qkv':
        a, b = (params[names[p+factor]].detach().double() for factor in 'AB')
        ga, gb = (obs.tensors[names[p+factor]].to(device=a.device, dtype=torch.float64)
                  for factor in 'AB')
        energy, independent, compressed = b_energy(features, a, b, gb)
        raw.append(energy)
        # Projecting away public image/template inputs in this 8-dimensional
        # space can saturate it. Record this instead of claiming a B span filter.
        public_compressed = view.public_layer0.double() @ a.T
        template_mask = view.known_mask.clone()
        template_mask[view.visual_start:view.visual_stop] = False
        template = view.layer_input(view.embeddings[:, template_mask], 0)[0].double() @ a.T
        audit[p] = {
            'b_input_dimension': a.shape[0],
            'b_gradient_rank': int(torch.linalg.matrix_rank(gb, rtol=1e-7)),
            'b_innovation_rank': int(torch.linalg.matrix_rank(independent, rtol=1e-7)),
            'public_input_rank_in_b_coordinates': int(torch.linalg.matrix_rank(public_compressed, rtol=1e-7)),
            'template_rank_in_b_coordinates': int(torch.linalg.matrix_rank(template, rtol=1e-7)),
            'sketch_consistency_relative_error': float((b.T @ gb - ga @ a.T).norm() / (ga @ a.T).norm().clamp_min(1e-20)),
            'innovation_fraction': float(independent.norm() / gb.norm().clamp_min(1e-20))}
        # Deleting any of the leading ten candidates still leaves many inputs.
        pool = torch.argsort(baseline)[:100]
        audit[p]['candidate_rank_after_each_top10_deletion'] = [
            int(torch.linalg.matrix_rank(compressed[torch.cat([pool[:j], pool[j+1:]])], rtol=1e-7))
            for j in range(10)]
    return {'raw': torch.stack(raw, dim=1).cpu(), 'baseline': baseline.cpu(),
            'ambiguous': ambiguous.cpu()}, audit


def group_ids(adapter):
    groups, lookup = [], {}
    for i in range(adapter.vocab_size):
        # Only tokenizer surface equivalence: no private words or corpus lookup.
        surface = (adapter.tokenizer.convert_ids_to_tokens(i) or '').lstrip('▁').casefold()
        key = surface if surface and surface.isalpha() else f'__id_{i}'
        if key not in lookup:
            lookup[key] = len(lookup)
        groups.append(lookup[key])
    return torch.tensor(groups)


def scores_from(raw, baseline, ambiguous, calibration, models, groups, forbidden):
    allowed = [i for i in range(len(baseline)) if i not in forbidden and not ambiguous[i]]
    features = evidence_features(raw, calibration)
    score = {'baseline': baseline, 'null_z': features[:, 4],
             'public_frequency': -calibration['public_frequency']}
    base_rank, z_rank = rank_vector(baseline, allowed), rank_vector(score['null_z'], allowed)
    for weight in (.25, .5, .75):
        score[f'null_mix_{weight}'] = (1-weight)*base_rank + weight*z_rank
    for name, model in models.items():
        score[name] = logistic_score(features, model)
    for base in ('baseline', 'AB_0.01'):
        ranks = rank_vector(score[base], allowed)
        minima = torch.full((int(groups.max())+1,), float(len(ranks)+1), dtype=ranks.dtype)
        minima.scatter_reduce_(0, groups, ranks, reduce='amin', include_self=True)
        score[f'group_{base}'] = minima[groups] + ranks / (len(ranks)+1)
    return score


def save_scores(root, scores, flags, forbidden, audit):
    allowed = [i for i in range(len(flags)) if i not in forbidden and not flags[i]]
    pool = sorted(allowed, key=lambda i: (float(scores['baseline'][i]), i))[:2000]
    write_tensors(root / 'scores.safetensors', {
        **{f'scores__{n}': x for n, x in scores.items()},
        'ambiguous': flags, 'pool_ids': torch.tensor(pool)})
    write_json(root / 'complete.json', {'score_sha256': file_hash(root / 'scores.safetensors'),
                                       'forbidden_ids': sorted(forbidden), 'diagnostics': audit})


def calibrate(args, adapter, base_obs, features, options, names, output):
    import skimage
    paths = [Path(skimage.__file__).parent / 'data' / p for p in PICTURES]
    public_plan = {'pictures': [{'name': p.name, 'sha256': file_hash(p)} for p in paths],
                   'questions': list(QUESTIONS), 'answers': list(ANSWERS), 'train_image_indices': list(range(8)),
                   'heldout_image_indices': list(range(8, 12)), 'seed': 7391,
                   'design': 'Two paired token substitutions per image: question and answer. 48 gradients total; same image and fixed lengths within each pair.'}
    write_json(output / 'public_calibration_plan.json', public_plan)
    if args.calibration_cache:
        cache = Path(args.calibration_cache)
        if read_json(cache / 'public_calibration_plan.json') != public_plan:
            raise ValueError('Cached public calibration design differs')
        if read_json(cache / 'plan.json')['model_sha256'] != read_json(output / 'plan.json')['model_sha256']:
            raise ValueError('Cached calibration model differs')
        write_json(output / 'cache_provenance.json', {
            'source': str(cache), 'plan_sha256': file_hash(cache / 'plan.json'),
            'reason': 'Reuse completed public features after tokenizer-padding boundary failure; no private scores existed.'})
    params = dict(adapter.named_parameters())
    trainable = [params[n] for n in names.values()]
    forbidden = forbidden_token_ids(adapter)
    vocabulary = [i for i in range(adapter.vocab_size) if i not in forbidden
                  and adapter.tokenizer.convert_ids_to_tokens(i).lstrip('▁').isalpha()]
    rng = torch.Generator().manual_seed(7391)
    records = []
    for image_index, path in enumerate(paths):
        if args.calibration_cache:
            for index in range(image_index*4, image_index*4+4):
                old = cache / 'public_calibration' / f'case-{index:03d}'
                values = read_tensors(old / 'features.safetensors')
                metadata = read_json(old / 'public_labels.json')
                if metadata['image_index'] != image_index:
                    raise ValueError('Cached image indexing differs')
                root = output / 'public_calibration' / f'case-{index:03d}'
                write_tensors(root / 'features.safetensors', values)
                write_json(root / 'public_labels.json', metadata)
                records.append((values, set(metadata['present_ids']), image_index))
            print(f'Reused public image group {image_index+1}/12', flush=True)
            continue
        with Image.open(path) as image:
            pixels = adapter.prepare_image(image.convert('RGB')).unsqueeze(0).to('cuda:0', adapter.dtype)
        for role in ('questions', 'targets'):
            q = adapter.encode([QUESTIONS[image_index]], adapter.spec.question_length)
            y = adapter.encode([ANSWERS[image_index]], adapter.spec.target_length)
            slot = int(torch.randint(adapter.content_lengths(q if role == 'questions' else y)[0], (), generator=rng))
            original_ids = set(q.flatten().tolist()+y.flatten().tolist())
            eligible = [i for i in vocabulary if i not in original_ids]
            replacement = eligible[int(torch.randint(len(eligible), (), generator=rng))]
            for changed in (False, True):
                question, target = q.clone(), y.clone()
                if changed:
                    (question if role == 'questions' else target)[0, slot] = replacement
                loss = adapter(pixels, question, target)
                gradients = torch.autograd.grad(loss, trainable)
                tensors = {n: g.detach() for n, g in zip(names.values(), gradients, strict=True)}
                obs = replace(base_obs, tensors=tensors,
                              public_question_lengths=adapter.content_lengths(question),
                              public_target_lengths=adapter.content_lengths(target))
                values, audit = public_features(adapter, obs, features, options, names)
                truth = set(question.flatten().tolist()+target.flatten().tolist())-forbidden
                index = len(records)
                root = output / 'public_calibration' / f'case-{index:03d}'
                write_tensors(root / 'features.safetensors', values)
                write_json(root / 'public_labels.json', {
                    'present_ids': sorted(truth), 'image_index': image_index, 'role': role,
                    'changed': changed, 'position': slot, 'replacement': replacement,
                    'split': 'train' if image_index < 8 else 'heldout', 'audit': audit})
                records.append((values, truth, image_index))
                print(f'public calibration {index+1}/48 saved', flush=True)
                del loss, gradients, tensors, obs
    train = records[:32]
    raw = torch.stack([r[0]['raw'] for r in train])
    present = torch.zeros(raw.shape[:2], dtype=torch.bool)
    for i, (_, truth, _) in enumerate(train):
        present[i, list(truth)] = True
    calibration = negative_moments(raw, present)
    calibration['public_frequency'] = present.double().mean(0)
    write_tensors(output / 'calibration.safetensors', calibration)
    xs, ys = [], []
    for values, truth, _ in train:
        informative_truth = truth - set(torch.where(values['ambiguous'])[0].tolist())
        order = torch.argsort(values['baseline']).tolist()
        valid = [i for i in order if i not in forbidden and not values['ambiguous'][i] and i not in truth]
        random_ids = torch.randperm(len(valid), generator=rng)[:256].tolist()
        negative = set(valid[:256]) | {valid[i] for i in random_ids}
        ids = sorted(informative_truth | negative)
        xs.append(evidence_features(values['raw'], calibration)[ids])
        ys.append(torch.tensor([i in informative_truth for i in ids], dtype=torch.float64))
    models = {}
    for label, columns in [('A', [0, 4]), ('AB', list(range(8))), ('B', [1, 2, 3, 5, 6, 7])]:
        for penalty in (.01, .1):
            name = f'{label}_{penalty}'
            models[name] = (read_tensors(cache / f'model_{name}.safetensors')
                            if args.calibration_cache else
                            fit_logistic(torch.cat(xs), torch.cat(ys), columns, penalty))
            write_tensors(output / f'model_{name}.safetensors', models[name])
    groups = group_ids(adapter)
    write_tensors(output / 'groups.safetensors', {'groups': groups})
    # Held-out public images and text templates do not train any calibrator.
    public_rows = []
    for index, (values, truth, image_index) in enumerate(records[32:], 32):
        scores = scores_from(values['raw'], values['baseline'], values['ambiguous'], calibration,
                             models, groups, forbidden)
        save_scores(output / 'public_holdout' / f'run-{index:05d}', scores, values['ambiguous'], forbidden, {})
        for name, score in scores.items():
            public_rows.append({'case': index, 'image_index': image_index, 'variant': name,
                                **_metrics(score, values['ambiguous'], forbidden, truth)})
    aggregate = []
    for name in scores:
        rows = [r for r in public_rows if r['variant'] == name]
        aggregate.append({'variant': name, 'curve': [
            {'k': k, 'complete': sum(r['curve'][j]['complete'] for r in rows),
             'mean_recall': float(np.mean([r['curve'][j]['recall'] for r in rows]))}
            for j, k in enumerate(TOPKS)]})
    write_json(output / 'public_holdout_report.json', {'sample_count': 16,
                'independent_images': 4, 'paired_cases_not_independent': True,
                'aggregate': aggregate, 'rows': public_rows})
    return calibration, models, groups


def commit_shuffled(output, count):
    for i in range(count):
        root = output / f'run-{i:05d}'
        values = read_tensors(root / 'scores.safetensors')
        other = read_tensors(output / f'run-{(i+1)%count:05d}/scores.safetensors')
        for name in ('baseline', 'AB_0.01'):
            key = f'scores__{name}'
            if key in other:
                values[f'scores__shuffled_{name}'] = other[key]
        write_tensors(root / 'scores.safetensors', values)
        meta = read_json(root / 'complete.json')
        meta['score_sha256'] = file_hash(root / 'scores.safetensors')
        write_json(root / 'complete.json', meta)


def run(args):
    output, study = Path(args.output), Path(args.study)
    if output.exists():
        raise FileExistsError('Use a new evidence study directory')
    options = DAGEROptions(**read_json(study / 'configs/known_residual.json')['attack']['dager'])
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'samples': 10, 'public_calibration_samples': 48,
        'public_train_samples': 32, 'public_holdout_samples': 16, 'seed': 7391,
        'primary_k': [50, 100], 'model_sha256': file_hash(Path(args.model) / 'model.safetensors'),
        'features': ['log_A_distance', 'log_B_innovation_energy_qkv', 'token_specific_null_z_of_all_four'],
        'B_scope': 'Statistical energy features plus span saturation/deletion audit; no exact recovery claim.',
        'group_scope': 'Whitespace/case tokenizer surface groups, no private dictionary.',
        'null_floor_ratio': .1, 'logistic_l2': [.01, .1],
        'selection': 'Within A-calibration, AB-calibration and token-group families: full coverage K50, recall K50, full K100, recall K100.',
        'fresh_validation': 'Twenty eval client-0 unique image groups, offsets 20--39, frozen after development.',
        'policy': 'Public calibration labels only for fitting. All development scores and shuffled controls commit before private-token evaluation.'})
    try:
        adapter = restore_model(args.model, 'cuda:0')
        obs = load_observation(study / 'run-00000/captures/private/public', 'cuda:0')
        if obs.model_fingerprint != adapter.fingerprint():
            raise ValueError('Model mismatch')
        adapter.set_training_spec(obs.training)
        with torch.no_grad():
            features = adapter.backend.language_model.model.layers[0].input_layernorm(adapter.embedding().weight).detach()
        names = names_for(adapter)
        pilot, pilot_audit = public_features(adapter, obs, features, options, names)
        expected = read_tensors(Path(args.original) / 'run-00000/random0/scores.safetensors')['scores__baseline']
        torch.testing.assert_close(pilot['baseline'], expected, atol=1e-6, rtol=1e-6)
        write_tensors(output / 'pilot_public_features.safetensors', pilot)
        write_json(output / 'pilot_audit.json', pilot_audit)
        print('Loaded public-observation A/B scoring pilot passed; original baseline reproduced', flush=True)
        calibration, models, groups = calibrate(args, adapter, obs, features, options, names, output)
        forbidden = forbidden_token_ids(adapter)
        for index in range(10):
            write_json(output / 'progress.json', {'stage': 'private_development_scoring', 'run': index})
            obs = load_observation(study / f'run-{index:05d}/captures/private/public', 'cuda:0')
            values, audit = public_features(adapter, obs, features, options, names)
            expected = read_tensors(Path(args.original) / f'run-{index:05d}/random0/scores.safetensors')['scores__baseline']
            torch.testing.assert_close(values['baseline'], expected, atol=1e-6, rtol=1e-6)
            scores = scores_from(values['raw'], values['baseline'], values['ambiguous'], calibration,
                                 models, groups, forbidden)
            root = output / f'run-{index:05d}'
            write_tensors(root / 'features.safetensors', values)
            save_scores(root, scores, values['ambiguous'], forbidden, audit)
            print(f'development {index+1}/10 saved', flush=True)
        commit_shuffled(output, 10)
        write_json(output / 'scores_complete.json', {'sample_count': 10})
        evaluate(output, [study / f'run-{i:05d}/captures/known/private' for i in range(10)])
        print('Evidence development complete', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--study', default='outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
    parser.add_argument('--original', default='outputs/diagnostics/dager_filter_gpu5_20261005_v2')
    parser.add_argument('--model', default='outputs/federation/slake_llava_f_l_fedavg/round-0010')
    parser.add_argument('--calibration-cache')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('GPU 5 only')
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
