"""Image-private visual-nuisance ablation, with post-commit token evaluation.

The bank consists solely of installed scikit-image example pictures. Scoring
receives public systems, never private images/text. This does not alter defaults.
"""
import argparse
from dataclasses import replace
import gc
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from attacks.analytic.dager_adapted import gradient_groups
from attacks.analytic.dager_filter_refinement import projected_system, sparse_support, spectral_scores
from attacks.analytic.dager_subspace import residual, row_basis
from attacks.analytic.dager_visual_filter import visual_weight
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.fl import load_observation, restore_model
from core.config import DAGEROptions
from evaluation.dager_filter_study import GPU5, TOPKS, _metrics
from evaluation.dager_surrogate_diagnostic import _public_parts


PICTURES = ('astronaut.png', 'camera.png', 'coffee.png', 'chelsea.png', 'rocket.jpg',
            'coins.png', 'moon.png', 'hubble_deep_field.jpg', 'motorcycle_left.png',
            'grass.png', 'brick.png', 'gravel.png')


def variants():
    result = [{'name': 'random_baseline', 'context': 'random', 'sparse': False},
              {'name': 'template_only', 'context': 'template', 'sparse': False}]
    for rank in (64, 256, 512):
        for ridge in (.1, .01, .001):
            result.append({'name': f'visual_r{rank}_tau{ridge}', 'context': 'visual',
                           'rank': rank, 'ridge': ridge, 'sparse': False})
    result.append({'name': 'visual_r512_hard', 'context': 'visual', 'rank': 512,
                   'ridge': 0., 'sparse': False})
    for context, ridge in [('template', None), ('random', None), ('visual', .01), ('visual', .001)]:
        for penalty in (.001, .01):
            result.append({'name': f'joint_{context}_tau{ridge}_lambda{penalty}',
                           'context': context, 'rank': 256, 'ridge': ridge,
                           'sparse': True, 'penalty': penalty})
    return result


@torch.no_grad()
def build_bank(args, output, options):
    import skimage
    image_dir = Path(skimage.__file__).parent / 'data'
    paths = [image_dir / name for name in PICTURES]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('An explicitly planned public example image is missing')
    write_json(output / 'bank_plan.json', {
        'source': 'Installed scikit-image public example images; no private data selection',
        'images': [{'name': p.name, 'sha256': file_hash(p)} for p in paths],
        'second_moment': 'uncentered, each image Frobenius-normalized after template projection',
        'model_sha256': file_hash(Path(args.model) / 'model.safetensors')})
    adapter = restore_model(args.model, 'cuda:0')
    obs = load_observation(Path(args.study) / 'run-00000/captures/private/public', 'cuda:0')
    if obs.public_images is not None or obs.model_fingerprint != adapter.fingerprint():
        raise ValueError('Expected matching image-private observation')
    adapter.set_training_spec(obs.training)
    inputs = []
    for path in paths:
        with Image.open(path) as image:
            pixels = adapter.prepare_image(image.convert('RGB')).unsqueeze(0)
        view = LlavaTextView(adapter, replace(obs, public_images=pixels))
        template, visual = _public_parts(view)
        inputs.append(visual.double())
        print(f'public image encoded: {path.name}', flush=True)
    template_basis, _ = row_basis(template.double(), options.rank_rtol, options.rank_atol)
    covariance = torch.zeros((template.shape[1], template.shape[1]), device='cuda:0', dtype=torch.float64)
    for visual in inputs:
        values = residual(visual, template_basis)
        values = values / values.norm()
        covariance.add_(values.T @ values / len(inputs))
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    bank = {'basis': vectors[:, -512:].flip(1).T, 'eigenvalues': eigenvalues[-512:].flip(0),
            'template': template, 'template_basis': template_basis}
    write_tensors(output / 'public_visual_bank.safetensors', bank)
    write_json(output / 'bank_complete.json', {
        'bank_sha256': file_hash(output / 'public_visual_bank.safetensors'),
        'image_count': len(paths), 'template_rank': len(template_basis),
        'captured_energy': {str(k): float(eigenvalues[-k:].sum() / eigenvalues.sum())
                            for k in (64, 256, 512)}})
    del adapter, view, inputs, obs, covariance, vectors, bank, visual, template, pixels
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def score_system(features, system, bank, options, forbidden, specs, target, expected=None):
    started = time.monotonic()
    gradients = [system[f'gradient_{name}'] for name in 'qkv']
    random = projected_system(gradients, system['public_inputs'], features, options)
    template = projected_system(gradients, bank['template'], features, options)
    baseline = spectral_scores(random[0], random[4]).cpu()
    flags = random[6].cpu()
    # All variants retain exactly the same ambiguity budget in this experiment.
    if not torch.equal(flags, template[6].cpu()):
        raise ValueError('Template and random ambiguity sets differ; budget comparison invalid')
    if expected is not None:
        torch.testing.assert_close(baseline, expected, atol=1e-6, rtol=1e-6)
    allowed = [i for i in range(len(features)) if i not in forbidden and not flags[i]]
    pool = sorted(allowed, key=lambda i: (float(baseline[i]), i))[:2000]
    scores, diagnostics = {'ambiguous': flags, 'pool_ids': torch.tensor(pool)}, {}
    for spec in specs:
        name = spec['name']
        if spec['context'] == 'random':
            dictionary, matrix = random[0], random[2]
        else:
            dictionary, matrix = template[0], template[2]
            if spec['context'] == 'visual':
                rank = spec['rank']
                basis, eigenvalues = bank['basis'][:rank], bank['eigenvalues'][:rank]
                dictionary = visual_weight(dictionary, basis, eigenvalues, spec['ridge'])
                matrix = visual_weight(matrix, basis, eigenvalues, spec['ridge'])
        vectors, _ = row_basis(matrix, options.rank_rtol, options.rank_atol)
        value = spectral_scores(dictionary, vectors).cpu()
        if spec['sparse']:
            strength, diagnostic = sparse_support(
                dictionary, matrix, pool, lambda_ratios=(spec['penalty'],),
                iterations=4000, analysis_dtype=torch.float64)[spec['penalty']]
            strength = strength.cpu()
            value = 1 + baseline.clone()
            active = strength > 1e-7
            value[torch.tensor(pool)[active]] = -strength[active]
            diagnostics[name] = diagnostic
        else:
            diagnostics[name] = {'gradient_rank': len(vectors)}
        scores[f'scores__{name}'] = value
        print(f'{target.name}: {name} scored ({time.monotonic()-started:.1f}s)', flush=True)
    write_tensors(target / 'scores.safetensors', scores)
    write_json(target / 'complete.json', {
        'score_sha256': file_hash(target / 'scores.safetensors'), 'forbidden_ids': sorted(forbidden),
        'diagnostics': diagnostics, 'seconds': time.monotonic() - started})


def evaluate(output, reference_roots):
    """The only function that joins private text, after the full score phase."""
    if not (output / 'scores_complete.json').exists():
        raise RuntimeError('Scores must all commit first')
    rows = []
    for index, ref_root in enumerate(reference_roots):
        root = output / f'run-{index:05d}'
        meta = read_json(root / 'complete.json')
        if file_hash(root / 'scores.safetensors') != meta['score_sha256']:
            raise ValueError('Score hash mismatch')
        tensors = read_tensors(root / 'scores.safetensors')
        refs = read_tensors(ref_root / 'text_tokens.safetensors')
        forbidden = set(meta['forbidden_ids'])
        fields = {field: set(refs[field].flatten().tolist()) - forbidden for field in ('questions', 'targets')}
        truth = fields['questions'] | fields['targets']
        pool = set(tensors['pool_ids'].tolist()) | set(torch.where(tensors['ambiguous'])[0].tolist())
        for key, value in tensors.items():
            if not key.startswith('scores__'):
                continue
            rows.append({'run': index, 'variant': key.removeprefix('scores__'),
                         **_metrics(value, tensors['ambiguous'], forbidden, truth),
                         'pool_complete': truth <= pool, 'pool_recall': len(truth & pool) / len(truth),
                         'by_field': {f: _metrics(value, tensors['ambiguous'], forbidden, ids)
                                      for f, ids in fields.items()}})
    aggregate = []
    for variant in sorted({r['variant'] for r in rows}):
        selected = [r for r in rows if r['variant'] == variant]
        aggregate.append({'variant': variant, 'sample_count': len(selected),
                          'pool_complete': sum(r['pool_complete'] for r in selected),
                          'worst_true_rank_max': max(r['worst_true_rank'] for r in selected),
                          'curve': [{'k': k, 'complete': sum(r['curve'][j]['complete'] for r in selected),
                                     'mean_recall': float(np.mean([r['curve'][j]['recall'] for r in selected])),
                                     'mean_count': float(np.mean([r['curve'][j]['count'] for r in selected]))}
                                    for j, k in enumerate(TOPKS)]})
    report = {'aggregate': aggregate, 'rows': rows}
    write_json(output / 'report.json', report)
    write_json(output / 'complete.json', {'report_sha256': file_hash(output / 'report.json')})
    return report


def development(args):
    output, source, study = Path(args.output), Path(args.source), Path(args.study)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    source_plan = read_json(source / 'plan.json')
    options = DAGEROptions(**source_plan['options'])
    write_json(output / 'plan.json', {
        'source_sha256': source_fingerprint(), 'variants': variants(), 'sample_count': 10,
        'pool': 'Fixed original random-image Top-2000 for every joint fit',
        'source_plan_sha256': file_hash(source / 'plan.json'),
        'primary_k': [50, 100], 'iterations': 4000, 'analysis_dtype': 'float64',
        'selection': 'Max full coverage at 50, then recall at 50, then coverage at 100, then recall at 100.',
        'validation_gate': 'Must improve full coverage at K50 without lower recall at K50 or K100.',
        'validation_cohort': 'Previously unused eval client 0 unique image groups, offsets 10 through 19',
        'policy': 'All development scores committed before reference evaluation; no private images in scoring.'})
    try:
        build_bank(args, output, options)
        bank = read_tensors(output / 'public_visual_bank.safetensors', 'cuda:0')
        features = read_tensors(source / 'public_vocabulary_features.safetensors', 'cuda:0')['features']
        for index in range(10):
            write_json(output / 'progress.json', {'stage': 'development', 'run': index})
            old = source / f'run-{index:05d}/random0'
            system = read_tensors(old / 'system.safetensors', 'cuda:0')
            meta = read_json(old / 'complete.json')
            expected = read_tensors(old / 'scores.safetensors')['scores__baseline']
            score_system(features, system, bank, options, set(meta['forbidden_ids']),
                         variants(), output / f'run-{index:05d}', expected)
        write_json(output / 'scores_complete.json', {'sample_count': 10, 'reference_access': False})
        evaluate(output, [study / f'run-{i:05d}/captures/known/private' for i in range(10)])
        print('Development complete', flush=True)
    except Exception as error:
        write_json(output / 'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


@torch.no_grad()
def observation_system(adapter, obs):
    if obs.public_images is not None or not obs.training.question_private:
        raise ValueError('Expected image and text private')
    pixels = torch.rand(1, 3, obs.model.image_size, obs.model.image_size,
                        generator=torch.Generator(device='cpu').manual_seed(0))
    view = LlavaTextView(adapter, replace(obs, public_images=pixels))
    gradients = [obs.tensors[n] for n in gradient_groups(adapter, obs, 'qkv')[0]]
    return {'public_inputs': view.public_layer0,
            **{f'gradient_{n}': g for n, g in zip('qkv', gradients, strict=True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='outputs/diagnostics/dager_filter_gpu5_20261005_v2')
    parser.add_argument('--study', default='outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100')
    parser.add_argument('--model', default='outputs/federation/slake_llava_f_l_fedavg/round-0010')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Only physical GPU 5 is authorized')
    torch.set_num_threads(1)
    development(args)


if __name__ == '__main__':
    main()
