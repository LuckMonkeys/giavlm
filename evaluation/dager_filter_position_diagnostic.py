"""Post-commit private-reference diagnosis of first-layer token signal.

This module may read reference token IDs only after filter scores commit. Its
outputs are descriptive evaluation diagnostics and must never feed an attack.
"""
import argparse
import os
from pathlib import Path

import torch

from attacks.analytic.dager_adapted import forbidden_token_ids, gradient_groups
from attacks.analytic.dager_filter_refinement import projected_system, spectral_scores
from attacks.analytic.dager_subspace import residual, row_basis
from core.adapters.llava import LlavaTextView
from core.artifacts import read_json, read_tensors, source_fingerprint, write_json
from core.config import DAGEROptions
from core.fl import load_observation, restore_model
from evaluation.dager_filter_study import GPU5, _metrics


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('This diagnostic is authorized on GPU 5 only')
    study, scored, output = Path(args.study), Path(args.scores), Path(args.output)
    count = read_json(scored / 'scores_complete.json')['sample_count']
    if output.exists():
        raise FileExistsError('Private diagnostic output exists')
    write_json(output / 'plan.json', {'scope': 'post-commit private-reference diagnostic',
                                    'source_sha256': source_fingerprint(), 'sample_count': count})
    try:
        adapter = restore_model(args.model, 'cuda:0')
        features = read_tensors(scored / 'public_vocabulary_features.safetensors', 'cuda:0')['features']
        rows = []
        for index in range(count):
            write_json(output / 'progress.json', {'run': index})
            root = study / f'run-{index:05d}'
            obs = load_observation(root / 'captures/known/public', 'cuda:0')
            adapter.set_training_spec(obs.training)
            view = LlavaTextView(adapter, obs)
            references = read_tensors(root / 'captures/known/private/text_tokens.safetensors', 'cuda:0')
            names = gradient_groups(adapter, obs, 'qkv')[0]
            modules = dict(adapter.named_modules())
            parameters = dict(adapter.named_parameters())
            cached, handles = {}, []

            def capture(name):
                def forward(module, inputs, result):
                    cached[name] = {'input': inputs[0].detach()}

                    def backward(gradient):
                        cached[name]['coefficient'] = gradient.detach()
                    result.register_hook(backward)
                return forward

            for name in names:
                handles.append(modules[name.removesuffix('.weight')].register_forward_hook(capture(name)))
            try:
                loss = adapter(obs.public_images.to('cuda:0', adapter.dtype),
                               references['questions'], references['targets'])
                gradients = torch.autograd.grad(loss, [parameters[name] for name in names])
            finally:
                for handle in handles:
                    handle.remove()
            norms, manual_gradients = [], []
            identities = []
            for name, gradient in zip(names, gradients, strict=True):
                inputs = cached[name]['input'][0].double()
                coefficients = cached[name]['coefficient'][0].double()
                manual = coefficients.T @ inputs
                manual_gradients.append(manual)
                identities.append({
                    'projection': name.split('.self_attn.')[1].split('_')[0],
                    'replay_relative_error': float((gradient.double() - obs.tensors[name].to(gradient).double()).norm()
                                                   / obs.tensors[name].double().norm().to(gradient.device)),
                    'matrix_product_relative_error': float((manual - gradient.double()).norm()
                                                            / gradient.double().norm())})
                norms.append(coefficients.norm(dim=1))
            norms = torch.stack(norms).square().sum(0).sqrt()
            public_basis, _ = row_basis(view.public_layer0.double(), 1e-5, 1e-8)
            first_inputs = cached[names[0]]['input'][0].double()
            # Exact row geometry from the replay, compared to the public-only view.
            slots = []
            for field, position, absolute in view.slots:
                projected = residual(first_inputs[absolute:absolute + 1], public_basis)[0]
                token_id = int(references[field][0, position])
                slots.append({'field': field, 'position': position,
                              'is_last_answer': field == 'targets' and position == obs.public_target_lengths[0] - 1,
                              'coefficient_norm': float(norms[absolute]),
                              'projected_input_norm': float(projected.norm()),
                              'signal_proxy': float(norms[absolute] * projected.norm()),
                              'token_id': token_id})
            qnorm = torch.tensor([s['coefficient_norm'] for s in slots if s['field'] == 'questions'])
            last = next(s for s in slots if s['is_last_answer'])
            # Oracle numerical diagnosis only: private activations/coefficient
            # products remove the native gradient GEMM's accumulation/rounding.
            # These synthetic gradients are never exposed to an attack scorer.
            forbidden = forbidden_token_ids(adapter)
            truth_ids = {s['token_id'] for s in slots} - forbidden
            numerical = {}
            with torch.no_grad():
                for tolerance in [1e-5, 1e-7]:
                    projected, _, _, singular, vectors, _, ambiguous = projected_system(
                        manual_gradients, view.public_layer0, features,
                        DAGEROptions(rank_rtol=tolerance))
                    values = spectral_scores(projected, vectors).cpu()
                    numerical[str(tolerance)] = {
                        'rank': len(singular),
                        **_metrics(values, ambiguous.cpu(), forbidden, truth_ids)}
            row = {'run': index, 'loss': float(loss.detach()), 'identities': identities,
                   'last_answer_to_question_median_coefficient': last['coefficient_norm'] / float(qnorm.median()),
                   'positions': slots, 'oracle_float64_product_metrics': numerical}
            rows.append(row)
            write_json(output / f'run-{index:05d}.json', row)
            print(f'run {index}: last/question coefficient ratio '
                  f'{row["last_answer_to_question_median_coefficient"]:.4g}', flush=True)
            cached.clear()
            del gradients, loss, view, references, projected, vectors, manual_gradients
        write_json(output / 'report.json', {'rows': rows, 'scope': 'private-reference diagnostic only'})
        write_json(output / 'complete.json', {'sample_count': len(rows)})
    except Exception as error:
        write_json(output / 'failed.json', {'error_type': type(error).__name__, 'reason': str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--scores', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == '__main__':
    main()
