"""Public-image geometry gate for concept probes and visual-token interfaces.

No target observation, private QA, or private image is consumed. Image property
labels enter only the post-commit evaluator, never vector or rank selection.
"""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

from attacks.analytic.dager_subspace import residual, row_basis
from core.adapters.llava import llava_input_pieces
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import restore_model

ROOT = Path('outputs/diagnostics/visual_concept_probe_20261028')
MODEL = Path('outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-32/round-0010')
GENERIC = Path('outputs/diagnostics/generic_head_n20_gpu5_20261018')
MEDICAL = Path('outputs/diagnostics/synthetic_head_n20_gpu5_20261017')
BANKS = [GENERIC / f'bank-{i}' for i in range(3)] + [MEDICAL]
GPU = 'GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09'
WORDS = {
    'primary': [['head', 'brain', 'skull', 'face'], ['chest', 'abdomen', 'arm', 'leg']],
    'synonyms': [['head', 'cranium', 'forehead', 'scalp'], ['torso', 'belly', 'hand', 'foot']],
    'unrelated': [['table', 'chair', 'door', 'window'], ['river', 'mountain', 'tree', 'cloud']],
}
CONTEXTS = {'neutral': ('Describe the image.', 'The image is shown.'),
            'head_word': ('Is the head visible?', 'yes')}
RANKS = (32, 96, 576)


def verify(hashes):
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def prepare():
    if (ROOT / 'plan.json').exists():
        raise FileExistsError('No overwrite')
    files = [MODEL / n for n in ['model.json', 'model.safetensors']]
    for bank in BANKS:
        files += [bank / 'public' / f'bank.{ext}' for ext in ['json', 'safetensors']]
    import random
    rng = random.Random(20261028)
    other = []
    for b in range(4):
        order = list(range(24))
        rng.shuffle(order)
        mapping = {order[j]: order[(j + 1) % 24] for j in range(24)}
        other.extend([24 * b + mapping[i] for i in range(24)])
    write_json(ROOT / 'plan.json', {
        'scope': 'Public synthetic mechanism screen; no private targets or gradient attack',
        'model': str(MODEL), 'banks': [str(p) for p in BANKS], 'images': 96,
        'words': WORDS, 'contexts': CONTEXTS, 'ranks': list(RANKS),
        'primary': 'Normalized first-LM-input visual residual subspace, neutral context, rank 96, primary word set',
        'controls': ['raw visual space', 'rank 32', 'full visual rank', 'rank-matched random quotient space',
                     'other-image subspace', 'unrelated word clusters', 'permuted concept groups',
                     'synonym sensitivity', 'concept appears in context', 'zero visual tokens'],
        'word_pooling': 'Leading-space lowercase tokenization; normalize each token through first RMSNorm, then mean pieces',
        'distance': 'Relative residual norm; mean valid words per class, at least 2/4 required; lower wins',
        'rank_rtol': 1e-6, 'rank_atol': 1e-10, 'zero_tolerance': 1e-6,
        'random_seed': 20261028, 'readout_question': 'Does this image show a human head? Answer yes or no.',
        'other_image_indices': other,
        'readout_answers': ['yes', 'no'], 'readout': 'Native answer mean NLL including EOS; no threshold fitting',
        'interface_checks': [0, 12, 72, 84], 'interface_tolerance': 1e-6,
        'selection': 'No target-label tuning; all words, ranks and contexts frozen before inference',
        'limits': 'Reused generated public banks with broad visual labels; not real-image validation or property leakage',
        'host': 'WN-A100', 'physical_gpu': 1, 'gpu_uuid': GPU,
        'hashes': {str(p): file_hash(p) for p in files}})


def inject_logits(adapter, visual, questions, targets):
    """Exact native LLaVA hard-slot layout with a supplied visual representation."""
    pieces = llava_input_pieces(adapter, visual, adapter.embedding()(questions),
                               adapter.embedding()(targets[:, :-1]))
    embeddings = torch.cat(pieces, 1)
    return adapter.backend.language_model(inputs_embeds=embeddings, use_cache=False,
                                          return_dict=True).logits[:, -targets.shape[1]:]


def native_hard_loss(logits, targets, pad):
    mask = targets != pad
    # Match VLMAdapter.forward's reduction, including vocabulary-axis summation.
    # Fused cross_entropy is mathematically equal but rounds differently in FP32.
    raw = F.one_hot(targets, logits.shape[-1]).float()
    nll = -(raw * logits.float().log_softmax(-1)).sum(-1)
    return (nll * mask).sum() / mask.sum()


def concept_distance(vectors, basis, template=None, zero_tolerance=1e-6):
    original = vectors.norm(dim=1)
    projected = residual(vectors, template) if template is not None else vectors
    norms = projected.norm(dim=1)
    valid = norms > zero_tolerance * original.clamp_min(torch.finfo(vectors.dtype).tiny)
    distances = residual(projected, basis).norm(dim=1) / norms.clamp_min(torch.finfo(vectors.dtype).tiny)
    return distances.clamp(0, 1), valid


def class_scores(distances, valid):
    output = []
    for start in [0, 4]:
        values = distances[start:start + 4][valid[start:start + 4]]
        if len(values) < 2:
            return None
        output.append(float(values.mean()))
    return output


@torch.no_grad()
def visual_basis(visual, template, rank, rtol=1e-6):
    values = residual(visual.double(), template) if template is not None else visual.double()
    # Gram eigendecomposition avoids a full hidden-dimension projector.
    gram = values @ values.T
    eigen, vectors = torch.linalg.eigh(gram)
    order = torch.arange(len(eigen) - 1, -1, -1, device=eigen.device)
    eigen, vectors = eigen[order].clamp_min(0), vectors[:, order]
    threshold = max(1e-20, float(eigen[0]) * rtol**2)
    count = min(rank, int((eigen > threshold).sum()))
    basis = (vectors[:, :count].T @ values) / eigen[:count, None].sqrt()
    return basis, eigen


def interface_check(adapter, image, visual, question, targets):
    params = adapter.trainable()
    if isinstance(params, dict):
        params = list(params.values())
    else:
        params = list(params)
    with torch.no_grad():
        base_logits = adapter.target_logits(image, adapter.probabilities(question), adapter.probabilities(targets))
        supplied = inject_logits(adapter, visual, question, targets)
        logits_error = float((base_logits - supplied).abs().max())
    baseline_loss = adapter(image, question, targets)
    baseline = [v.detach().float().cpu() for v in torch.autograd.grad(baseline_loss, params)]
    supplied_visual = visual.detach().requires_grad_(True)
    supplied_loss = native_hard_loss(inject_logits(adapter, supplied_visual, question, targets), targets, adapter.pad)
    all_grads = torch.autograd.grad(supplied_loss, [*params, supplied_visual])
    grads, visual_grad = all_grads[:-1], all_grads[-1]
    error = sum(float((a - b.detach().float().cpu()).double().square().sum())
                for a, b in zip(baseline, grads, strict=True))
    norm = sum(float(a.double().square().sum()) for a in baseline)
    result = {'max_logit_abs_error': logits_error,
              'loss_abs_error': abs(float(baseline_loss) - float(supplied_loss)),
              'gradient_relative_l2': (error / max(norm, 1e-60)) ** .5,
              'visual_gradient_norm': float(visual_grad.float().norm()),
              'uploaded_tensors': len(params)}
    if not torch.isfinite(visual_grad).all() or not visual_grad.norm() > 0:
        raise ValueError('Visual tokens have no finite optimization gradient')
    if max(result[k] for k in ['max_logit_abs_error', 'loss_abs_error', 'gradient_relative_l2']) > 1e-6:
        raise ValueError(f'Visual injection interface mismatch: {result}')
    return result


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or torch.cuda.device_count() != 1:
        raise ValueError('Expose only the selected WN-A100 GPU 1')
    if (ROOT / 'started.json').exists():
        raise FileExistsError('No automatic retries')
    write_json(ROOT / 'started.json', {'pid': os.getpid(), 'gpu_uuid': GPU})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        plan = read_json(ROOT / 'plan.json')
        verify(plan['hashes'])
        verify(read_json(ROOT / 'runtime_source.json')['hashes'])
        if torch.cuda.mem_get_info()[0] < 35 * 2**30:
            raise RuntimeError('Insufficient free memory; do not change conditions')
        adapter = restore_model(plan['model'], 'cuda:0')
        fingerprint = adapter.fingerprint()
        versions = {n: p._version for n, p in adapter.named_parameters()}
        norm = adapter.backend.language_model.model.layers[0].input_layernorm
        templates, probes, vocabulary, random_spaces = {}, {}, {}, {}
        with torch.no_grad():
            for context, (q, a) in CONTEXTS.items():
                question = adapter.encode([q], adapter.spec.question_length)
                target = adapter.encode([a], adapter.spec.target_length)
                # Include actual template, question, response and padding slots.
                pieces = llava_input_pieces(adapter, adapter.embedding().weight.new_empty(
                    1, 0, adapter.embedding().weight.shape[1]),
                                            adapter.embedding()(question), adapter.embedding()(target[:, :-1]))
                text_features = norm(torch.cat(pieces, 1))[0].double()
                templates[context] = row_basis(text_features, 1e-6, 1e-10)[0]
            for cluster, groups in WORDS.items():
                vectors, ids = [], []
                for word in sum(groups, []):
                    tokens = adapter.tokenizer.encode(' ' + word, add_special_tokens=False)
                    ids.append(tokens)
                    vectors.append(norm(adapter.embedding()(torch.tensor(tokens, device='cuda:0'))).double().mean(0))
                probes[cluster] = torch.stack(vectors)
                vocabulary[cluster] = {'words': sum(groups, []), 'token_ids': ids}
        write_json(ROOT / 'vocabulary.json', vocabulary)
        basis_hashes, prediction_hashes = {}, {}
        read_q = adapter.encode([plan['readout_question']], adapter.spec.question_length)
        read_targets = [adapter.encode([v], adapter.spec.target_length) for v in plan['readout_answers']]
        row_index = 0
        for b, folder in enumerate(BANKS):
            rows = read_json(folder / 'public/bank.json')
            pixels = read_tensors(folder / 'public/bank.safetensors')
            for row in rows:
                # Deliberately never use row['label'] during prediction.
                from PIL import Image
                image = adapter.prepare_image(Image.fromarray(pixels[row['id']].numpy())).unsqueeze(0).to(adapter.device, adapter.dtype)
                with torch.no_grad():
                    visual = adapter.visual_embeddings(image)[0].detach()
                    features = norm(visual)[0].double()
                    result = {'index': row_index, 'bank': b, 'scores': {}, 'diagnostics': {}}
                    stored = {}
                    for context, template in [('raw', None), *templates.items()]:
                        full, eigen = visual_basis(features, template, 576)
                        stored[context] = full.float().cpu()
                        for rank in RANKS:
                            basis = full[:rank]
                            effective_rank = len(basis)
                            # Shared random space for equal ranks and contexts; no image-label dependence.
                            cache_key = (context, effective_rank)
                            if cache_key not in random_spaces:
                                gen = torch.Generator(device='cuda:0').manual_seed(20261028)
                                random = torch.randn((effective_rank, features.shape[1]), generator=gen,
                                                     device='cuda:0', dtype=torch.float64)
                                if template is not None:
                                    random = residual(random, template)
                                random_spaces[cache_key] = torch.linalg.qr(random.T, mode='reduced')[0].T
                            random = random_spaces[cache_key]
                            for cluster, vectors in probes.items():
                                distances, valid = concept_distance(vectors, basis, template)
                                random_distances, random_valid = concept_distance(vectors, random, template)
                                name = f'{context}_r{rank}_{cluster}'
                                result['scores'][name] = class_scores(distances, valid)
                                result['scores'][name + '_random'] = class_scores(random_distances, random_valid)
                                # One fixed random partition of the primary concepts, preserving 4/4 size.
                                if cluster == 'primary':
                                    permutation = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device='cuda:0')
                                    result['scores'][name + '_permuted'] = class_scores(distances[permutation], valid[permutation])
                                result['diagnostics'][name] = {'valid_words': valid.cpu().tolist(),
                                                              'distances': distances.cpu().tolist()}
                            result['diagnostics'][f'{context}_r{rank}'] = {
                                'rank': effective_rank, 'template_rank': 0 if template is None else len(template),
                                'energy_retained': float(eigen[:effective_rank].sum() / eigen.sum()),
                                'orthogonality_error': float((basis @ basis.T - torch.eye(effective_rank, device='cuda:0')).abs().max())}
                    result['readout'] = [float(native_hard_loss(inject_logits(adapter, visual, read_q, y), y, adapter.pad)) for y in read_targets]
                    result['zero_visual_readout'] = [float(native_hard_loss(inject_logits(adapter, torch.zeros_like(visual), read_q, y), y, adapter.pad)) for y in read_targets]
                if row_index in plan['interface_checks']:
                    result['interface_check'] = interface_check(adapter, image, visual, read_q, read_targets[0])
                basis_path = ROOT / 'bases' / f'{row_index:03d}.safetensors'
                write_tensors(basis_path, stored)
                basis_hashes[str(basis_path)] = file_hash(basis_path)
                prediction = ROOT / 'predictions' / f'{row_index:03d}.json'
                write_json(prediction, result)
                prediction_hashes[str(prediction)] = file_hash(prediction)
                row_index += 1
                write_json(ROOT / 'progress.json', {'completed': row_index, 'images': 96, 'seconds': time.monotonic() - start})
                if row_index == 1:
                    write_json(ROOT / 'first_checkpoint.json', {'prediction_sha256': file_hash(prediction),
                               'interface_check': result['interface_check'], 'private_reference_read': False})
                print(f'Public image {row_index}/96 committed', flush=True)
                del result, stored, visual, features, full, basis, random
                gc.collect()
        if adapter.fingerprint() != fingerprint or any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError('Model changed')
        verify(plan['hashes'])
        verify(read_json(ROOT / 'runtime_source.json')['hashes'])
        write_json(ROOT / 'predictions_complete.json', {'hashes': prediction_hashes, 'basis_hashes': basis_hashes,
                   'predictions': row_index, 'public_labels_read_for_scoring': False, 'private_reference_read': False,
                   'seconds': time.monotonic() - start, 'model_unchanged': True,
                   'peak_mib': torch.cuda.max_memory_allocated() / 2**20})
    except BaseException:
        write_json(ROOT / 'failed.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    args = parser.parse_args()
    globals()[args.stage]()
