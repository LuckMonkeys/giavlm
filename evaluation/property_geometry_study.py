"""Frozen replays and candidate-only selection for the property geometry pilot."""
import argparse
from dataclasses import asdict
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation, restore_model, simulate_update
from core.types import Batch
from evaluation.dager_rank_study import GPU5
from evaluation.image_property_study import text_key, validate_text
from evaluation.property_geometry import score_geometry

ROOT = Path('outputs/diagnostics/property_geometry_20261028')
SOURCE = Path('outputs/diagnostics/synthetic_head_n20_gpu5_20261017')
GENERIC = Path('outputs/diagnostics/generic_head_n20_gpu5_20261018')
BANKS = [GENERIC / f'bank-{i}' for i in range(3)] + [SOURCE]


def verify(hashes):
    for path, sha in hashes.items():
        if file_hash(path) != sha:
            raise ValueError(f'Frozen input changed: {path}')


def prepare():
    if (ROOT / 'plan.json').exists():
        raise FileExistsError('Use fresh output; no overwrite')
    old = read_json(SOURCE / 'plan.json')
    files = []
    for base in BANKS:
        files += [base / 'public' / f'bank.{ext}' for ext in ['json', 'safetensors']]
        for phase in ['reconstructed', 'oracle']:
            marker = read_json(base / phase / 'complete.json')
            for relative, sha in marker['hashes'].items():
                if file_hash(base / relative) != sha:
                    raise ValueError('Previous result changed')
            files += list((base / phase / 'predictions').glob('*.json'))
            files += list((base / phase / 'scores').glob('*.json'))
    for i in range(20):
        if file_hash(SOURCE / 'captures' / f'run-{i:05d}' / 'private/update.safetensors') != file_hash(
                SOURCE / 'captures' / f'run-{i:05d}' / 'oracle/update.safetensors'):
            raise ValueError('Private and text_known uploads differ')
        for condition in ['private', 'oracle']:
            files += [SOURCE / 'captures' / f'run-{i:05d}' / condition / name
                      for name in ['observation.json', 'update.safetensors']]
        files += [SOURCE / 'reconstruction' / f'run-{i:05d}' / name
                  for name in ['text_tokens.safetensors', 'result.json']]
    files += [Path(old['model']) / name for name in ['model.json', 'model.safetensors']]
    files += [SOURCE / 'report.json', GENERIC / 'report.json']
    hashes = {str(p): file_hash(p) for p in set(files)}
    write_json(ROOT / 'plan.json', {
        'date': '2026-10-28', 'scope': 'Repeated-target mechanism diagnostic; not new validation',
        'model': old['model'], 'targets': 20, 'banks': [str(p) for p in BANKS],
        'classes': old['classes'], 'phases': ['reconstructed', 'oracle'], 'top_m': 3,
        'frozen_hashes': hashes, 'physical_gpu': 5, 'gpu_uuid': GPU5,
        'selection': 'Reconstructed QA only; mean balanced accuracy on other two generic banks; '
                     'equal weight per unique QA, fitting bank, heldout bank and class; deterministic grid order breaks ties',
        'weighted_grid': ['weighted_b0', 'weighted_b0.5', 'weighted_b1'],
        'suppression_grid': ['prototype_equal', 'suppress_k1_a0.5', 'suppress_k1_a1',
                             'suppress_k4_a0.5', 'suppress_k4_a1'],
        'fisher_grid': ['fisher_r0.1', 'fisher_r1'],
        'primary': 'Selected weighting vs centered_equal_tensor on mean of all three generic banks',
        'secondary': 'Suppression vs prototype_equal; Fisher; medical bank; oracle text; shuffled weights/labels; wrong gradient',
        'constraints': 'No new images, training, text search, target-label tuning or reference initialization; '
                       'per-tensor float64 Gram geometry; no dense parameter-space projector',
        'domain_limit': 'Generic heldouts vary seeds, not domain. Medical targets remain diagnostic only.'})


def collect():
    migration = read_json(ROOT / 'migration.json') if (ROOT / 'migration.json').exists() else None
    gpu = migration['gpu_uuid'] if migration else GPU5
    marker = 'started-gpu6.json' if migration else 'started.json'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or torch.cuda.device_count() != 1:
        raise ValueError('Expose only explicitly authorized GPU')
    if (ROOT / marker).exists():
        raise FileExistsError('No automatic retry')
    plan = read_json(ROOT / 'plan.json')
    write_json(ROOT / marker, {'pid': os.getpid(), 'plan_sha256': file_hash(ROOT / 'plan.json'), 'gpu_uuid': gpu})
    source_manifest = ROOT / ('runtime_source_gpu6.json' if migration else 'runtime_source.json')
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        verify(plan['frozen_hashes'])
        verify(read_json(source_manifest)['hashes'])
        if torch.cuda.mem_get_info()[0] < 35 * 2**30:
            raise RuntimeError('Less than fixed 35 GiB headroom; no condition changes')
        adapter = restore_model(plan['model'], 'cuda:0')
        fingerprint = adapter.fingerprint()
        versions = {n: p._version for n, p in adapter.named_parameters()}
        images, bank_rows = [], []
        for b, folder in enumerate(BANKS):
            rows = read_json(folder / 'public/bank.json')
            raw = read_tensors(folder / 'public/bank.safetensors')
            for row in rows:
                images.append(adapter.prepare_image(Image.fromarray(raw[row['id']].numpy()))
                              .unsqueeze(0).to(adapter.device, adapter.dtype))
                bank_rows.append({'bank': b, 'label': row['label'], 'id': row['id']})
        cache, references, replays = {}, [], 0
        if migration:
            verify(migration['committed_grams'])
            for path, sha in migration['committed_grams'].items():
                cache[Path(path).stem] = sha
            replays = migration['preserved_replays']
        for phase in plan['phases']:
            obs = [load_observation(SOURCE / 'captures' / f'run-{i:05d}' /
                                    ('private' if phase == 'reconstructed' else 'oracle'), 'cpu') for i in range(20)]
            for o in obs:
                if (o.model_fingerprint != fingerprint or o.sample_count != 1
                        or o.training.algorithm != 'fedsgd' or not o.training.token_lengths_known
                        or asdict(o.training) != asdict(obs[0].training)
                        or o.training.knowledge != ('private' if phase == 'reconstructed' else 'text_known')):
                    raise ValueError('Observation mismatch')
                if phase == 'reconstructed' and (o.public_questions or o.public_targets):
                    raise ValueError('Private observation exposes text')
            adapter.set_training_spec(obs[0].training)
            for i, o in enumerate(obs):
                if phase == 'reconstructed':
                    folder = SOURCE / 'reconstruction' / f'run-{i:05d}'
                    if read_json(folder / 'result.json')['status'] != 'completed':
                        raise ValueError('This frozen pilot expects completed reconstruction')
                    text = read_tensors(folder / 'text_tokens.safetensors', 'cuda:0')
                else:
                    text = {'questions': torch.tensor(o.public_question_ids, device='cuda:0'),
                            'targets': torch.tensor(o.public_target_ids, device='cuda:0')}
                validate_text(adapter, text, o)
                key = text_key(text)
                if key not in cache:
                    gradients = []
                    for j, image in enumerate(images):
                        g = simulate_update(adapter, Batch(image, text['questions'], text['targets']), o.training)
                        gradients.append({n: v.detach().float().cpu() for n, v in g.items()})
                        del g
                        replays += 1
                        if (j + 1) % 24 == 0:
                            write_json(ROOT / 'progress.json', {'phase': phase, 'run': i,
                                       'candidates': j + 1, 'replays': replays,
                                       'seconds': time.monotonic() - start})
                            print(f'{phase} run={i} images={j+1}/96 replays={replays}', flush=True)
                    names = sorted(gradients[0])
                    mats = []
                    for name in names:
                        x = torch.stack([v[name].flatten() for v in gradients] +
                                        [v.tensors[name].float().flatten() for v in obs]).to('cuda:0', torch.float64)
                        mats.append((x @ x.T).cpu())
                        del x
                    gram = torch.stack(mats)
                    if not torch.isfinite(gram).all():
                        raise ValueError('Nonfinite Gram matrix')
                    write_tensors(ROOT / 'grams' / f'{key}.safetensors', {'gram': gram})
                    write_json(ROOT / 'grams' / f'{key}.json', {'tensor_names': names, 'bank_rows': bank_rows,
                               'text_sha256': key, 'candidate_count': 96, 'observations': 20})
                    cache[key] = file_hash(ROOT / 'grams' / f'{key}.safetensors')
                    del gradients, mats, gram
                    gc.collect()
                references.append({'phase': phase, 'run': i, 'text_sha256': key})
                write_json(ROOT / 'references.json', references)
                if not (ROOT / 'first_checkpoint.json').exists():
                    write_json(ROOT / 'first_checkpoint.json', {'grams_committed': True, 'replays': replays,
                               'private_reference_read': False, 'peak_mib': torch.cuda.max_memory_allocated() / 2**20})
        if adapter.fingerprint() != fingerprint or any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError('Read-only replay changed model')
        verify(plan['frozen_hashes'])
        verify(read_json(source_manifest)['hashes'])
        write_json(ROOT / 'collect_complete.json', {'unique_texts': len(cache), 'replays': replays,
                   'hashes': {str(ROOT / 'grams' / f'{key}.safetensors'): sha for key, sha in cache.items()},
                   'references_sha256': file_hash(ROOT / 'references.json'),
                   'seconds': time.monotonic() - start, 'model_unchanged': True,
                   'private_reference_read': False, 'gpu_uuid': gpu,
                   'reused_replays': migration['preserved_replays'] if migration else 0,
                   'peak_mib': torch.cuda.max_memory_allocated() / 2**20})
    except BaseException:
        write_json(ROOT / 'collect_failed.json', {'traceback': traceback.format_exc()})
        raise


def score():
    if (ROOT / 'scores_started.json').exists():
        raise FileExistsError('No automatic rescore')
    write_json(ROOT / 'scores_started.json', {'pid': os.getpid()})
    start = time.monotonic()
    try:
        plan = read_json(ROOT / 'plan.json')
        completion = read_json(ROOT / 'collect_complete.json')
        verify(completion['hashes'])
        verify(plan['frozen_hashes'])
        refs = read_json(ROOT / 'references.json')
        dev = {name: [] for name in plan['weighted_grid'] + plan['suppression_grid'] + plan['fisher_grid']}
        reconstructed = {r['text_sha256'] for r in refs if r['phase'] == 'reconstructed'}
        for path in sorted((ROOT / 'grams').glob('*.safetensors')):
            key = path.stem
            meta = read_json(path.with_suffix('.json'))
            grams = read_tensors(path)['gram'].numpy()
            banks = {}
            for b in range(4):
                fit = list(range(b * 24, (b + 1) * 24))
                labels = [plan['classes'].index(meta['bank_rows'][j]['label']) for j in fit]
                heldout = [j for j in range(72) if j not in fit] if b < 3 else []
                query = heldout + list(range(96, 116))
                values, diag = score_geometry(grams, fit, query, labels)
                nh = len(heldout)
                banks[str(b)] = {'target_scores': {m: [row.tolist() if np.isfinite(row).all() else None
                                                       for row in v[nh:]] for m, v in values.items()},
                                 'diagnostics': diag}
                if b < 3 and key in reconstructed:
                    y = np.array([plan['classes'].index(meta['bank_rows'][j]['label']) for j in heldout])
                    for name in dev:
                        valid = np.isfinite(values[name][:nh]).all(1)
                        dev[name].append(float(np.mean((values[name][:nh].argmin(1) == y) & valid)))
            write_json(ROOT / 'scores' / f'{key}.json', banks)
            print(f'scored {key[:8]}', flush=True)
        summary = {m: float(np.mean(v)) for m, v in dev.items()}
        selected = {group: max(plan[f'{group}_grid'], key=lambda m: summary[m])
                    for group in ['weighted', 'suppression', 'fisher']}
        write_json(ROOT / 'selection.json', {'selected': selected, 'public_development_accuracy': summary,
                   'private_labels_read': False, 'source': plan['selection'],
                   'unique_reconstructed_texts': len(reconstructed)})
        hashes = {}
        for r in refs:
            banks = read_json(ROOT / 'scores' / f'{r["text_sha256"]}.json')
            for b in range(4):
                data = banks[str(b)]['target_scores']
                record = {**r, 'bank': b, 'class_scores': {m: v[r['run']] for m, v in data.items()},
                          'all_upload_scores': data, 'selected': selected}
                target = ROOT / 'predictions' / f'{r["phase"]}-{b}-{r["run"]:02d}.json'
                write_json(target, record)
                hashes[str(target)] = file_hash(target)
        write_json(ROOT / 'predictions_complete.json', {'hashes': hashes, 'predictions': len(hashes),
                   'private_labels_read': False, 'seconds': time.monotonic() - start,
                   'selection_sha256': file_hash(ROOT / 'selection.json')})
    except BaseException:
        write_json(ROOT / 'score_failed.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'score'])
    args = parser.parse_args()
    globals()[args.stage]()
