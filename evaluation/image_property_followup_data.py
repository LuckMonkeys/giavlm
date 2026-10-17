"""Freeze balanced, nonexplicit modality targets and public prototype banks."""
from collections import Counter, defaultdict
import itertools
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from core.data import read_manifest
from evaluation.image_property_data import CLASSES, DATA, annotation_join, pixel_hash
from evaluation.image_property_scoring import classification_stats

OLD = Path('outputs/diagnostics/image_property_gpu5_20261016')
ROOT = Path('outputs/diagnostics/image_property_followup_gpu5_20261016')
MODEL = Path('outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-32/round-0010')
CONFIG = Path('outputs/diagnostics/dager_rank_gpu5_20261010_v2/rank-32/configs/random_residual.json')
NONEXPLICIT = re.compile(
    r'\b(?:ct|mri\w*|xray|radiograph\w*|scan\w*|modality|modalities|t1|t2|weighted)\b'
    r'|x[ -]ray|computed tomography|magnetic resonance|imaging technique|type of image|kind of image', re.I)
TARGET_QUOTAS = {'CT': {'abdomen': 4, 'chest': 4, 'head': 2},
                 'MRI': {'abdomen': 4, 'head': 6}, 'X-Ray': {'chest': 10}}
BANK_QUOTAS = {'CT': {'abdomen': 3, 'chest': 1, 'head': 1},
               'MRI': {'abdomen': 3, 'head': 2}, 'X-Ray': {'chest': 5}}


def region(locations):
    if len(locations) != 1:
        return None
    return {'Abdomen': 'abdomen', 'Brain': 'head', 'Brain_Tissue': 'head',
            'Brain_Face': 'head', 'Lung': 'chest', 'Chest_lung': 'chest',
            'Chest_heart': 'chest'}.get(locations[0])


def literal_property(question, answer):
    answer = answer.strip().lower().rstrip('.!?')
    direct = {'ct': 'CT', 'computed tomography': 'CT', 'mri': 'MRI',
              'magnetic resonance imaging': 'MRI', 'x-ray': 'X-Ray', 'x ray': 'X-Ray', 'xray': 'X-Ray'}
    if answer in direct:
        return direct[answer]
    patterns = [r'\bct\b|computed tomography', r'\bmri\b|magnetic resonance', r'x[ -]?ray']
    mentioned = [c for c, p in zip(CLASSES, patterns) if re.search(p, question, re.I)]
    if (answer == 'yes' and len(mentioned) == 1
            and re.match(r'^is (?:this|it|the image|the picture)\b', question.strip(), re.I)):
        return mentioned[0]
    return None


def calibrate_text():
    labels = [r['label'] for r in read_json(OLD/'private/rank-32-development.json')]
    scores = np.array([read_json(OLD/'rank-32-development/predictions'/f'{i:05d}-lm_prior.json')['text_only']
                       for i in range(10)])
    trials = []
    for a, b in itertools.product([-2., -1., 0., 1., 2.], repeat=2):
        offsets = [a, b, 0.]
        ba = classification_stats(labels, (scores+offsets).tolist(), CLASSES)['balanced_accuracy']
        trials.append((ba, -sum(x*x for x in offsets), -a, -b, offsets))
    best = max(trials)
    return {'offsets': best[-1], 'development_balanced_accuracy': best[0],
            'grid': [-2., -1., 0., 1., 2.], 'anchor': 'X-Ray offset 0',
            'selection': 'Old rank-32 development LM-prior text only; maximize BA, then min squared offsets, then smaller CT and MRI offsets',
            'source_report_sha256': file_hash(OLD/'report.json')}


def prepare():
    if (ROOT/'plan.json').exists():
        raise FileExistsError('Frozen followup already prepared')
    local, joined = annotation_join(read_manifest(DATA), OLD/'metadata_source')
    excluded = {r['image_id'] for p in (OLD/'private').glob('rank-*.json') for r in read_json(p)}
    eligible, bank_pool = defaultdict(list), defaultdict(list)
    for key, group in local.items():
        info = joined[key]
        reg = region(info['locations'])
        if key in excluded or len(info['matches']) != 1 or info['label'] not in CLASSES or reg is None:
            continue
        pair = info['label'], reg
        if group[0]['split'] == 'eval':
            candidates = [r for r in group if not NONEXPLICIT.search(r['question']+' '+r['target'])]
            if candidates:
                eligible[pair].append((key, candidates))
        elif group[0]['split'] == 'tune':
            bank_pool[pair].append(key)
    rng = random.Random(2026101601)
    targets, seen_qa = [], set()
    for label, quotas in TARGET_QUOTAS.items():
        for reg, n in quotas.items():
            for key, rows in rng.sample(sorted(eligible[label, reg]), n):
                choices = sorted(rows, key=lambda r: r['sample_id'])
                novel = [r for r in choices if (r['question'], r['target']) not in seen_qa]
                row = rng.choice(novel or choices)
                seen_qa.add((row['question'], row['target']))
                targets.append({'run': len(targets), 'label': label, 'region': reg, 'row': row})
    write_json(ROOT/'private/targets.json', targets)
    target_hashes = {pixel_hash(Image.open(r['row']['image'])) for r in targets}
    if len(target_hashes) != 30:
        raise ValueError('Thirty distinct targets required')
    banks, bank_hashes = [], {}
    for b, seed in enumerate([2026101601, 2026101602, 2026101603]):
        rng = random.Random(seed)
        bank, tensors, provenance = [], {}, []
        for label, quotas in BANK_QUOTAS.items():
            for reg, n in quotas.items():
                for key in rng.sample(sorted(bank_pool[label, reg]), n):
                    image = Image.open(local[key][0]['image']).convert('RGB')
                    pixel = pixel_hash(image)
                    if pixel in target_hashes:
                        raise ValueError('Public bank contains target pixels')
                    identifier = f'candidate-{len(bank):02d}'
                    bank.append({'id': identifier, 'label': label, 'region': reg, 'pixel_sha256': pixel})
                    tensors[identifier] = torch.from_numpy(np.array(image, copy=True))
                    provenance.append({'id': identifier, 'image_id': key, 'annotation': joined[key]})
        if len({r['pixel_sha256'] for r in bank}) != 15:
            raise ValueError('Duplicate bank candidates')
        directory = ROOT/'public'/f'bank-{b}'
        write_json(directory/'bank.json', bank)
        write_tensors(directory/'bank.safetensors', tensors)
        write_json(ROOT/'private'/f'bank-{b}-provenance.json', provenance)
        banks.append({'bank': b, 'seed': seed, 'path': str(directory)})
        for name in ['bank.json', 'bank.safetensors']:
            bank_hashes[str(directory/name)] = file_hash(directory/name)
    def counts(pool):
        return {f'{c}:{r}': len(v) for (c, r), v in sorted(pool.items())}
    write_json(ROOT/'data_audit.json', {
        'targets': 30, 'target_classes': dict(Counter(r['label'] for r in targets)),
        'target_regions': dict(Counter(f"{r['label']}:{r['region']}" for r in targets)),
        'eligible_target_counts': counts(eligible), 'eligible_bank_counts': counts(bank_pool),
        'excluded_old_target_images': len(excluded), 'old_target_overlap': 0,
        'bank_target_pixel_overlap': 0, 'target_unique_pixels': 30, 'unique_raw_qa': len(seen_qa),
        'explicit_modality_mentions': 0, 'target_split': 'eval; across original clients, single-sample observations',
        'bank_split': 'declared tune auxiliary subset', 'original_annotation_join': 'unique full QA multiset',
        'patient_overlap': 'unknown; patient IDs unavailable',
        'confound': 'X-Ray is chest only; no three-way location match. Prespecified abdomen CT/MRI subset has 4 targets/class and 3 bank images/class.'})
    prior = read_json(OLD/'plan.json')
    hashes = {str(p): file_hash(p) for p in [MODEL/'model.json', MODEL/'model.safetensors', CONFIG, DATA]}
    write_json(ROOT/'plan.json', {
        'schema_version': 1, 'classes': CLASSES, 'model': str(MODEL), 'config': str(CONFIG),
        'physical_gpu': 5, 'rank': 32, 'round': 10, 'targets': 30, 'seed': 2026101601,
        'target_quotas': TARGET_QUOTAS, 'bank_quotas': BANK_QUOTAS, 'banks': banks,
        'bank_hashes': bank_hashes, 'source_hashes': hashes,
        'target_manifest_sha256': file_hash(ROOT/'private/targets.json'),
        'top_m': 3, 'primary_score': 'centered', 'ablation_score': 'global',
        'primary_bank': 0, 'lm_prior_weight': 0.01,
        'text_only_prompt': prior['text_only_prompt'], 'text_calibration': calibrate_text(),
        'literal_rule': 'Frozen from previous pilot; exact modality answer or Yes to explicit modality confirmation; otherwise abstain',
        'nonexplicit_regex': NONEXPLICIT.pattern,
        'stages': ['capture', 'oracle_bank_0', 'oracle_bank_1', 'oracle_bank_2', 'gate', 'reconstruct_if_gate_passes', 'reconstructed_all_banks_if_gate_passes'],
        'gate': 'Primary-bank oracle centered accuracy strictly exceeds both model-only accuracy and mean wrong-gradient accuracy. Exploratory point-estimate gate, not significance test; no individual samples filtered.',
        'subset': 'Prespecified abdomen targets; restrict prototypes to abdomen CT/MRI; equal 3/class, top_m=3',
        'reference_policy': 'Oracle uses declared text_known Observation only; private reconstruction uses separate private Observation. Labels enter only after each phase predictions commit.',
        'uncommitted': True})
    print(read_json(ROOT/'data_audit.json'))
    print('Text-only offsets:', read_json(ROOT/'plan.json')['text_calibration']['offsets'])


if __name__ == '__main__':
    prepare()
