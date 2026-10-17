"""Evaluator-owned preparation of a declared public bank and private labels.

The GPU inference worker receives only the exported bank and public observation
locations. Original sample IDs, annotation joins and property truth stay here.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from core.data import read_manifest
from evaluation.lm_prior_discriminability import inputs_for, validate_result


CLASSES = ['CT', 'MRI', 'X-Ray']
COHORTS = [(32, 'development'), (8, 'development'), (32, 'validation')]
DATA = Path('data/slake/samples.jsonl')
MODALITY_PATTERN = re.compile(r'\bct\b|\bmri\b|x[ -]?ray|computed tomography|magnetic resonance', re.I)


def qa_signature(rows, answer_key):
    return tuple(sorted((r['question'].strip(), r[answer_key].strip()) for r in rows))


def annotation_join(rows, annotation_root):
    original = defaultdict(list)
    for split in ['train', 'validation', 'test']:
        for row in read_json(annotation_root/f'{split}.json'):
            if row['q_lang'] == 'en':
                original[row['img_name']].append(row)
    index = defaultdict(list)
    for name, group in original.items():
        index[qa_signature(group, 'answer')].append(name)
    local = defaultdict(list)
    for row in rows:
        local[row['image_id']].append(row)
    joined = {}
    for key, group in local.items():
        matches = index.get(qa_signature(group, 'target'), [])
        modalities = {r['modality'] for name in matches for r in original[name]}
        locations = {r['location'] for name in matches for r in original[name]}
        joined[key] = {'matches': matches,
                       'label': next(iter(modalities)) if len(modalities) == 1 else None,
                       'locations': sorted(locations), 'modality_values': sorted(modalities)}
    return local, joined


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode()+image.convert('RGB').tobytes()).hexdigest()


def prepare(root):
    if (root/'plan.json').exists():
        raise FileExistsError('Prepared experiment already exists')
    rows = read_manifest(DATA)
    local, joined = annotation_join(rows, root/'metadata_source')
    targets, cohorts, signatures = set(), {}, {}
    for rank, phase in COHORTS:
        source, model, prior = inputs_for(rank, phase)
        prior_plan = read_json(prior/'plan.json')
        if prior_plan['model_sha256'] != file_hash(model/'model.safetensors'):
            raise ValueError('Reconstruction model hash mismatch')
        if read_json(prior/'complete.json')['summary_sha256'] != file_hash(prior/'summary.json'):
            raise ValueError('Reconstruction completion hash mismatch')
        name = f'rank-{rank}-{phase}'
        cohort = {'rank': rank, 'phase': phase, 'model': str(model), 'runs': []}
        labels = []
        for i in range(10):
            src, attack = source/f'run-{i:05d}', prior/f'run-{i:05d}'
            truth = read_json(src/'captures/private/private/truth.json')['samples'][0]
            key = truth['image_id']
            if joined[key]['label'] not in CLASSES:
                raise ValueError(f'Missing or conflicting modality for {name} run {i}')
            targets.add(key)
            observation = src/'captures/private/public'
            obs = read_json(observation/'observation.json')
            tokens = {}
            for condition, weight in [('baseline', 0.0), ('lm_prior', 0.01)]:
                folder = attack/'attacks'/('weight-0p0' if weight == 0 else 'weight-0p01')
                validate_result(read_json(folder/'result.json'), obs, rank, weight)
                tokens[condition] = str(folder/'text_tokens.safetensors')
                for filename in ['result.json', 'text_tokens.safetensors']:
                    signatures[str(folder/filename)] = file_hash(folder/filename)
            for filename in ['observation.json', 'update.safetensors']:
                signatures[str(observation/filename)] = file_hash(observation/filename)
            cohort['runs'].append({'run': i, 'observation': str(observation), 'tokens': tokens})
            labels.append({'run': i, 'label': joined[key]['label'],
                           'explicit': bool(MODALITY_PATTERN.search(truth['question']+' '+truth['target'])),
                           'image_id': key, 'annotation_matches': joined[key]['matches'],
                           'oracle_tokens': str(src/'captures/known/private/text_tokens.safetensors'),
                           'reference_image': str(src/'captures/known/public/public_images.safetensors')})
        signatures[str(model/'model.json')] = file_hash(model/'model.json')
        signatures[str(model/'model.safetensors')] = file_hash(model/'model.safetensors')
        write_json(root/'private'/f'{name}.json', labels)
        cohorts[name] = cohort
    if len(targets) != 20:
        raise ValueError('Expected twenty distinct source image groups')
    target_hashes = {pixel_hash(Image.open(local[key][0]['image'])) for key in targets}
    eligible = {c: [] for c in CLASSES}
    for key, group in local.items():
        info = joined[key]
        # Use unambiguous original-image joins for the public bank.
        if (group[0]['split'] == 'tune' and key not in targets
                and len(info['matches']) == 1 and info['label'] in CLASSES):
            eligible[info['label']].append(key)
    rng = random.Random(20261016)
    bank, tensors, provenance = [], {}, []
    for label in CLASSES:
        choices = sorted(eligible[label])
        if len(choices) < 5:
            raise ValueError(f'Insufficient isolated public prototypes for {label}')
        for j, key in enumerate(rng.sample(choices, 5)):
            image = Image.open(local[key][0]['image']).convert('RGB')
            image_hash = pixel_hash(image)
            if image_hash in target_hashes:
                raise ValueError('Public bank overlaps a target by decoded pixels')
            identifier = f'candidate-{len(bank):02d}'
            tensors[identifier] = torch.from_numpy(np.array(image, copy=True))
            bank.append({'id': identifier, 'label': label, 'pixel_sha256': image_hash,
                         'locations': joined[key]['locations']})
            provenance.append({'id': identifier, 'image_id': key,
                               'annotation': joined[key], 'ordinal': j})
    if len({x['pixel_sha256'] for x in bank}) != 15:
        raise ValueError('Duplicate public candidates')
    write_tensors(root/'public/bank.safetensors', tensors)
    write_json(root/'public/bank.json', bank)
    write_json(root/'private/bank_provenance.json', provenance)
    write_json(root/'private/annotation_join.json', joined)
    counts = {}
    for name in cohorts:
        labels = read_json(root/'private'/f'{name}.json')
        counts[name] = {
            'classes': dict(Counter(r['label'] for r in labels)),
            'nonexplicit_classes': dict(Counter(r['label'] for r in labels if not r['explicit'])),
            'explicit_count': sum(r['explicit'] for r in labels),
            'ambiguous_image_join_consistent_modality': sum(len(r['annotation_matches']) > 1 for r in labels)}
    write_json(root/'data_audit.json', {
        'classes': CLASSES, 'cohorts': counts, 'distinct_target_images': len(targets),
        'bank_counts': dict(Counter(r['label'] for r in bank)),
        'eligible_bank_counts': {c: len(x) for c, x in eligible.items()},
        'bank_target_overlap': 0, 'bank_unique_decoded_pixels': 15,
        'label_method': 'Exact full English QA multiset join to pinned author annotations; accept only unanimous modality',
        'all_local_groups': len(local),
        'join_cardinality': dict(Counter(len(r['matches']) for r in joined.values())),
        'metadata_provenance_sha256': file_hash(root/'metadata_source/provenance.json'),
        'manifest_sha256': file_hash(DATA),
        'patient_overlap': 'not audited: no patient IDs',
        'validation_limitation': 'No MRI target; three-class balanced accuracy and MRI AUROC undefined',
        'nonexplicit_limitation': 'All nonexplicit target examples are CT; cannot assess balanced nonexplicit classification'})
    plan = {
        'schema_version': 1, 'classes': CLASSES, 'cohorts': cohorts, 'seed': 20261016,
        'bank_per_class': 5, 'top_m': 3, 'physical_gpu': 5,
        'text_conditions': ['unknown', 'baseline', 'lm_prior', 'oracle'],
        'scores': ['global', 'centered', 'equal_tensor', 'centered_equal_tensor'],
        'tie_priority': ['global', 'centered', 'equal_tensor', 'centered_equal_tensor'],
        'selection': 'rank-32 development lm_prior balanced accuracy; fixed tie priority',
        'unknown_question': 'What is shown in the image?', 'unknown_answer': 'An image.',
        'unknown_length_policy': 'Repeat/truncate template content IDs to public content lengths; public EOS/PAD',
        'text_only_prompt': 'USER: Infer the imaging modality from the question and answer. Choose CT, MRI, or X-Ray.\nQuestion: {question}\nAnswer: {answer}\nASSISTANT:',
        'text_only_scoring': 'Mean continuation token NLL for space plus class name; no EOS; no images',
        'model_only_scoring': 'Victim response mean NLL including EOS, excluding PAD; same class top-3 mean',
        'shuffled': 'Each of the other nine observed gradients within cohort; text and bank fixed; no self-pair',
        'zero_norm_policy': 'Require residual norm > 1e-8 times original norm, with 1e-30 absolute floor; fixed common valid tensor mask over entire bank and target; report omissions',
        'missing_class_policy': 'Keep three classes; report supported-class BA separately and three-class BA null',
        'validation_user_authorization': 'User approved existing-sample pilot despite absent validation MRI',
        'public_bank_policy': 'Explicit attacker auxiliary subset of tune; exclude every development and validation target; bank labels public',
        'reference_policy': 'All nonoracle predictions committed before reference reading; oracle isolated diagnostic',
        'source_hashes': signatures,
        'bank_hashes': {f: file_hash(root/'public'/f) for f in ['bank.json', 'bank.safetensors']},
        'uncommitted': True, 'not_a_formal_benchmark': True}
    write_json(root/'plan.json', plan)
    print(read_json(root/'data_audit.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    try:
        prepare(root)
    except Exception as error:
        write_json(root/'prepare_failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
