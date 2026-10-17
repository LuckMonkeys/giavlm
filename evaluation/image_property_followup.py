"""Frozen balanced modality followup, with an oracle gate before private text search."""
from dataclasses import replace
import gc
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import load_config
from core.data import load_batch
from core.fl import capture, load_observation, restore_model, save_observation, simulate_update
from core.types import Batch
from evaluation.dager_rank_study import GPU5, attack_public
from evaluation.image_property_followup_data import ROOT, literal_property, NONEXPLICIT
from evaluation.image_property_scoring import METHODS, aggregate_classes, classification_stats, paired_bootstrap, score_bank
from evaluation.image_property_study import text_key, text_only, validate_text
from evaluation.token_image_ordering import GradientScorer


def capture_targets(adapter, config, plan):
    """Victim-only stage; exports distinct text-known and private observations."""
    path = ROOT/'private/targets.json'
    if file_hash(path) != plan['target_manifest_sha256']:
        raise ValueError('Target selection changed')
    targets = read_json(path)
    adapter.set_training_spec(replace(config.training, knowledge='text_known', server_round=10))
    truncations = 0
    for item in targets:
        i, row = item['run'], item['row']
        batch = load_batch([row], adapter)
        question, answer = adapter.decode(batch.questions)[0], adapter.decode(batch.targets)[0]
        if NONEXPLICIT.search(question+' '+answer):
            raise ValueError('Captured text explicitly identifies modality')
        for value, slots in [(row['question'], adapter.spec.question_length), (row['target'], adapter.spec.target_length)]:
            truncations += len(adapter.tokenizer.encode(value, add_special_tokens=False)) >= slots
        known = capture(adapter, batch, [question], [answer])
        private = replace(known, training=replace(known.training, knowledge='private'),
                          public_questions=[], public_targets=[], public_question_ids=[], public_target_ids=[])
        private.validate()
        run = ROOT/'captures'/f'run-{i:05d}'
        save_observation(run/'oracle', known)
        save_observation(run/'private', private)
        write_tensors(ROOT/'private'/f'reference-{i:05d}.safetensors',
                      {'images': batch.images, 'questions': batch.questions, 'targets': batch.targets})
        if read_json(run/'oracle/observation.json')['update_sha256'] != read_json(run/'private/observation.json')['update_sha256']:
            raise ValueError('Paired captures differ')
        error = GradientScorer(known.tensors, adapter.device).score(simulate_update(adapter, batch, known.training))['all']['relative_l2']
        if error > 1e-12:
            raise ValueError('Exact capture replay failed')
        write_json(ROOT/'progress.json', {'stage': 'capture', 'run': i, 'exact_replay_relative_l2': error})
        print(f'capture {i+1}/30 committed; exact replay={error}', flush=True)
    write_json(ROOT/'capture_complete.json', {
        'observations': 30, 'exact_replays': 30, 'max_replay_relative_l2': 0,
        'truncated_fields': truncations,
        'hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT/'captures').rglob('*')) if p.is_file()}})


def oracle_text(obs, device):
    if obs.training.knowledge != 'text_known':
        raise ValueError('Oracle diagnostic needs declared public text')
    return {'questions': torch.tensor(obs.public_question_ids, device=device),
            'targets': torch.tensor(obs.public_target_ids, device=device)}


def make_scores(scores, labels, classes):
    for row in scores:
        row['class_scores'] = {m: aggregate_classes(row['scores'][m], labels, classes, 3) for m in METHODS}
    return scores


def predict_phase(adapter, plan, phase):
    known = phase == 'oracle'
    observations = [load_observation(ROOT/'captures'/f'run-{i:05d}'/('oracle' if known else 'private'), adapter.device)
                    for i in range(30)]
    adapter.set_training_spec(observations[0].training)
    targets = [o.tensors for o in observations]
    text_cache = {}
    totals = {'gradient_evaluations': 0, 'model_only_forwards': 0, 'text_only_forwards': 0}
    for bank_info in plan['banks']:
        b = bank_info['bank']
        folder = ROOT/phase/f'bank-{b}'
        folder.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        bank = read_json(Path(bank_info['path'])/'bank.json')
        raw = read_tensors(Path(bank_info['path'])/'bank.safetensors')
        images = [adapter.prepare_image(Image.fromarray(raw[r['id']].numpy())).unsqueeze(0).to(adapter.device, adapter.dtype) for r in bank]
        labels = [r['label'] for r in bank]
        abdomen = [i for i, r in enumerate(bank) if r['region'] == 'abdomen']
        abdomen_labels = [labels[i] for i in abdomen]
        cache = {}
        for i, obs in enumerate(observations):
            text = (oracle_text(obs, adapter.device) if known else
                    read_tensors(ROOT/'reconstruction'/f'run-{i:05d}'/'text_tokens.safetensors', str(adapter.device)))
            validate_text(adapter, text, obs)
            key = text_key(text)
            if key not in cache:
                gradients, nlls = [], []
                for image in images:
                    batch = Batch(image, text['questions'], text['targets'])
                    gradients.append({k: v.detach() for k, v in simulate_update(adapter, batch, obs.training).items()})
                    with torch.no_grad():
                        nlls.append(float(adapter(batch.images, batch.questions, batch.targets)))
                    totals['gradient_evaluations'] += 1
                    totals['model_only_forwards'] += 1
                full = make_scores(score_bank(gradients, targets, str(adapter.device)), labels, plan['classes'])
                matched = make_scores(score_bank([gradients[k] for k in abdomen], targets, str(adapter.device)), abdomen_labels, ['CT', 'MRI'])
                old = GradientScorer(targets[i], adapter.device).score(gradients[0])['all']['cosine']
                if abs(old-full[i]['scores']['global'][0]) > 2e-5:
                    raise ValueError('Scorer crosscheck failed')
                del gradients
                if key not in text_cache:
                    raw_text = text_only(adapter, text, plan)
                    literal = literal_property(adapter.decode(text['questions'])[0], adapter.decode(text['targets'])[0])
                    text_cache[key] = {'text_only': raw_text,
                                       'text_calibrated': (np.array(raw_text)+plan['text_calibration']['offsets']).tolist(),
                                       'literal': literal}
                    totals['text_only_forwards'] += 3
                scores = {'text_sha256': key, 'full': full, 'abdomen': matched,
                          'candidate_nll': nlls,
                          'model_only': aggregate_classes(nlls, labels, plan['classes'], 3),
                          'abdomen_model_only': aggregate_classes([nlls[k] for k in abdomen], abdomen_labels, ['CT', 'MRI'], 3),
                          **text_cache[key]}
                file = folder/'scores'/f'{key}.json'
                write_json(file, scores)
                cache[key] = (file, scores)
            file, scores = cache[key]
            prediction = {'run': i, 'bank': b, 'phase': phase, 'text_sha256': key,
                          'scores_file': str(file.relative_to(ROOT)), 'scores_sha256': file_hash(file),
                          'class_scores': scores['full'][i]['class_scores'],
                          'abdomen_class_scores': scores['abdomen'][i]['class_scores'],
                          **{k: scores[k] for k in ['model_only', 'abdomen_model_only', 'text_only', 'text_calibrated', 'literal']},
                          'wrong': {str(j): scores['full'][j]['class_scores'] for j in range(30) if j != i},
                          'abdomen_wrong': {str(j): scores['abdomen'][j]['class_scores'] for j in range(30) if j != i}}
            write_json(folder/'predictions'/f'{i:05d}.json', prediction)
            write_json(ROOT/'progress.json', {'stage': phase, 'bank': b, 'run': i, **totals})
            if i == 0:
                write_json(folder/'first_checkpoint.json', {'predictions': 1, 'scorer_crosscheck': 'passed',
                           'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20})
            print(f'{phase} bank {b} run {i+1}/30 committed; {time.monotonic()-started:.1f}s', flush=True)
        write_json(folder/'complete.json', {'seconds': time.monotonic()-started,
                   'unique_texts': len(cache), 'reference_labels_read': False,
                   'hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in folder.rglob('*.json')}})
        del images, raw, cache
        gc.collect()
    write_json(ROOT/phase/'complete.json', {**totals, 'banks': 3, 'predictions': 90,
               'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
               'reference_labels_read': False})


def evaluate_phase(plan, phase):
    read_json(ROOT/phase/'complete.json')
    truth = read_json(ROOT/'private/targets.json')
    rows, paired = [], []
    for b in range(3):
        folder = ROOT/phase/f'bank-{b}'
        for path, expected in read_json(folder/'complete.json')['hashes'].items():
            if file_hash(ROOT/path) != expected:
                raise ValueError('Committed scores changed')
        predictions = [read_json(folder/'predictions'/f'{i:05d}.json') for i in range(30)]
        for subset, ids, classes in [('all', list(range(30)), plan['classes']),
                                    ('abdomen', [r['run'] for r in truth if r['region'] == 'abdomen'], ['CT', 'MRI'])]:
            labels = [truth[i]['label'] for i in ids]
            prefix = '' if subset == 'all' else 'abdomen_'
            values = {m: [predictions[i][prefix+'class_scores'][m] for i in ids] for m in METHODS}
            values['model_only'] = [predictions[i][prefix+'model_only'] for i in ids]
            for method in ['text_only', 'text_calibrated']:
                values[method] = [[predictions[i][method][plan['classes'].index(c)] for c in classes] for i in ids]
            values['literal'] = [[0. if c == predictions[i]['literal'] else 1. for c in classes]
                                 if predictions[i]['literal'] in classes else None for i in ids]
            for method, scores in values.items():
                stats = classification_stats(labels, scores, classes)
                stats.update(paired_bootstrap(labels, scores, classes))
                rows.append({'bank': b, 'phase': phase, 'subset': subset, 'method': method, **stats})
            for other in ['model_only', 'text_only', 'text_calibrated']:
                a, z = values['centered'], values[other]
                gain = classification_stats(labels, a, classes)['accuracy']-classification_stats(labels, z, classes)['accuracy']
                paired.append({'bank': b, 'phase': phase, 'subset': subset, 'compared_to': other,
                               'accuracy_gain': gain, **paired_bootstrap(labels, a, classes, z)})
            wrong = []
            for i, label in zip(ids, labels):
                alternatives = [predictions[i][prefix+'wrong'][str(j)]['centered'] for j in ids if j != i]
                wrong.append(float(np.mean([s is not None and classes[int(np.argmin(s))] == label for s in alternatives])))
            rows.append({'bank': b, 'phase': phase, 'subset': subset, 'method': 'wrong_centered',
                         'n': len(ids), 'accuracy': float(np.mean(wrong)), 'per_image_accuracy': wrong,
                         'note': 'Mismatches averaged within image; same-region subset uses only other abdomen uploads'})
    report = {'phase': phase, 'rows': rows, 'paired': paired, 'plan_sha256': file_hash(ROOT/'plan.json')}
    write_json(ROOT/phase/'report.json', report)
    return report


def gate(report):
    primary = {r['method']: r['accuracy'] for r in report['rows'] if r['bank'] == 0 and r['subset'] == 'all'}
    passed = primary['centered'] > max(primary['model_only'], primary['wrong_centered'])
    result = {'passed': passed, 'primary_bank_accuracies': primary,
              'oracle_report_sha256': file_hash(ROOT/'oracle/report.json'),
              'rule': read_json(ROOT/'plan.json')['gate']}
    write_json(ROOT/'gate.json', result)
    print('Oracle gate:', result, flush=True)
    return passed


def reconstruct(adapter, config, plan, fingerprint):
    config = replace(config, training=replace(config.training, server_round=10),
                     attack=replace(config.attack, dager=replace(config.attack.dager, lm_prior_weight=0.01)))
    write_json(ROOT/'reconstruction/config.json', config)
    for i in range(30):
        write_json(ROOT/'progress.json', {'stage': 'reconstruct', 'run': i})
        status, costs = attack_public(adapter, config, ROOT/'captures'/f'run-{i:05d}'/'private',
                                      ROOT/'reconstruction'/f'run-{i:05d}', fingerprint)
        if status != 'completed':
            raise ValueError(f'Reconstruction incomplete: {status}')
        print(f'reconstruct {i+1}/30: {status}, seconds={costs["seconds"]:.1f}', flush=True)
        gc.collect()
    write_json(ROOT/'reconstruction/complete.json', {'reconstructions': 30,
               'result_hashes': {str(p.relative_to(ROOT)): file_hash(p) for p in (ROOT/'reconstruction').glob('run-*/*') if p.is_file()}})


def text_audit(adapter):
    rows = []
    for i in range(30):
        predicted = read_tensors(ROOT/'reconstruction'/f'run-{i:05d}'/'text_tokens.safetensors')
        truth = read_tensors(ROOT/'private'/f'reference-{i:05d}.safetensors')
        exact = {k: adapter.decode(predicted[k])[0] == adapter.decode(truth[k])[0] for k in ['questions', 'targets']}
        rows.append({'run': i, 'question_em': exact['questions'], 'answer_em': exact['targets'],
                     'joint_em': all(exact.values())})
    write_json(ROOT/'text_audit.json', {'rows': rows, 'aggregate': {
        k: float(np.mean([r[k] for r in rows])) for k in ['question_em', 'answer_em', 'joint_em']}})


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError('Only authorized GPU 5 may be exposed')
    if (ROOT/'started.json').exists():
        raise FileExistsError('No automatic retry or partial overwrite')
    torch.set_num_threads(1)
    started = time.monotonic()
    source = source_fingerprint()
    write_json(ROOT/'started.json', {'source_sha256': source, 'plan_sha256': file_hash(ROOT/'plan.json')})
    try:
        plan = read_json(ROOT/'plan.json')
        for path, expected in {**plan['source_hashes'], **plan['bank_hashes']}.items():
            if file_hash(path) != expected:
                raise ValueError('Frozen input changed')
        adapter = restore_model(plan['model'], 'cuda:0')
        fingerprint = adapter.fingerprint()
        versions = {n: p._version for n, p in adapter.named_parameters()}
        config = load_config(plan['config'])
        torch.cuda.reset_peak_memory_stats()
        capture_targets(adapter, config, plan)
        gc.collect()
        predict_phase(adapter, plan, 'oracle')
        report = evaluate_phase(plan, 'oracle')
        proceed = gate(report)
        if proceed:
            reconstruct(adapter, config, plan, fingerprint)
            predict_phase(adapter, plan, 'reconstructed')
            evaluate_phase(plan, 'reconstructed')
            text_audit(adapter)
        if source_fingerprint() != source or any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError('Source or model changed')
        if adapter.fingerprint() != fingerprint:
            raise ValueError('Model fingerprint changed')
        write_json(ROOT/'complete.json', {'seconds': time.monotonic()-started,
                   'oracle_report_sha256': file_hash(ROOT/'oracle/report.json'),
                   'private_cascade_run': proceed, 'model_unchanged': True,
                   'source_sha256': source,
                   'reconstructed_report_sha256': file_hash(ROOT/'reconstructed/report.json') if proceed else None})
        print('Completed balanced modality followup', flush=True)
    except Exception as error:
        write_json(ROOT/'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
