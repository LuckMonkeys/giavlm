"""Frozen public-prototype modality pilot; no training or text reconstruction.

Public-input inference is completed before the separate oracle/replay evaluator
reads reference files. All failure states are saved; there is no automatic retry.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json)
from core.fl import load_observation, restore_model, simulate_update
from core.types import Batch
from evaluation.dager_rank_study import GPU5
from evaluation.image_property_scoring import (METHODS, aggregate_classes,
                                               classification_stats, paired_bootstrap,
                                               score_bank)
from evaluation.token_image_ordering import GradientScorer


FIELDS = ('questions', 'targets')


def unknown_text(adapter, obs, plan):
    text = {}
    for field, length, slots, template in [
        ('questions', obs.public_question_lengths[0], adapter.spec.question_length,
         plan['unknown_question']),
        ('targets', obs.public_target_lengths[0], adapter.spec.target_length,
         plan['unknown_answer'])]:
        ids = adapter.tokenizer.encode(template, add_special_tokens=False)
        if not ids or not 0 <= length < slots:
            raise ValueError('Invalid public template or content length')
        values = (ids*((length+len(ids)-1)//len(ids)))[:length]
        values += [adapter.eos]+[adapter.pad]*(slots-length-1)
        text[field] = torch.tensor([values], dtype=torch.long, device=adapter.device)
    return text


def validate_text(adapter, text, obs):
    expected = [obs.public_question_lengths, obs.public_target_lengths]
    for field, lengths in zip(FIELDS, expected, strict=True):
        if adapter.content_lengths(text[field]) != lengths:
            raise ValueError('Candidate violates public token lengths')


def text_key(text):
    payload = repr([(f, text[f].cpu().tolist()) for f in FIELDS]).encode()
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def text_only(adapter, text, plan):
    prompt = plan['text_only_prompt'].format(
        question=adapter.decode(text['questions'])[0], answer=adapter.decode(text['targets'])[0])
    prefix = adapter.tokenizer.encode(prompt, add_special_tokens=False)
    losses = []
    for label in plan['classes']:
        suffix = adapter.tokenizer.encode(' '+label, add_special_tokens=False)
        ids = torch.tensor([prefix+suffix], device=adapter.device)
        logits = adapter.backend.language_model(input_ids=ids, use_cache=False,
                                                 return_dict=True).logits
        logp = logits[0, len(prefix)-1:-1].float().log_softmax(-1)
        tokens = ids[0, len(prefix):]
        losses.append(float(-logp.gather(1, tokens[:, None]).mean()))
    return losses


def validate_sources(root, plan, cohort):
    for filename, expected in plan['bank_hashes'].items():
        if file_hash(root/'public'/filename) != expected:
            raise ValueError('Frozen bank changed')
    paths = [str(Path(cohort['model'])/f) for f in ['model.json', 'model.safetensors']]
    for run in cohort['runs']:
        paths += [str(Path(run['observation'])/f) for f in ['observation.json', 'update.safetensors']]
        for path in run['tokens'].values():
            paths += [path, str(Path(path).parent/'result.json')]
    for path in paths:
        if file_hash(path) != plan['source_hashes'][path]:
            raise ValueError('Frozen model, observation or reconstruction changed')


class PublicWorker:
    def __init__(self, adapter, observations, bank, images, root, plan):
        self.adapter, self.observations = adapter, observations
        self.bank, self.images, self.root, self.plan = bank, images, root, plan
        self.cache = {}
        self.evaluations = 0
        self.model_forwards = 0
        self.text_only_forwards = 0
        self.scoring_seconds = 0.
        self.targets = [obs.tensors for obs in observations]
        self.started = time.monotonic()

    def score(self, text):
        key = text_key(text)
        if key in self.cache:
            return self.cache[key]
        gradients, nlls = [], []
        for image in self.images:
            batch = Batch(image, text['questions'], text['targets'])
            # Only first-order parameter gradients. Images never require grad.
            predicted = simulate_update(self.adapter, batch, self.observations[0].training)
            gradients.append({k: v.detach() for k, v in predicted.items()})
            self.evaluations += 1
            with torch.no_grad():
                nlls.append(float(self.adapter(batch.images, batch.questions, batch.targets)))
            self.model_forwards += 1
        start = time.monotonic()
        scores = score_bank(gradients, self.targets, device=str(self.adapter.device))
        self.scoring_seconds += time.monotonic()-start
        # Independent existing scorer sanity check, without any reference data.
        existing = GradientScorer(self.targets[0], self.adapter.device).score(gradients[0])['all']['cosine']
        if abs(existing-scores[0]['scores']['global'][0]) > 2e-5:
            raise ValueError('Global cosine differs from existing scorer')
        del gradients
        labels = [r['label'] for r in self.bank]
        for row in scores:
            row['class_scores'] = {m: aggregate_classes(row['scores'][m], labels,
                                   self.plan['classes'], self.plan['top_m']) for m in METHODS}
        data = {'text_sha256': key, 'targets': scores, 'candidate_nll': nlls,
                'model_only': aggregate_classes(nlls, labels, self.plan['classes'], self.plan['top_m']),
                'text_only': text_only(self.adapter, text, self.plan),
                'gradient_evaluations': len(self.bank)}
        self.text_only_forwards += len(self.plan['classes'])
        filename = f'scores/{key}.json'
        write_json(self.root/filename, data)
        self.cache[key] = (filename, data)
        return filename, data

    def predict(self, run, condition, text):
        validate_text(self.adapter, text, self.observations[run])
        filename, data = self.score(text)
        result = {'run': run, 'condition': condition, 'scores_file': filename,
                  'scores_sha256': file_hash(self.root/filename),
                  'text_sha256': data['text_sha256'],
                  'class_scores': data['targets'][run]['class_scores'],
                  'model_only': data['model_only'], 'text_only': data['text_only'],
                  'shuffled_class_scores': [r['class_scores'] for i, r in enumerate(data['targets']) if i != run],
                  'shuffled_target_runs': [i for i in range(len(self.targets)) if i != run]}
        write_json(self.root/'predictions'/f'{run:05d}-{condition}.json', result)
        print(f'{self.root.name} run {run} {condition}: committed; '
              f'{self.evaluations} replays, {time.monotonic()-self.started:.1f}s', flush=True)
        return result


def run_cohort(root, name):
    destination = root/name
    if destination.exists():
        raise FileExistsError('Use a fresh cohort directory; no automatic retries')
    plan = read_json(root/'plan.json')
    cohort = plan['cohorts'][name]
    if cohort['phase'] == 'validation' and not (root/'selection.json').exists():
        raise ValueError('Freeze development selection before validation')
    started = time.monotonic()
    fingerprint = source_fingerprint()
    write_json(destination/'started.json', {'source_sha256': fingerprint,
               'plan_sha256': file_hash(root/'plan.json'),
               'selection_sha256': file_hash(root/'selection.json') if cohort['phase'] == 'validation' else None})
    try:
        validate_sources(root, plan, cohort)
        adapter = restore_model(cohort['model'], 'cuda:0')
        model_fingerprint = adapter.fingerprint()
        observations = [load_observation(r['observation'], 'cuda:0') for r in cohort['runs']]
        for obs in observations:
            if (obs.model_fingerprint != model_fingerprint or obs.sample_count != 1
                    or obs.training.algorithm != 'fedsgd' or obs.training.knowledge != 'private'
                    or not obs.training.token_lengths_known
                    or asdict(obs.training) != asdict(observations[0].training)):
                raise ValueError('Unsupported or incompatible observation')
        adapter.set_training_spec(observations[0].training)
        versions = {n: p._version for n, p in adapter.named_parameters()}
        bank = read_json(root/'public/bank.json')
        raw = read_tensors(root/'public/bank.safetensors')
        images = [adapter.prepare_image(Image.fromarray(raw[r['id']].numpy())).unsqueeze(0)
                  .to(adapter.device, adapter.dtype) for r in bank]
        torch.cuda.reset_peak_memory_stats()
        worker = PublicWorker(adapter, observations, bank, images, destination, plan)
        for row in cohort['runs']:
            i = row['run']
            conditions = {'unknown': unknown_text(adapter, observations[i], plan),
                          **{k: read_tensors(v, 'cuda:0') for k, v in row['tokens'].items()}}
            for condition in ['unknown', 'baseline', 'lm_prior']:
                worker.predict(i, condition, conditions[condition])
                write_json(root/'progress.json', {'cohort': name, 'run': i, 'condition': condition,
                           'replays': worker.evaluations})
            if i == 0:
                write_json(destination/'first_run_checkpoint.json', {
                    'predictions': 3, 'replays': worker.evaluations,
                    'seconds_including_load': time.monotonic()-started,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                    'cosine_crosscheck': 'passed', 'private_reference_read': False})
        public_files = sorted((destination/'predictions').glob('*.json'))
        if len(public_files) != 30:
            raise ValueError('Every nonoracle output must commit before references are read')
        write_json(destination/'public_complete.json', {
            'predictions': {p.name: file_hash(p) for p in public_files},
            'private_reference_read': False})

        # Evaluator-only boundary: truth and oracle text enter only now.
        references = read_json(root/'private'/f'{name}.json')
        replay_errors = []
        for row in references:
            i = row['run']
            text = read_tensors(row['oracle_tokens'], 'cuda:0')
            truth = read_tensors(row['reference_image'], 'cuda:0')['images'].to(adapter.dtype)
            observed = observations[i]
            replay = simulate_update(adapter, Batch(truth, text['questions'], text['targets']), observed.training)
            error = GradientScorer(observed.tensors, 'cuda:0').score(replay)['all']['relative_l2']
            replay_errors.append(error)
            del replay, truth
            if error > 1e-12:
                raise ValueError('Exact victim replay failed')
            worker.predict(i, 'oracle', text)
        if any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError('Read-only inference mutated model parameters')
        if adapter.fingerprint() != model_fingerprint or source_fingerprint() != fingerprint:
            raise ValueError('Model or source changed during experiment')
        prediction_files = sorted((destination/'predictions').glob('*.json'))
        score_files = sorted((destination/'scores').glob('*.json'))
        write_json(destination/'complete.json', {
            'prediction_hashes': {str(p.relative_to(destination)): file_hash(p) for p in prediction_files},
            'score_hashes': {str(p.relative_to(destination)): file_hash(p) for p in score_files},
            'seconds': time.monotonic()-started, 'scoring_seconds': worker.scoring_seconds,
            'gradient_evaluations': worker.evaluations, 'exact_replays': len(replay_errors),
            'max_replay_relative_l2': max(replay_errors), 'model_only_forwards': worker.model_forwards,
            'text_only_forwards': worker.text_only_forwards, 'unique_texts': len(worker.cache),
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
            'model_unchanged': True, 'source_sha256': fingerprint})
    except Exception as error:
        write_json(destination/'failed.json', {'type': type(error).__name__, 'reason': str(error)})
        raise


def read_predictions(root, name, condition):
    folder = root/name
    completion = read_json(folder/'complete.json')
    for relative, expected in {**completion['prediction_hashes'], **completion['score_hashes']}.items():
        if file_hash(folder/relative) != expected:
            raise ValueError('Committed scores changed')
    return [read_json(folder/'predictions'/f'{i:05d}-{condition}.json') for i in range(10)]


def select(root):
    if (root/'selection.json').exists():
        raise FileExistsError('Development selection already frozen')
    plan = read_json(root/'plan.json')
    cohort = 'rank-32-development'
    predictions = read_predictions(root, cohort, 'lm_prior')
    labels = [r['label'] for r in read_json(root/'private'/f'{cohort}.json')]
    scores = {m: classification_stats(labels, [p['class_scores'][m] for p in predictions],
                                     plan['classes'])['balanced_accuracy'] for m in METHODS}
    if any(v is None for v in scores.values()):
        raise ValueError('Development selection requires all three classes')
    selected = max(plan['tie_priority'], key=lambda m: scores[m])
    write_json(root/'selection.json', {'selected': selected, 'development_balanced_accuracy': scores,
               'tie_priority': plan['tie_priority'], 'plan_sha256': file_hash(root/'plan.json'),
               'development_completion_sha256': file_hash(root/cohort/'complete.json')})
    print('Frozen selection:', selected, flush=True)


def summarize(root):
    plan = read_json(root/'plan.json')
    classes = plan['classes']
    selected = read_json(root/'selection.json')['selected']
    rows, paired, confusion = [], [], {}
    for cohort in plan['cohorts']:
        truth = read_json(root/'private'/f'{cohort}.json')
        labels = [r['label'] for r in truth]
        all_predictions = {c: read_predictions(root, cohort, c) for c in plan['text_conditions']}
        for condition, predictions in all_predictions.items():
            for method in METHODS:
                values = [p['class_scores'][method] for p in predictions]
                for subset, idx in [('all', list(range(10))),
                                    ('nonexplicit', [i for i, r in enumerate(truth) if not r['explicit']])]:
                    y, s = [labels[i] for i in idx], [values[i] for i in idx]
                    metrics = classification_stats(y, s, classes)
                    if idx and subset == 'all':
                        metrics.update(paired_bootstrap(y, s, classes))
                    rows.append({'cohort': cohort, 'text': condition, 'method': method,
                                 'subset': subset, **metrics})
                if condition == 'lm_prior':
                    confusion[f'{cohort}:{method}'] = classification_stats(labels, values, classes)['confusion']
        prior = all_predictions['lm_prior']
        for control in ['text_only', 'model_only']:
            values = [p[control] for p in prior]
            for subset, idx in [('all', list(range(10))),
                                ('nonexplicit', [i for i, r in enumerate(truth) if not r['explicit']])]:
                y, s = [labels[i] for i in idx], [values[i] for i in idx]
                metrics = classification_stats(y, s, classes)
                if idx and subset == 'all':
                    metrics.update(paired_bootstrap(y, s, classes))
                rows.append({'cohort': cohort, 'text': 'lm_prior', 'method': control,
                             'subset': subset, **metrics})
        for method in METHODS:
            a = [p['class_scores'][method] for p in prior]
            for other in ['baseline', 'text_only', 'model_only']:
                b = ([p['class_scores'][method] for p in all_predictions[other]] if other == 'baseline'
                     else [p[other] for p in prior])
                ma, mb = classification_stats(labels, a, classes), classification_stats(labels, b, classes)
                paired.append({'cohort': cohort, 'method': method, 'compared_to': other,
                               'accuracy_gain': ma['accuracy']-mb['accuracy'],
                               'supported_class_ba_gain': ma['supported_class_balanced_accuracy']-mb['supported_class_balanced_accuracy'],
                               **paired_bootstrap(labels, a, classes, b)})
            # Nine alternatives per image are a control distribution, not 90 independent samples.
            per_image = []
            for label, p in zip(labels, prior, strict=True):
                trials = [r[method] for r in p['shuffled_class_scores']]
                per_image.append(np.mean([s is not None and classes[int(np.argmin(s))] == label for s in trials]))
            supported = [np.mean([v for y, v in zip(labels, per_image) if y == c]) for c in classes if c in labels]
            rows.append({'cohort': cohort, 'text': 'lm_prior', 'method': 'shuffled_'+method,
                         'subset': 'all', 'n': 10, 'accuracy': float(np.mean(per_image)),
                         'supported_class_balanced_accuracy': float(np.mean(supported)),
                         'balanced_accuracy': float(np.mean(supported)) if len(supported) == 3 else None,
                         'per_image_accuracy': [float(v) for v in per_image],
                         'note': 'Mean of nine mismatched same-model uploads per image; 10 independent image groups'})
    report = {'selected': selected, 'classes': classes, 'rows': rows, 'paired': paired,
              'confusion': confusion, 'data_audit': read_json(root/'data_audit.json'),
              'costs': {name: read_json(root/name/'complete.json') for name in plan['cohorts']},
              'plan_sha256': file_hash(root/'plan.json'), 'selection_sha256': file_hash(root/'selection.json')}
    write_json(root/'report.json', report)
    write_json(root/'complete.json', {'report_sha256': file_hash(root/'report.json'),
               'cohorts': list(plan['cohorts']), 'selected': selected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['all', 'cohort', 'select', 'summarize'], default='all')
    parser.add_argument('--cohort')
    args = parser.parse_args()
    root = Path(args.output)
    torch.set_num_threads(1)
    try:
        if args.stage in ['all', 'cohort']:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU5 or torch.cuda.device_count() != 1:
                raise ValueError('Expose only authorized GPU 5')
        if args.stage == 'cohort':
            run_cohort(root, args.cohort)
        elif args.stage == 'select':
            select(root)
        elif args.stage == 'summarize':
            summarize(root)
        else:
            for cohort in ['rank-32-development', 'rank-8-development', 'rank-32-validation']:
                if cohort == 'rank-32-validation':
                    select(root)
                subprocess.run([sys.executable, '-m', 'evaluation.image_property_study',
                                '--output', str(root), '--stage', 'cohort', '--cohort', cohort], check=True)
            summarize(root)
    except Exception as error:
        write_json(root/'failed.json', {'stage': args.stage, 'cohort': args.cohort,
                   'type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    main()
