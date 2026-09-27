"""Reproducible private-reference study; never an attacker input or benchmark result."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import time

from omegaconf import OmegaConf
from PIL import Image, ImageOps
import torch

from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.config import digest, load_config
from core.data import read_manifest
from core.types import Batch, Observation
from evaluation.discriminability_metrics import (RADII, candidates, direction_probe,
                                                image_scores, replay, update_scores)
from utils.discriminability_runner import dispatch, process_lock, worker_binding

SCHEMA = 1
DEFAULT_RESOURCES = {'gpu_ids': [5, 6], 'max_gpus': 2, 'max_jobs_per_gpu': 1,
                     'min_free_mib': 36000, 'fp32_min_free_mib': 70000, 'poll_seconds': 30}


def load_spec(path):
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    allowed = {'schema_version', 'manifest', 'protocol', 'conditions', 'resources',
               'development_count', 'validation_count', 'pilot_count', 'reference_count',
               'negative_count', 'direction_count', 'seed', 'lpips', 'historical_roots'}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get('schema_version') != SCHEMA:
        raise ValueError('Unsupported discriminability spec schema or unknown fields')
    for key in ['manifest', 'protocol', 'conditions']:
        if key not in spec:
            raise ValueError(f'Missing study field: {key}')
    for key, value in {'development_count': 20, 'validation_count': 20, 'pilot_count': 3,
                       'reference_count': 20, 'negative_count': 10, 'direction_count': 5,
                       'seed': 42, 'lpips': True, 'historical_roots': {}}.items():
        spec.setdefault(key, value)
    for key in ['development_count', 'validation_count', 'pilot_count', 'reference_count',
                'negative_count', 'direction_count']:
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    resources = {**DEFAULT_RESOURCES, **spec.get('resources', {})}
    if (set(resources) != set(DEFAULT_RESOURCES) or resources['gpu_ids'] != [5, 6]
            or resources['max_gpus'] != 2 or resources['max_jobs_per_gpu'] != 1):
        raise ValueError('Study resource policy is fixed to GPUs 5/6, two GPUs, one job per GPU')
    for key in ['min_free_mib', 'fp32_min_free_mib', 'poll_seconds']:
        if not isinstance(resources[key], (float, int)) or resources[key] <= 0:
            raise ValueError(f'Invalid resource value: {key}')
    spec['resources'] = resources
    names = []
    for condition in spec['conditions']:
        if set(condition) - {'name', 'strategy', 'round', 'snapshot'}:
            raise ValueError('Unknown condition field')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', condition['name']):
            raise ValueError('Invalid condition name')
        if condition['round'] and not condition.get('snapshot'):
            raise ValueError('Trained condition requires a snapshot')
        names.append(condition['name'])
        cfg = configuration(spec, condition, 'cpu')
        if cfg.model.device_map or cfg.model.max_memory:
            raise ValueError('Multi-device placement is forbidden in diagnostics')
        if cfg.model.image_size % 7:
            raise ValueError('Study image size must be divisible by 7')
        if cfg.model.family != 'tiny' and not spec['lpips']:
            raise ValueError('Real-model studies require LPIPS')
    if not names or len(set(names)) != len(names):
        raise ValueError('Condition names must be nonempty and unique')
    return spec


def configuration(spec, condition, device):
    overrides = [f'{section}.{key}={json.dumps(value)}'
                 for section, fields in spec['protocol'].items() for key, value in fields.items()]
    overrides += [f'model.device={device}',
                  f'training.fine_tuning_strategy={condition["strategy"]}',
                  f'training.server_round={condition["round"]}']
    cfg = load_config(overrides=overrides)
    train = cfg.training
    if (train.knowledge != 'text_known' or train.algorithm != 'fedsgd' or train.sample_count != 1
            or train.task != 'vqa' or train.token_lengths_known):
        raise ValueError('Study requires text_known, VQA, single-sample single-step FedSGD')
    return cfg


def ordered(rows, seed):
    return sorted(rows, key=lambda row: digest([seed, row['image_id']]))


def select_samples(spec):
    pools = {split: read_manifest(spec['manifest'], 'vqa', split, unique_images=True)
             for split in ['tune', 'eval']}
    pilot = pools['tune'][:spec['pilot_count']]
    if spec['development_count'] < len(pilot):
        raise ValueError('Development set must contain all pilot images')
    used = {r['image_id'] for r in pilot}
    development = pilot + ordered([r for r in pools['tune'] if r['image_id'] not in used],
                                  spec['seed'])[:spec['development_count'] - len(pilot)]
    validation = ordered(pools['eval'], spec['seed'])[:spec['validation_count']]
    used = {r['image_id'] for r in development}
    reference = ordered([r for r in pools['tune'] if r['image_id'] not in used],
                        spec['seed'])[:spec['reference_count']]
    for name, rows, count in [('pilot', pilot, spec['pilot_count']),
                              ('development', development, spec['development_count']),
                              ('validation', validation, spec['validation_count']),
                              ('reference', reference, spec['reference_count'])]:
        if len(rows) != count:
            raise ValueError(f'Insufficient unique images for {name}')
    if any(len(rows) < 2 * spec['negative_count'] + 1 for rows in pools.values()):
        raise ValueError('Insufficient images for disjoint hard/random negative sets')
    return {'pilot': pilot, 'development': development, 'validation': validation,
            'reference': reference, 'pools': pools}


def input_signature(spec):
    rows = read_manifest(spec['manifest'], 'vqa', unique_images=True)
    snapshots = {}
    for condition in spec['conditions']:
        if condition.get('snapshot'):
            path = Path(condition['snapshot'])
            meta = read_json(path / 'model.json')
            if meta.get('schema_version') != 4:
                raise ValueError('Study requires model schema v4; no silent migration')
            snapshots[condition['name']] = {
                name: file_hash(path / name) for name in ['model.json', 'model.safetensors']}
    return digest({'spec': spec, 'source': source_fingerprint(),
                   'manifest': file_hash(spec['manifest']), 'snapshots': snapshots,
                   'images': [(r['image_id'], file_hash(r['image'])) for r in rows]})


def validate_shared_lora_bases(spec):
    """Round-zero LoRA strategies must start from the exact same adapter basis."""
    if all(configuration(spec, c, 'cpu').model.family == 'tiny' for c in spec['conditions']):
        return
    initial = {c['strategy']: c for c in spec['conditions']
               if c['round'] == 0 and c.get('snapshot') and c['strategy'] in {'f_l', 'f_cl'}}
    if set(initial) != {'f_l', 'f_cl'}:
        raise ValueError('Study requires round-zero F-L and F-CL snapshots')
    states = {strategy: read_tensors(Path(condition['snapshot']) / 'model.safetensors')
              for strategy, condition in initial.items()}
    names = {name for name in states['f_l'] if '.lora_' in name}
    if not names or not names <= set(states['f_cl']):
        raise ValueError('Round-zero snapshots have different LoRA parameter names')
    for name in names:
        if not torch.equal(states['f_l'][name], states['f_cl'][name]):
            raise ValueError(f'Round-zero snapshots do not share the LoRA basis: {name}')


def prepare(spec, root, resume):
    validate_shared_lora_bases(spec)
    signature = input_signature(spec)
    pointer = root / 'study.json'
    if pointer.exists():
        if not resume:
            raise FileExistsError('Study exists; use --resume')
        study = read_json(pointer)
        if study['signature'] != signature:
            raise ValueError('Study source/protocol/data/snapshot fingerprint changed')
        return study
    if root.exists() and any(p.name != 'failures' for p in root.iterdir()):
        raise FileExistsError('Study output is nonempty')
    selection = select_samples(spec)
    write_json(root / 'private' / 'selection.json', selection)
    study = {'schema_version': SCHEMA, 'uses_private_reference': True, 'signature': signature,
             'spec': spec, 'source_sha256': source_fingerprint(),
             'selection_sha256': file_hash(root / 'private' / 'selection.json')}
    write_json(pointer, study)
    return study


def check_study(spec, root):
    study = read_json(root / 'study.json')
    if study['schema_version'] != SCHEMA or study['signature'] != input_signature(spec):
        raise ValueError('Study source/protocol/data/snapshot fingerprint changed')
    path = root / 'private' / 'selection.json'
    if file_hash(path) != study['selection_sha256']:
        raise ValueError('Frozen selection changed')
    return study, read_json(path)


def image_view(adapter, row):
    with Image.open(row['image']) as source:
        return adapter.prepare_image(
            ImageOps.exif_transpose(source).convert('RGB')).float()[None].contiguous()


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def batch_for(adapter, row, images):
    # Keep one shared fp32 image representation; replay alone casts the victim input.
    encoded = adapter.batch(images, [row['question']], [row['target']])
    return Batch(images, encoded.questions, encoded.targets)


def make_adapter(spec, condition, device, precision):
    from core.fl import restore_model
    from core.vlm_wrapper import build_model
    cfg = configuration(spec, condition, device)
    if condition.get('snapshot'):
        adapter = restore_model(condition['snapshot'], device)
        if asdict(adapter.spec) != asdict(cfg.model):
            raise ValueError('Snapshot model configuration differs from study')
        old = adapter.training_spec
        if (old.fine_tuning_strategy, old.server_round, old.lora_rank, old.lora_alpha) != (
                condition['strategy'], condition['round'], cfg.training.lora_rank,
                cfg.training.lora_alpha):
            raise ValueError('Snapshot strategy, round or LoRA parameterization differs')
        adapter.set_training_spec(cfg.training)
    else:
        adapter = build_model(cfg.model, cfg.training)
    adapter.eval()
    if precision == 'float32':
        adapter.float()
        adapter.spec = replace(adapter.spec, dtype='float32')
    return adapter


def commit_capture(adapter, batch, directory, model_fingerprint):
    from core.fl import load_observation, save_observation
    cast = Batch(batch.images.to(adapter.device, adapter.dtype).contiguous(),
                 batch.questions, batch.targets)
    if (directory / 'public' / 'observation.json').exists():
        observation = load_observation(directory / 'public', str(adapter.device))
        if (observation.model_fingerprint != model_fingerprint or
                asdict(observation.training) != asdict(adapter.training_spec) or
                observation.public_question_ids != batch.questions.cpu().tolist() or
                observation.public_target_ids != batch.targets.cpu().tolist()):
            raise ValueError('Stored capture does not match frozen model/protocol/text')
        return observation
    spec = adapter.training_spec
    update = {name: value.detach().clone() for name, value in replay(
        adapter, cast.images, cast).items()}
    questions, targets = adapter.decode(batch.questions), adapter.decode(batch.targets)
    observation = Observation(
        model=replace(adapter.spec), training=replace(spec), tensors=update,
        model_fingerprint=model_fingerprint,
        public_questions=list(questions) if spec.question_public else [],
        public_targets=list(targets) if spec.target_public else [],
        public_question_ids=batch.questions.detach().cpu().tolist() if spec.question_public else [],
        public_target_ids=batch.targets.detach().cpu().tolist() if spec.target_public else [])
    observation.validate()
    write_tensors(directory / 'private' / 'images.safetensors', {'images': batch.images})
    save_observation(directory / 'public', observation)
    return observation


def negative_images(adapter, truth, row, pool, count, seed):
    available = [r for r in pool if r['image_id'] != row['image_id']]
    ranked = []
    for candidate in available:
        x = image_view(adapter, candidate)
        ranked.append((image_scores(truth, x)['ssim'], candidate, x))
    ranked.sort(key=lambda item: (-item[0], digest([seed, item[1]['image_id']])))
    hard = ranked[:count]
    remaining = sorted(ranked[count:], key=lambda item: digest([seed, item[1]['image_id']]))[:count]
    return [(f'{kind}-{i:02d}', kind, item[2]) for kind, values in
            [('natural_hard', hard), ('natural_random', remaining)]
            for i, item in enumerate(values)]


def perceptual_model(enabled):
    if not enabled:
        return None
    import lpips
    return lpips.LPIPS(net='alex', verbose=False).cpu().eval()


def checked_record(path, signature, resume):
    if not path.exists():
        return None
    if not resume:
        raise FileExistsError('Diagnostic record exists; use --resume')
    saved = read_json(path)
    if saved.get('signature') != signature or saved.get('status') != 'completed':
        raise ValueError('Diagnostic record fingerprint or completion state differs')
    return saved


def reference_mean(adapter, batch, rows):
    mean = None
    for row in rows:
        update = replay(adapter, image_view(adapter, row), batch)
        if mean is None:
            mean = {k: v.detach().float().clone() / len(rows) for k, v in update.items()}
        else:
            for name, value in update.items():
                mean[name].add_(value.detach().float(), alpha=1 / len(rows))
        del update
    return mean


def historical_candidates(spec, condition, row, observation):
    """Read only committed matching captures; export opaque IDs, never private paths."""
    from core.fl import load_observation
    for root_index, root in enumerate(spec.get('historical_roots', {}).get(condition['name'], [])):
        for run_index, result in enumerate(sorted(Path(root).rglob('result.json'))):
            attack = result.parent
            capture = attack.parent / 'capture'
            truth_path = capture / 'private' / 'truth.json'
            if not truth_path.exists():
                continue
            truth = read_json(truth_path)
            if [s['sample_id'] for s in truth['samples']] != [row['sample_id']]:
                continue
            old = load_observation(capture / 'public', 'cpu')
            if (old.model_fingerprint != observation.model_fingerprint or
                    upload_protocol(old.training) != upload_protocol(observation.training) or
                    old.public_question_ids != observation.public_question_ids or
                    old.public_target_ids != observation.public_target_ids):
                continue
            files = [('final', attack / 'images.safetensors', 'images')]
            files += [(f'checkpoint-{i:04d}', p / 'state.safetensors', 'candidate.images')
                      for i, p in enumerate(sorted((attack / 'checkpoints').glob('*')))]
            for label, path, key in files:
                if not path.exists():
                    continue
                saved = read_tensors(path)
                if key not in saved:
                    continue
                yield f'history-{root_index}-{run_index}-{label}', {'family': 'historical'}, saved[key]


def upload_protocol(training):
    """Fields that determine a single client's observed update and public inputs."""
    names = [
        'fine_tuning_strategy', 'algorithm', 'task', 'knowledge', 'token_lengths_known',
        'batch_size', 'local_steps', 'gradient_accumulation_steps', 'lr', 'local_optimizer',
        'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'training_protocol',
        'lora_rank', 'lora_alpha', 'server_round', 'two_stage_connector_rounds',
        'upload_parameters',
    ]
    return {name: getattr(training, name) for name in names}


def run_worker(args, spec, study, selection):
    conditions = [c for c in spec['conditions'] if not args.condition or c['name'] == args.condition]
    if args.worker and len(conditions) != 1:
        raise ValueError('Worker requires exactly one condition')
    if not conditions:
        raise ValueError('Unknown diagnostic condition')
    device = args.device or 'cuda:0'
    tiny = all(configuration(spec, c, device).model.family == 'tiny' for c in conditions)
    binding = worker_binding(device, tiny)
    lock = process_lock(f'gpu-{binding["physical_gpu"]}') if binding else nullcontext()
    with lock:
        for condition in conditions:
            _condition_worker(args, spec, study, selection, condition, device, binding)


def _condition_worker(args, spec, study, selection, condition, device, binding):
    root = Path(args.output)
    rows = selection[args.cohort]
    if args.stage == 'directions':
        rows = ordered(rows, spec['seed'])[:spec['direction_count']]
    perceptual = perceptual_model(spec['lpips'])
    adapter = make_adapter(spec, condition, device, args.precision)
    started = time.monotonic()
    if binding:
        torch.cuda.reset_peak_memory_stats()
    output = root / 'records' / args.cohort / args.precision / condition['name']
    # Preserve the cohort ID across scoring and direction subsets.
    identifiers = {row['image_id']: f'image-{i:03d}' for i, row in enumerate(selection[args.cohort])}
    model_fingerprint = adapter.fingerprint()
    completed = 0
    for index, row in enumerate(rows):
        identifier = identifiers[row['image_id']]
        directory = output / identifier
        truth = image_view(adapter, row)
        batch = batch_for(adapter, row, truth)
        observation = commit_capture(adapter, batch, directory / 'capture', model_fingerprint)
        target = observation.tensors
        if args.stage == 'directions':
            wrong_row = rows[(index + 1) % len(rows)]
            if len(rows) < 2:
                raise ValueError('Wrong-update control requires at least two direction images')
            wrong_batch = batch_for(adapter, wrong_row, image_view(adapter, wrong_row))
            wrong = replay(adapter, wrong_batch.images, wrong_batch)
            for alpha in RADII:
                for kind in ['cosine', 'relative_l2']:
                    for control, update in [('correct', target), ('wrong_update', wrong)]:
                        path = directory / 'directions' / f'{alpha:g}-{kind}-{control}.json'
                        signature = digest([study['signature'], args.precision, condition['name'],
                                            identifier, alpha, kind, control, model_fingerprint])
                        if checked_record(path, signature, args.resume):
                            continue
                        result = direction_probe(adapter, batch, update, alpha, kind, perceptual)
                        write_json(path, {'status': 'completed', 'signature': signature,
                                          'uses_private_reference': True, 'control': control, **result})
                        completed += 1
            del wrong
        else:
            mean = reference_mean(adapter, batch, selection['reference'])
            others = negative_images(adapter, truth, row, selection['pools'][row['split']],
                                     spec['negative_count'], spec['seed'])
            from itertools import chain
            variants = chain(candidates(truth, others), historical_candidates(
                spec, condition, row, observation) if args.precision == 'native' else [])
            candidate_ids = []
            for candidate_id, metadata, images in variants:
                images = images.float().cpu().clamp(0, 1)
                if images.shape != truth.shape or not torch.isfinite(images).all():
                    raise ValueError('Invalid candidate image')
                candidate_ids.append(candidate_id)
                image_hash = tensor_hash(images)
                signature = digest([study['signature'], args.precision, condition['name'],
                                    identifier, candidate_id, image_hash, model_fingerprint])
                path = directory / 'scores' / f'{candidate_id}.json'
                if checked_record(path, signature, args.resume):
                    continue
                then = time.monotonic()
                predicted = replay(adapter, images, batch)
                scores = update_scores(predicted, target, mean, adapter.is_connector)
                error = scores['update']['relative_l2']
                if candidate_id == 'truth' and error is not None and error > 1e-12:
                    raise ValueError('Truth replay exceeds relative squared error 1e-12')
                from attacks.priors import total_variation
                record = {'status': 'completed', 'uses_private_reference': True,
                          'signature': signature, 'candidate_id': candidate_id,
                          'image_sha256': image_hash, **metadata, **scores,
                          'image': image_scores(truth, images, perceptual),
                          'tv': float(total_variation(images)), 'seconds': time.monotonic() - then}
                write_json(path, record)
                del predicted
                completed += 1
            write_json(directory / 'candidate_index.json', {'candidate_ids': candidate_ids,
                                                           'signature': study['signature']})
            del mean
        print(json.dumps({'condition': condition['name'], 'image': identifier,
                          'stage': args.stage, 'status': 'completed'}), flush=True)
    write_json(output / f'{args.stage}-complete.json', {
        'status': 'completed', 'uses_private_reference': True, 'signature': study['signature'],
        'model_fingerprint': model_fingerprint, 'images': len(rows), 'new_records': completed,
        'seconds': time.monotonic() - started, 'binding': binding,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if binding else None,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20 if binding else None})
    del adapter
    if binding:
        torch.cuda.empty_cache()


def freeze(root, study):
    hashes = {}
    for condition in study['spec']['conditions']:
        directory = root / 'records' / 'development' / 'native' / condition['name']
        for stage in ['score', 'directions']:
            path = directory / f'{stage}-complete.json'
            if read_json(path)['signature'] != study['signature']:
                raise ValueError('Cannot freeze incomplete or mismatched development results')
            hashes[f'{condition["name"]}/{stage}'] = file_hash(path)
    write_json(root / 'freeze.json', {'signature': study['signature'], 'development': hashes,
                                      'uses_private_reference': True})


def execute(args):
    root = Path(args.output)
    try:
        spec = load_spec(args.spec)
        if args.stage == 'prepare':
            # Different study roots are independent; reject only duplicate preparation
            # of the same root rather than serializing unrelated test/experiment trees.
            lock_id = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
            with process_lock(f'prepare-{lock_id}'):
                prepare(spec, root, args.resume)
            return
        study, selection = check_study(spec, root)
        if args.stage == 'freeze':
            freeze(root, study)
            return
        if args.precision == 'float32' and args.cohort != 'pilot':
            raise ValueError('The fp32 control is restricted to the pilot cohort')
        if args.cohort == 'validation':
            if read_json(root / 'freeze.json')['signature'] != study['signature']:
                raise ValueError('Validation requires the frozen development protocol')
        if args.stage == 'report':
            from evaluation.discriminability_report import report
            report(root, study, args.cohort, args.precision)
        elif args.worker or args.device == 'cpu':
            run_worker(args, spec, study, selection)
        else:
            dispatch(args, spec)
    except BaseException as error:
        # Error text may contain private paths: keep it inside the diagnostic tree.
        write_json(root / 'failures' / f'{args.stage}-{os.getpid()}.json', {
            'status': 'failed', 'uses_private_reference': True,
            'error_type': type(error).__name__, 'error': str(error)})
        raise
