from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from core.artifacts import read_json, write_json
from core.config import load_config
from core.data import synthetic
from core.vlm_wrapper import build_model
from evaluation.discriminability_metrics import (alignment, bootstrap, candidates,
                                                candidate_statistics, objective, replay,
                                                update_scores)
from evaluation.gradient_discriminability import (configuration, execute, load_spec,
                                                 select_samples, upload_protocol)
from utils.discriminability_runner import (DiagnosticRunner, process_lock, validate_gpu_ids,
                                           worker_binding)


def test_candidates_reproducible_and_structurally_distinct():
    truth = torch.rand(1, 3, 14, 14, generator=torch.Generator().manual_seed(3))
    others = [(f'other-{i}', 'natural_random', truth.flip(-1)) for i in range(20)]
    first, second = list(candidates(truth, others)), list(candidates(truth, others))
    assert len(first) == 95 and len({item[0] for item in first}) == 95
    assert all(torch.equal(a[2], b[2]) for a, b in zip(first, second))
    by_name = {name: x for name, _, x in first}
    assert torch.equal(by_name['shift-right-1'][..., 1:], truth[..., :-1])
    assert torch.equal(by_name['shift-down-4'][..., 4:, :], truth[..., :-4, :])
    assert torch.equal(by_name['constant-0'], torch.zeros_like(truth))
    assert all(x.dtype == torch.float32 and x.min() >= 0 and x.max() <= 1 for _, _, x in first)


def test_metrics_distinguish_direction_magnitude_and_zero_updates():
    target = {'a': torch.tensor([1., 2.]), 'b': torch.zeros(2)}
    predicted = {k: v * 3 for k, v in target.items()}
    result = update_scores(predicted, target)
    assert result['losses']['cosine'] == pytest.approx(0, abs=1e-7)
    assert result['losses']['relative_l2'] == pytest.approx(4)
    assert result['losses']['equal_tensor_cosine'] == pytest.approx(0, abs=1e-7)
    assert result['tensors']['b']['cosine'] is None
    mean = {k: torch.ones_like(v) * .25 for k, v in target.items()}
    centered_l2 = sum(((predicted[k] - mean[k]) - (target[k] - mean[k])).square().sum()
                      for k in target)
    assert centered_l2 == sum((predicted[k] - target[k]).square().sum() for k in target)
    assert update_scores(predicted, target, mean)['losses']['centered_cosine'] > 0
    assert update_scores({'z': torch.zeros(2)}, {'z': torch.zeros(2)})['losses']['cosine'] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cross-device scoring needs CUDA')
def test_update_scores_moves_deserialized_cpu_target_to_candidate_device():
    predicted = {'a': torch.tensor([1., 2.], device='cuda')}
    target = {'a': torch.tensor([1., 2.], device='cpu')}
    result = update_scores(predicted, target)
    assert result['losses']['cosine'] == pytest.approx(0, abs=1e-7)
    assert result['losses']['relative_l2'] == 0


def test_ordering_ties_empty_and_image_bootstrap():
    def rows(values):
        return [{'losses': {'cosine': loss}, 'image': {'mse': mse, 'one_minus_ssim': mse}}
                for loss, mse in zip(values, [.0001, .0005, .1, .2])]
    good = candidate_statistics(rows([0, 1, 2, 3]), 'cosine')
    bad = candidate_statistics(rows([3, 2, 1, 0]), 'cosine')
    ties = candidate_statistics(rows([1, 1, 1, 1]), 'cosine')
    assert good['near_win']['value'] == 1 and bad['near_win']['value'] == 0
    assert good['spearman']['mse']['value'] == pytest.approx(1)
    assert bad['spearman']['mse']['value'] == pytest.approx(-1)
    assert ties['near_win']['value'] == .5
    assert ties['bottom_fraction']['0.01']['count'] == 4
    assert ties['spearman']['mse']['reason'] == 'constant_values'
    assert candidate_statistics([], 'cosine')['near_win']['value'] is None
    assert bootstrap([0., 1., None])['images'] == 2
    assert bootstrap([0., 1.]) == bootstrap([0., 1.])


def test_image_derivative_matches_finite_difference():
    cfg = load_config(overrides=['training.knowledge=text_known',
                                 'training.fine_tuning_strategy=f_c', 'model.dtype=float64'])
    adapter = build_model(cfg.model, cfg.training)
    truth = torch.rand(1, 3, 8, 8, dtype=torch.float64)
    batch = adapter.batch(truth, ['question'], ['answer'])
    target = replay(adapter, truth, batch)
    start = (truth * .95 + .025).requires_grad_(True)
    direction = torch.randn_like(start)
    direction /= direction.norm()
    # Matching uses fp32 sums by design; finite differences need a resolvable step.
    loss = objective(replay(adapter, start, batch, True), target, 'relative_l2')
    derivative, = torch.autograd.grad(loss, start)
    step = 1e-3
    high = objective(replay(adapter, start.detach() + step * direction, batch), target, 'relative_l2')
    low = objective(replay(adapter, start.detach() - step * direction, batch), target, 'relative_l2')
    assert float((derivative * direction).sum()) == pytest.approx(
        float((high - low) / (2 * step)), rel=.05, abs=1e-6)
    assert alignment(torch.zeros(2), torch.ones(2)) is None
    assert alignment(torch.ones(2), -torch.ones(2)) == pytest.approx(-1)


@pytest.mark.parametrize('value', ['0', '7', '5,6,7', '5,5', '', '5,6,5'])
def test_invalid_gpu_limits(value):
    with pytest.raises(ValueError, match='GPU'):
        validate_gpu_ids(value)


def test_worker_binding_and_duplicate_lock(monkeypatch):
    inventory = {'5': {'uuid': 'GPU-five'}, '6': {'uuid': 'GPU-six'}}
    monkeypatch.setenv('GIAVLM_DIAGNOSTIC_GPU', '5')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-five')
    assert worker_binding('cuda:0', inventory=inventory)['physical_gpu'] == 5
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-five,GPU-six')
    with pytest.raises(ValueError, match='exactly'):
        worker_binding('cuda:0', inventory=inventory)
    with pytest.raises(ValueError, match='single'):
        worker_binding('cuda:1', inventory=inventory)
    with pytest.raises(ValueError, match='tiny'):
        worker_binding('cpu')
    assert worker_binding('cpu', tiny=True) is None
    with process_lock('test-discriminability'):
        with pytest.raises(RuntimeError, match='held'):
            with process_lock('test-discriminability'):
                pass


@pytest.fixture
def study_args(tmp_path):
    manifest = synthetic(tmp_path / 'data', 180, clients=1)
    spec = {'schema_version': 1, 'manifest': str(manifest), 'lpips': False,
            'pilot_count': 2, 'development_count': 2, 'validation_count': 2,
            'reference_count': 2, 'negative_count': 2, 'direction_count': 2,
            'protocol': {'model': {'image_size': 14, 'patch_size': 2},
                         'training': {'knowledge': 'text_known', 'clients': 1,
                                      'clients_per_round': 1}},
            'conditions': [{'name': 'fc', 'strategy': 'f_c', 'round': 0}]}
    path = tmp_path / 'spec.json'
    write_json(path, spec)
    return SimpleNamespace(spec=str(path), output=str(tmp_path / 'study'), stage='prepare',
                           resume=False, cohort='pilot', precision='native', condition=None,
                           worker=False, device='cpu', gpu_ids='5,6', threads=1)


def test_selection_disjoint_and_protocol_fixed(study_args):
    spec = load_spec(study_args.spec)
    selected = select_samples(spec)
    groups = {k: {r['image_id'] for r in selected[k]}
              for k in ['pilot', 'development', 'validation', 'reference']}
    assert groups['pilot'] <= groups['development']
    assert not groups['development'] & groups['validation']
    assert not groups['reference'] & groups['development']
    assert select_samples(spec) == selected
    cfg = configuration(spec, spec['conditions'][0], 'cpu')
    assert cfg.training.knowledge == 'text_known'


def test_historical_compatibility_ignores_federation_metadata():
    base = load_config(overrides=['training.knowledge=text_known']).training
    changed = replace(base, clients=1, clients_per_round=1, rounds=2, snapshots=[0, 1], seed=99)
    assert upload_protocol(base) == upload_protocol(changed)
    assert upload_protocol(base) != upload_protocol(replace(changed, lr=base.lr * 2))


def test_study_scoring_resume_freeze_and_directions(study_args):
    from pathlib import Path
    execute(study_args)
    root = Path(study_args.output)
    study_args.stage = 'score'
    execute(study_args)
    directory = root / 'records/pilot/native/fc'
    records = list((directory / 'image-000/scores').glob('*.json'))
    assert len(records) == 79
    truth = read_json(directory / 'image-000/scores/truth.json')
    assert truth['image']['mse'] == 0 and truth['losses']['relative_l2'] < 1e-12
    assert truth['uses_private_reference']
    stamp = records[0].stat().st_mtime_ns
    study_args.resume = True
    execute(study_args)
    assert records[0].stat().st_mtime_ns == stamp
    study_args.cohort = 'validation'
    with pytest.raises(FileNotFoundError):
        execute(study_args)
    assert list((root / 'failures').glob('*.json'))
    study_args.cohort = 'development'
    execute(study_args)
    study_args.stage = 'directions'
    execute(study_args)
    directions = list((root / 'records/development/native/fc/image-000/directions').glob('*.json'))
    assert len(directions) == 24
    assert len(read_json(directions[0])['steps']) == 6
    study_args.stage = 'freeze'
    execute(study_args)
    assert read_json(root / 'freeze.json')['signature'] == read_json(root / 'study.json')['signature']
    study_args.stage, study_args.cohort = 'score', 'validation'
    execute(study_args)
    # Refuse any source data change rather than mixing records during resume.
    spec = read_json(study_args.spec)
    spec['seed'] = 999
    write_json(study_args.spec, spec)
    with pytest.raises(ValueError, match='fingerprint'):
        execute(study_args)


def test_report_has_paired_image_units_and_artifacts(study_args):
    from pathlib import Path
    execute(study_args)
    study_args.stage = 'score'
    execute(study_args)
    study_args.stage = 'directions'
    execute(study_args)
    study_args.stage = 'report'
    execute(study_args)
    path = Path(study_args.output) / 'reports/pilot/native/report.json'
    report = read_json(path)
    assert report['uses_private_reference']
    assert report['conditions']['fc']['summary']['cosine']['all']['near_win']['images'] == 2
    assert 'image_id' not in json.dumps(report) and 'sample_id' not in json.dumps(report)
    for name in ['distance_loss', 'near_win', 'low_loss_errors', 'local_directions',
                 'module_discriminability']:
        assert (path.parent / f'{name}.png').exists()


def test_scheduler_uses_uuid_and_failure_is_not_retried(tmp_path):
    calls = []

    def popen(argv, **kwargs):
        calls.append(kwargs['env'])
        raise OSError('launch failed')

    job = {'name': 'test', 'kind': 'command', 'argv': ['python', '-V'], 'min_free_mib': 1}
    schedule = {'output_root': tmp_path, 'jobs': [job], 'scheduler': {}}
    runner = DiagnosticRunner(schedule, {'5': {'uuid': 'GPU-five'}}, popen=popen)
    with pytest.raises(RuntimeError, match='failed to start'):
        runner.start(job, '5')
    assert calls[0]['CUDA_VISIBLE_DEVICES'] == 'GPU-five'
    assert calls[0]['GIAVLM_DIAGNOSTIC_GPU'] == '5'
    assert len(calls) == 1
