from dataclasses import asdict, replace

import torch

from core.artifacts import read_json, read_tensors
from core.config import Config
from core.fl import load_observation
from evaluation.dager_rank_study import attack_public, capture_pair, rank_configs
from test_dager import fixture, run, specification


def test_rank_configs_keep_training_schedule_and_lora_scale():
    base = Config()
    trained = replace(base.training, algorithm='fedavg', local_optimizer='adamw',
                      batch_size=2, local_steps=10, gradient_accumulation_steps=4,
                      rounds=10, server_round=10, snapshots=[5, 10], lr=1e-4)
    metadata = {'training': asdict(trained), 'model': asdict(base.model)}
    for rank in (16, 32):
        train, attack = rank_configs(base, metadata, rank)
        assert train.training == replace(trained, lora_rank=rank, lora_alpha=2*rank, server_round=0)
        assert attack.training == replace(base.training, lora_rank=rank, lora_alpha=2*rank)
        assert attack.attack == base.attack


def test_capture_pair_and_reused_adapter_match_native_attack(tmp_path, monkeypatch):
    torch.manual_seed(81)
    adapter, batch, _ = fixture(knowledge='image_known')
    monkeypatch.setattr('evaluation.dager_rank_study.load_batch', lambda rows, model: batch)
    capture_pair(adapter, {'question': 'what color', 'target': 'red circle'},
                 adapter.training_spec, tmp_path/'run', tmp_path/'model')
    known = tmp_path/'run/captures/known/public'
    private = tmp_path/'run/captures/private/public'
    assert read_json(known/'observation.json')['update_sha256'] == read_json(private/'observation.json')['update_sha256']
    assert load_observation(private).public_images is None
    assert not (private/'public_images.safetensors').exists()
    for public, image_source in ((known, 'declared'), (private, 'random')):
        config = replace(Config(), attack=specification(image_source=image_source,
                         token_selection='topk', max_candidates=8, beam_width=2))
        obs = load_observation(public)
        adapter.set_training_spec(obs.training)
        expected = run(adapter, obs, config.attack, directory=tmp_path/f'native-{image_source}')
        path = tmp_path/f'wrapped-{image_source}'
        status, _ = attack_public(adapter, config, public, path, adapter.fingerprint())
        actual = read_json(path/'result.json')
        assert status == expected.status == 'completed'
        assert actual['questions'] == expected.questions
        assert actual['targets'] == expected.targets
        for key, value in read_tensors(tmp_path/f'native-{image_source}'/'token_candidates.safetensors').items():
            torch.testing.assert_close(value, read_tensors(path/'token_candidates.safetensors')[key], rtol=0, atol=0)
