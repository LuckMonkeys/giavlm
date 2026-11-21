from dataclasses import asdict
from types import SimpleNamespace

import pytest

from core.config import TrainingSpec
from evaluation.medical_kv_cohort_a import validate_observation


def test_original_a_protocol_is_preserved_with_public_lengths_but_without_public_qa():
    training = TrainingSpec(knowledge="private", token_lengths_known=True)
    observation = SimpleNamespace(model_fingerprint="frozen-model", training=training,
        sample_count=1, public_images=None, public_questions=[], public_targets=[],
        public_question_ids=[], public_target_ids=[])
    plan = {"model_fingerprint": "frozen-model", "training": asdict(training)}
    validate_observation(observation, plan)
    observation.public_questions = ["Reference QA must not enter this inference path"]
    with pytest.raises(ValueError, match="Unexpected original cohort A"):
        validate_observation(observation, plan)
    observation.public_questions = []
    observation.training.lr *= 2
    with pytest.raises(ValueError, match="Unexpected original cohort A"):
        validate_observation(observation, plan)
