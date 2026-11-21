import torch

from evaluation.medical_kv_data import select_public_module
from evaluation.medical_kv_transfer import METHODS, score


def test_module_rule_prioritizes_worst_qa_not_large_average_and_has_no_medical_input():
    public = {prop: {"L24_q": {"old_six_qa_worst_balanced_accuracy": 1. if prop == "head" else .5,
                             "old_six_qa_mean_balanced_accuracy": 1. if prop == "head" else .99},
                     "L24_v": {"old_six_qa_worst_balanced_accuracy": 1. if prop == "head" else .7,
                               "old_six_qa_mean_balanced_accuracy": 1. if prop == "head" else .8}}
              for prop in ("head", "abdomen")}
    rule = select_public_module(public)
    assert rule["primary"] == "L24_v" and rule["target_labels_used"] is False


def test_medical_classifier_reads_module_specific_uploads_in_shared_input_coordinates():
    prototypes = {prop: {"visual_24": torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
                        "visual_32": torch.tensor([[1., 0., 0.], [0., 1., 0.]])} for prop in ("head", "abdomen")}
    gradients = {}
    for method in METHODS:
        layer, module = method.split("_")
        value = [[0., 2., 0.]] if module == "k" else [[3., 0., 0.]]
        key = f"backend.language_model.model.layers.{int(layer[1:])-1}.self_attn.{module}_proj.lora_A.default.weight"
        gradients[key] = torch.tensor(value)
    result = score(gradients, prototypes)
    assert result["head"]["L24_q"] == [0., 1.]
    assert result["head"]["L24_k"] == [1., 0.]
    assert result["abdomen"]["L32_v"] == [0., 1.]
    assert result == score({key: -17 * value for key, value in gradients.items()}, prototypes)
