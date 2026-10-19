"""Public synthetic GPU gate for Qwen3.5; this is not a benchmark measurement."""
import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import socket
import time
import traceback

import torch

from attacks.analytic.dager_adapted import gradient_groups
from attacks.factory import create_attacker
from core.adapters.qwen35 import Qwen35TextView
from core.artifacts import environment, write_json
from core.config import AttackSpec, DAGEROptions, ModelSpec, TrainingSpec
from core.fl import capture, save_observation, simulate_update
from core.knowledge import AdversaryKnowledge
from core.vlm_wrapper import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    state = {"status": "running", "phase": "preflight", "host": socket.gethostname(),
             "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "benchmark_result": False, "public_synthetic_data": True}
    started = time.monotonic()

    def progress(phase):
        state.update(phase=phase, seconds=time.monotonic() - started)
        write_json(output / "state.json", state)
        print(phase, flush=True)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Smoke test requires exactly one visible CUDA GPU")
        state["environment"] = environment()
        model = ModelSpec(
            family="qwen3_5", name="/home/zx/public/model-hub/huggingface/Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", device="cuda:0", dtype="bfloat16",
            image_size=256, question_length=64, target_length=64, patch_size=16)
        training = TrainingSpec(knowledge="image_known", token_lengths_known=True,
                                lora_rank=32, lora_alpha=64)
        progress("load_model")
        adapter = build_model(model, training)
        state["trainable_parameters"] = sum(p.numel() for p in adapter.trainable().values())
        state["model"] = asdict(model)
        state["training"] = asdict(training)
        image = torch.zeros(1, 3, 256, 256, device="cuda:0", dtype=torch.bfloat16)
        image[:, 0] = 1
        batch = adapter.batch(image, ["what color?"], ["red"])

        # A public warm-up is necessary: fresh LoRA-B is zero, hence dL/dA is zero.
        # Never treat this synthetic one-step state as the ten-round victim.
        progress("public_warmup_backward")
        gradients = simulate_update(adapter, batch, training)
        if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("Warm-up gradients are empty or nonfinite")
        with torch.no_grad():
            for name, parameter in adapter.trainable().items():
                parameter.add_(gradients[name], alpha=-.01)
        del gradients
        progress("capture")
        obs = capture(adapter, batch, ["what color?"], ["red"])
        save_observation(output / "capture", obs)
        state["gradient_groups"] = gradient_groups(adapter, obs, "linear_inputs")
        state["gradient_norms"] = [[float(obs.tensors[name].float().norm()) for name in group]
                                   for group in state["gradient_groups"]]

        progress("prefix_replay")
        view = Qwen35TextView(adapter, obs)
        recorded = {}
        def hook(module, inputs):
            recorded["input"] = inputs[0].detach().clone()
        handle = adapter.backend.model.language_model.layers[1].linear_attn.in_proj_qkv.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                state["synthetic_loss"] = float(adapter(batch.images, batch.questions, batch.targets))
        finally:
            handle.remove()
        truth = {"questions": batch.questions[0], "targets": batch.targets[0]}
        prefix = [int(truth[field][index]) for field, index, _ in view.slots]
        differences = []
        for depth in range(1, len(prefix) + 1):
            actual = view.prefix_features([prefix[:depth]])
            expected = recorded["input"][:, view.slots[depth - 1][2]]
            differences.append(float((actual.float() - expected.float()).abs().max()))
            torch.testing.assert_close(actual, expected, atol=.06, rtol=.03)
        state["prefix_max_absolute_errors"] = differences
        recorded.clear()
        del view

        for name, observation in (("gt", obs), ("random", replace(
                obs, training=replace(training, knowledge="private"), public_images=None))):
            progress(f"attack_{name}")
            adapter.set_training_spec(observation.training)
            options = DAGEROptions(
                projections="linear_inputs", image_source="declared" if name == "gt" else "random",
                token_selection="topk", max_candidates=32, beam_width=2,
                vocab_chunk_size=8192, prefix_batch_size=32, lm_prior_weight=.01)
            spec = AttackSpec(method="dager_adapted", text_method="none", seconds=600,
                              max_evaluations=5000, dager=options)
            result = create_attacker(adapter, spec).attack(
                observation.tensors, observation, AdversaryKnowledge(observation.training.knowledge, True),
                directory=output / f"attack-{name}")
            write_json(output / f"attack-{name}" / "result.json", asdict(result))
            state[f"attack_{name}"] = {"status": result.status, "reason": result.reason,
                                      "costs": result.costs}
            if result.status != "completed":
                raise RuntimeError(f"Synthetic {name} search did not complete: {result.status}: {result.reason}")
        state.update(status="completed", peak_memory_bytes=torch.cuda.max_memory_allocated())
        progress("finished")
    except BaseException:
        state.update(status="failed", traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
