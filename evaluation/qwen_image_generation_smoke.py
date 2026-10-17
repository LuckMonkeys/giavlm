"""Local Qwen-Image generation smoke test; no private data or victim gradients."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    prompts = [
        ("control_en", "A realistic studio photograph of exactly two objects on a white table: a red cube on the left and a blue sphere on the right. Soft daylight, plain light gray background, no text."),
        ("control_zh", "一张写实的摄影照片：白色桌面上只有两个物体，左边是一个红色立方体，右边是一个蓝色球体。柔和的日光，纯浅灰色背景，没有文字。"),
        ("abdomen_ct", "A single grayscale axial CT scan slice of the upper abdomen, soft tissue window. Liver, spleen and kidneys visible, body centered against a black background. Realistic diagnostic radiology image, transverse cross section, no text, no labels, no arrows, no collage, no color."),
        ("abdomen_mri", "A single grayscale axial T2-weighted MRI scan slice of the upper abdomen. Liver, spleen and kidneys visible, body centered against a black background. Realistic diagnostic radiology image, transverse cross section, no text, no labels, no arrows, no collage, no color."),
        ("chest_ct", "A single grayscale axial CT scan slice of the chest, lung window. Both lungs and the heart visible, body centered against a black background. Realistic diagnostic radiology image, transverse cross section, no text, no labels, no arrows, no collage, no color."),
        ("chest_xray", "A single grayscale frontal PA chest X-ray radiograph. Both lungs, ribs, clavicles and the heart silhouette visible, chest centered against a black background. Realistic diagnostic radiology image, no text, no labels, no arrows, no collage, no color."),
    ]
    return [
        {"case": name, "seed": seed, "prompt": prompt}
        for name, prompt in prompts
        for seed in ([42] if name.startswith("control") else [42, 43])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    omni = None
    started = time.perf_counter()
    try:
        from PIL import Image, ImageDraw, ImageStat
        import torch
        from vllm_omni import Omni
        from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA GPU must be exposed.")
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 40 * 1024**3:
            raise RuntimeError("Less than 40 GiB free; stop without changing settings.")
        model_index = args.model / "model_index.json"
        plan = {
            "model": str(args.model), "model_index_sha256": sha256(model_index),
            "source_sha256": sha256(Path(__file__)), "uses_private_reference": False,
            "uses_victim_gradients": False, "width": 1024, "height": 1024,
            "dtype": "bfloat16", "cfg_scale": 1.0, "enforce_eager": True,
            "sample_sigmas": json.loads(model_index.read_text())["sample_sigmas"],
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "initial_free_mib": free_bytes / 1024**2,
            "versions": {name: importlib.metadata.version(name) for name in
                         ["torch", "vllm", "vllm-omni", "diffusers", "transformers", "Pillow"]},
            "cases": cases(),
        }
        save_json(args.output / "plan.json", plan)
        load_start = time.perf_counter()
        omni = Omni(model=str(args.model), dtype="bfloat16", enforce_eager=True,
                    init_timeout=900, stage_init_timeout=900)
        load_seconds = time.perf_counter() - load_start
        rows = []
        for case in plan["cases"]:
            generation_start = time.perf_counter()
            params = OmniDiffusionSamplingParams(
                width=1024, height=1024, seed=case["seed"], true_cfg_scale=1.0,
                num_outputs_per_prompt=1,
            )
            outputs = omni.generate({"prompt": case["prompt"], "modalities": ["image"]},
                                    sampling_params_list=[params])
            images = extract_images_from_outputs(outputs)
            if len(images) != 1 or images[0].size != (1024, 1024):
                raise RuntimeError("Unexpected image count or size.")
            name = f"{case['case']}_seed{case['seed']}.png"
            images[0].save(args.output / name)
            elapsed = time.perf_counter() - generation_start
            stats = ImageStat.Stat(images[0].convert("RGB"))
            if max(stats.stddev) < 1:
                raise RuntimeError(f"Nearly blank output: {name}")
            row = {**case, "file": name, "generation_and_save_seconds": elapsed,
                   "sha256": sha256(args.output / name), "mode": images[0].mode,
                   "size": list(images[0].size), "rgb_mean": stats.mean,
                   "rgb_stddev": stats.stddev}
            rows.append(row)
            save_json(args.output / name.replace(".png", ".json"), row)
            save_json(args.output / "progress.json", {"completed": len(rows), "total": 10})
            print(f"SAVED {name}: {elapsed:.3f} s", flush=True)
        # Contact sheet is an inspection artifact, not a clinical quality metric.
        sheet = Image.new("RGB", (1024, 5 * 550), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            x, y = (index % 2) * 512, (index // 2) * 550
            with Image.open(args.output / row["file"]) as original:
                sheet.paste(original.convert("RGB").resize((512, 512)), (x, y + 30))
            draw.text((x + 8, y + 8), row["file"], fill="black")
        sheet.save(args.output / "contact_sheet.jpg", quality=95)
        omni.close()
        omni = None
        save_json(args.output / "complete.json", {
            "status": "complete", "load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started, "images": rows,
            "timing_note": "Sequential requests; repeated prompts can reuse text conditioning. Includes PNG save.",
        })
    except BaseException:
        save_json(args.output / "failed.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if omni is not None:
            omni.close()


if __name__ == "__main__":
    main()
