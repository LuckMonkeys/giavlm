"""Fixed mixed-modality head/non-head bank; no target input or victim gradients."""

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
    common = " Realistic grayscale radiology appearance, anatomy centered against a black background. No text, no labels, no arrows, no collage, no color."
    groups = [
        ("head", "CT", "head", "A single axial CT slice of the head showing the brain, brain soft tissue window."),
        ("head", "MRI", "head", "A single axial T2-weighted MRI slice of the head showing the brain."),
        ("head", "X-Ray", "head", "A single lateral skull X-ray radiograph showing the whole head and skull."),
        ("non-head", "CT", "abdomen", "A single axial CT slice of the upper abdomen, soft tissue window."),
        ("non-head", "CT", "chest", "A single axial CT slice of the chest, lung window."),
        ("non-head", "MRI", "abdomen", "A single axial T2-weighted MRI slice of the upper abdomen."),
        ("non-head", "MRI", "chest", "A single axial T2-weighted MRI slice of the chest showing the heart and lungs."),
        ("non-head", "X-Ray", "chest", "A single frontal chest X-ray radiograph showing both lungs and the heart silhouette."),
    ]
    rows = []
    offsets = {}
    for label, modality, region, prompt in groups:
        count = 4 if label == "head" or modality == "X-Ray" else 2
        key = (label, modality)
        start = offsets.get(key, 0)
        for j in range(start, start + count):
            rows.append({"case": f"{label}_{modality.lower()}_{region}",
                         "property_label": label, "modality": modality, "region": region,
                         "seed": 202610170 + j, "prompt": prompt + common})
        offsets[key] = start + count
    return rows


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

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d":
            raise RuntimeError("Only GPU 5 is authorized")
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
            save_json(args.output / "progress.json", {"completed": len(rows), "total": len(plan["cases"])})
            print(f"SAVED {name}: {elapsed:.3f} s", flush=True)
        # Contact sheet is an inspection artifact, not a clinical quality metric.
        sheet = Image.new("RGB", (1200, ((len(rows) + 3) // 4) * 340), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows):
            x, y = (index % 4) * 300, (index // 4) * 340
            with Image.open(args.output / row["file"]) as original:
                sheet.paste(original.convert("RGB").resize((300, 300)), (x, y + 30))
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
