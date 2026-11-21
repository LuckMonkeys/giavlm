"""Local fixed-budget public style bank generation, without victim inputs."""
import argparse
import os
from pathlib import Path
import time
import traceback

from core.artifacts import file_hash, read_json, write_json


STYLES = {
    "photo": "A realistic studio photograph.",
    "line": "A clean monochrome contour line drawing with simple flat gray shading.",
    "render": "A realistic three-dimensional computer-rendered illustration with smooth neutral materials.",
}


def generation_cases():
    """Paired content groups; labels and fit/dev partition fixed before generation."""
    heads = [
        "a front view of an adult man's head and face, cropped below the chin",
        "a front view of an adult woman's head and face, cropped below the chin",
        "a left profile view of an adult man's head, cropped below the chin",
        "a right profile view of an adult woman's head, cropped below the chin",
        "a three-quarter view of an older man's head and face, cropped below the chin",
        "a three-quarter view of an older woman's head and face, cropped below the chin",
        "a back view of a person's head with short hair, cropped below the neck",
        "a back view of a person's head with long tied hair, cropped below the neck",
        "a front view of a person's head wearing glasses, cropped below the chin",
        "a side view of a person's head wearing glasses, cropped below the chin",
        "a three-quarter view of a bald person's head, cropped below the chin",
        "a front view of a person's head with curly hair, cropped below the chin",
        "a side view of an older person's head with gray hair, cropped below the chin",
        "a front view of a person's head with wavy hair, cropped below the chin",
        "a back view of a person's bald head, cropped below the neck",
        "a three-quarter view of a person's head with short straight hair, cropped below the chin",
    ]
    negatives = [
        "a single person's left forearm and open hand, with no head or face in the frame",
        "a single person's right forearm and closed hand, with no head or face in the frame",
        "a person's lower leg and bare foot in side view, with no head or face in the frame",
        "a person's lower leg and bare foot in front view, with no head or face in the frame",
        "a person's torso covered by a plain long-sleeve shirt, cropped from shoulders to waist, no head or face",
        "a person's torso covered by a plain short-sleeve shirt, cropped from shoulders to waist, no head or face",
        "a person's pair of hands resting together, with no head or face in the frame",
        "a person's pair of feet, with no head or face in the frame",
        "a single person's open palm and wrist in front view, with no head or face in the frame",
        "a single person's hand and wrist in side view, with no head or face in the frame",
        "a person's bent elbow and upper arm, with no head or face in the frame",
        "a person's knee and lower thigh, with no head or face in the frame",
        "a person's clothed back cropped from shoulders to waist, with no head or face in the frame",
        "a person's torso in a plain sweater, cropped below the neck and above the waist, no head or face",
        "a person's ankle and heel in side view, with no head or face in the frame",
        "a person's two forearms crossed, with no head or face in the frame",
    ]
    rows = []
    for label, contents in [("head", heads), ("non-head", negatives)]:
        for index, content in enumerate(contents):
            group = f"{label}-{index:02d}"
            for style, description in STYLES.items():
                rows.append({"id": f"image-{len(rows):03d}", "label": label,
                             "group": group, "split": "fit" if index < 8 else "dev",
                             "style": style, "seed": 2026102800 + 100 * (label == "non-head") + index,
                             "prompt": f"{description} Show only {content}. Single centered subject occupying most of a square frame, plain light gray background, balanced soft lighting. No text, no labels, no collage, no additional people."})
    # Separate content families for held-out render queries in the token pilot.
    queries = [
        ("head", "a front view of a middle-aged person's head with freckles, cropped below the chin"),
        ("head", "a left profile of a person's head with a ponytail, cropped below the chin"),
        ("head", "a three-quarter view of a person's head with a beard, cropped below the chin"),
        ("head", "a right profile of a person's head with braided hair, cropped below the chin"),
        ("head", "a front view of an older bald person's head wearing glasses, cropped below the chin"),
        ("head", "a back view of a person's head with curly shoulder-length hair, cropped below the neck"),
        ("non-head", "a person's hand with fingers spread, cropped at the forearm, no head or face"),
        ("non-head", "a person's clenched fist seen from above, cropped at the forearm, no head or face"),
        ("non-head", "a person's two lower legs and feet viewed from behind, no head or face"),
        ("non-head", "a person's torso in a buttoned cardigan, cropped below the neck and above the waist, no head or face"),
        ("non-head", "a person's bent knee in side view, cropped at the thigh and calf, no head or face"),
        ("non-head", "a person's two hands with interlaced fingers, cropped at the wrists, no head or face"),
    ]
    for index, (label, content) in enumerate(queries):
        rows.append({"id": f"image-{len(rows):03d}", "label": label, "group": f"query-{index:02d}",
                     "split": "query", "style": "render", "seed": 2026102900 + index,
                     "prompt": f"{STYLES['render']} Show only {content}. Single centered subject occupying most of a square frame, plain light gray background, balanced soft lighting. No text, no labels, no collage, no additional people."})
    return rows


def generate(root, stage="generation"):
    import torch
    from vllm_omni import Omni
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from PIL import Image, ImageDraw, ImageStat

    if (root / stage / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    plan = read_json(root / stage / "plan.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
        raise ValueError("Expose only the selected GPU")
    if torch.cuda.mem_get_info()[0] < 60 * 2**30:
        raise RuntimeError("Generation requires at least 60 GiB free")
    output = root / stage
    write_json(output / "started.json", {"pid": os.getpid()})
    omni = None
    started = time.monotonic()
    try:
        omni = Omni(model=plan["model"], dtype="bfloat16", enforce_eager=True,
                    init_timeout=900, stage_init_timeout=900)
        rows = []
        for case in plan["cases"]:
            params = OmniDiffusionSamplingParams(width=1024, height=1024, seed=case["seed"],
                                                true_cfg_scale=1., num_outputs_per_prompt=1)
            images = extract_images_from_outputs(omni.generate(
                {"prompt": case["prompt"], "modalities": ["image"]}, sampling_params_list=[params]))
            if len(images) != 1 or images[0].size != (1024, 1024):
                raise RuntimeError("Unexpected generated image shape or count")
            if max(ImageStat.Stat(images[0].convert("RGB")).stddev) < 1:
                raise ValueError("Blank image; stop without replacements")
            path = output / f"{case['id']}.png"
            images[0].save(path)
            row = {**case, "file": str(path), "sha256": file_hash(path)}
            rows.append(row)
            write_json(output / f"{case['id']}.json", row)
            write_json(output / "progress.json", {"completed": len(rows), "total": len(plan["cases"]),
                       "seconds": time.monotonic() - started})
            print(f"Generated {len(rows)}/{len(plan['cases'])}", flush=True)
        for start in range(0, len(rows), 24):
            subset = rows[start:start + 24]
            sheet = Image.new("RGB", (1440, 6 * 280), "white")
            draw = ImageDraw.Draw(sheet)
            for i, row in enumerate(subset):
                x, y = (i % 4) * 360, (i // 4) * 280
                sheet.paste(Image.open(row["file"]).convert("RGB").resize((256, 256)), (x, y + 24))
                draw.text((x + 2, y + 3), f"{row['id']} {row['label']} {row['style']}", fill="black")
            sheet.save(output / f"inspection-{start // 24}.jpg", quality=90)
        omni.close()
        omni = None
        write_json(output / "complete.json", {"images": rows, "seconds": time.monotonic() - started,
                   "plan_sha256": file_hash(output / "plan.json"), "visual_review_pending": True})
    except BaseException:
        write_json(output / "failed.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        if omni is not None:
            omni.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", default="generation")
    args = parser.parse_args()
    generate(args.root, args.stage)


if __name__ == "__main__":
    main()
