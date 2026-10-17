"""Generate a fixed, target-independent head/abdomen CT prototype bank."""

import argparse
from pathlib import Path

from evaluation import qwen_image_generation_smoke as smoke


def ct_cases():
    common = (
        "A single grayscale axial CT scan slice of the {region}, {window}. "
        "Realistic diagnostic radiology image, transverse cross section, "
        "body centered against a black background. "
        "No text, no labels, no arrows, no collage, no color."
    )
    return [
        {"case": label, "seed": seed,
         "prompt": common.format(region=region, window=window)}
        for label, region, window in [
            ("head", "head showing the brain", "brain soft tissue window"),
            ("abdomen", "upper abdomen", "soft tissue window"),
        ]
        for seed in [202610160, 202610161, 202610162, 202610163, 202610164]
    ]


def main():
    # The smoke harness owns loading, failure handling and image checks. Replacing
    # its public case provider changes only the predeclared prompts and seeds.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    smoke.cases = ct_cases
    smoke.main()
    smoke.save_json(args.output / "case_provider.json", {
        "source": "evaluation/synthetic_ct_generation.py",
        "source_sha256": smoke.sha256(Path(__file__)),
        "uses_target_images_or_text": False,
        "classes": ["head", "abdomen"], "candidates_per_class": 5,
    })


if __name__ == "__main__":
    main()
