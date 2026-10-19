"""Three frozen photographic body-part banks, without target/domain inputs."""

from pathlib import Path
import sys

from evaluation import synthetic_head_generation as harness


def cases():
    common = (
        "普通彩色摄影，浅灰色纯色背景，柔和均匀照明，主体居中，占画面约七成。"
        "单人局部特写，真实皮肤纹理，无文字，无标记，无拼图，无装饰。"
    )
    groups = [
        ("head", "front", "成年人的完整头部正面特写，从头顶到下巴，面容自然。"),
        ("head", "side", "成年人的完整头部侧面特写，从头顶到下巴，清晰侧脸轮廓。"),
        ("head", "back", "成年人的完整头部后面特写，从后方看后脑勺和头发。"),
        ("non-head", "torso", "成年男性胸部到腹部的躯干局部特写，画面严格裁切在肩膀以下、腰部以上，不出现头部和脸。"),
        ("non-head", "arm", "成年人的一条手臂局部特写，从上臂到手掌，画面只有手臂和手，不出现头部和脸。"),
        ("non-head", "legs", "成年人的小腿和脚的局部特写，从膝盖到脚部，画面只有腿和脚，不出现头部和脸。"),
    ]
    people = ["肤色较浅。", "肤色较深。", "肤色中等。", "肤色较浅。"]
    return [
        {"case": f"bank-{bank}_{label}_{region}", "bank": bank,
         "property_label": label, "region": region,
         "seed": 202610180 + bank * 100 + j,
         "prompt": prompt + people[j] + common}
        for bank in range(3) for label, region, prompt in groups for j in range(4)
    ]


def main():
    # Reuse the existing loading/checkpoint/failure harness with public cases only.
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    harness.cases = cases
    harness.main()
    harness.save_json(output / "case_provider.json", {
        "source": str(Path(__file__)), "source_sha256": harness.sha256(Path(__file__)),
        "uses_target_images_or_text": False, "uses_victim_gradients": False,
        "domain_prior": "None in prompts; human head attribute specified",
        "banks": 3, "candidates_per_class_per_bank": 12,
    })


if __name__ == "__main__":
    main()
