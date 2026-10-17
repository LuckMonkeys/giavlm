"""Pre-scoring replacement of four visually rejected non-head MRI prototypes."""

from evaluation import synthetic_head_generation as generation


def cases():
    prompt = (
        "腹部磁共振成像，腹部轴位横断面，单张灰度医学影像。"
        "画面显示人体躯干的宽椭圆形横切面，肝脏占据右上腹，脾脏位于左上腹，"
        "双侧肾脏位于后腹部，中央可见椎体，周围是腹壁肌肉和皮下脂肪，"
        "可见部分肠管。典型腹部 MRI 影像，黑色背景，无文字，无标注，无拼图。"
    )
    return [{"case": "non-head_mri_abdomen_repair", "property_label": "non-head",
             "modality": "MRI", "region": "abdomen", "seed": 202610170 + j, "prompt": prompt}
            for j in range(4)]


if __name__ == "__main__":
    generation.cases = cases
    generation.main()
