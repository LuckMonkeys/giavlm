"""Aggregate-only tables and figures for the paired federated text benchmark."""
import argparse
import csv
from pathlib import Path

from core.artifacts import file_hash, read_json, write_json


LABELS = {
    "fedsgd-b1": "FedSGD\nB=1", "fedsgd-b2": "FedSGD\nB=2",
    "fedavg-b2-s1": "FedAvg\nB=2, S=1", "fedavg-b1-s2": "FedAvg\nB=1, S=2",
    "clipping": "Global\nclipping", "noise": "Gaussian\nnoise", "sparsify": "Top-k\n10%",
    "ideal-fp64": "Ideal delta\nFP64 scale", "sgd-bf16": "Standard SGD\nbf16 parameters",
    "sgd-fp32": "Standard SGD\nFP32 parameters",
    "fp32-gradient": "FP32 gradient\nB=2",
    "fp32-one-step": "FP32 SGD\nB=2, S=1",
    "fp32-two-steps": "FP32 SGD\nB=1, S=2",
    "mixed-gradient": "Mixed FedSGD\nB=2",
    "mixed-one-step": "Mixed SGD\nB=2, S=1",
    "mixed-two-steps": "Mixed SGD\nB=1, S=2",
    "adamw-gradient-control": "Gradient control\nFP32 master, B=2",
    "adamw-fp32-s1": "AdamW, FP32 master\nB=2, S=1",
    "adamw-fp32-s2": "AdamW, FP32 master\nB=1, S=2",
    "adamw-bf16-s1": "AdamW, bf16 weights\nB=2, S=1",
    "adamw-bf16-s2": "AdamW, bf16 weights\nB=1, S=2",
    "fixed4-b1-s4": "Mixed SGD\nN=4, B=1, S=4",
    "fixed4-b2-s2": "Mixed SGD\nN=4, B=2, S=2",
    "fixed4-b4-s1": "Mixed SGD\nN=4, B=4, S=1",
}


def table(rows):
    lines = ["| Condition | Image | Length | N | Joint EM | Q EM | A EM | R-1 | R-2 | R-L | Candidate recall | FullCoverage |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def count(key):
            return "—" if row[key] is None else f"{round(row[key] * row['n'])}/{row['n']}"
        def score(key):
            return "—" if row[key] is None else f"{row[key]:.3f}"
        values = [row["condition"], row["image"], row.get("length_bin", "All"), str(row["n"]),
                  count("joint_em"), count("question_em"), count("answer_em"), score("rouge1"),
                  score("rouge2"), score("rougeL"), score("candidate_recall"), count("full_coverage")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summary = read_json(output / "summary.json")
    plan = read_json(output / "plan.json")
    if summary["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Evaluation and capture source fingerprints differ")
    rows, by_length = summary["aggregate"], summary.get("by_length", [])
    directory = output / "report"
    directory.mkdir(exist_ok=True)
    for name, values in (("summary", rows), ("by-length", by_length)):
        if not values:
            continue
        with (directory / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        (directory / f"{name}.md").write_text(table(values))
    order = list(dict.fromkeys(row["condition"] for row in rows))

    def draw(ax, values, metric, title):
        for index, (image, color) in enumerate((("gt", "#27628C"), ("random", "#D68B36"))):
            selected = [next(row for row in values if row["condition"] == c and row["image"] == image)
                        for c in order]
            heights = [row[metric] if row[metric] is not None else 0 for row in selected]
            bars = ax.bar(np.arange(len(order)) + (index - .5) * .36, heights, width=.34, color=color,
                          label="Ground-truth image" if image == "gt" else "Random image")
            for bar, row in zip(bars, selected, strict=True):
                label = "NA" if row[metric] is None else f"{round(row[metric] * row['n'])}/{row['n']}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .02,
                        label, ha="center", fontsize=8)
        ax.set_xticks(range(len(order)), [LABELS[c] for c in order], fontsize=9)
        ax.set_ylim(0, 1.16)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Fraction of QA samples")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), layout="constrained")
    draw(axes[0], rows, "full_coverage", "All true tokens in candidate set")
    draw(axes[1], rows, "joint_em", "Exact recovery of question and answer")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"SLAKE / LLaVA-1.5-7B: rank 32, Round 10, LM prior 0.01; same {summary['cohort_n']} QA")
    fig.savefig(directory / "recovery.png", dpi=150)
    fig.savefig(directory / "recovery.svg")
    plt.close(fig)
    if by_length:
        bins = [b for b in ("1-16", "17-32", "33-64") if any(r["length_bin"] == b for r in by_length)]
        fig, axes = plt.subplots(len(bins), 1, figsize=(13, 4 * len(bins)), layout="constrained", squeeze=False)
        for ax, length_bin in zip(axes[:, 0], bins, strict=True):
            draw(ax, [r for r in by_length if r["length_bin"] == length_bin], "joint_em",
                 f"Question + answer length: {length_bin} content tokens")
        axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
        fig.suptitle("Joint exact match by the original frozen LLaVA length bins")
        fig.savefig(directory / "joint-em-by-length.png", dpi=150)
        fig.savefig(directory / "joint-em-by-length.svg")
        plt.close(fig)
    safe = {key: summary[key] for key in ("schema_version", "cohort_n", "attack_count", "source_sha256", "note")}
    safe.update(aggregate=rows, by_length=by_length, uncommitted=True,
                summary_sha256=file_hash(output / "summary.json"),
                metric_note="relative_l2_change measures upload defense perturbation, not FedAvg rounding error")
    write_json(directory / "aggregate.json", safe)
    write_json(output / "report-complete.json", {
        "status": "completed", "cohort_n": summary["cohort_n"], "attack_count": summary["attack_count"],
        "files": [{"path": str(path.relative_to(output)), "sha256": file_hash(path)}
                  for path in sorted(directory.iterdir()) if path.is_file()]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output)


if __name__ == "__main__":
    main()
