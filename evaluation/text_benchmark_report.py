"""Export aggregate tables and figures without private examples or sample IDs."""
import argparse
import csv
import io
from pathlib import Path

from core.artifacts import file_hash, read_json, source_fingerprint, write_json


MAIN_COLUMNS = ("model", "dataset", "image", "n", "completed", "joint_em", "rouge1", "rouge2", "rougeL", "word_recall")
FILTER_COLUMNS = ("model", "dataset", "image", "length_bin", "n", "recall_at_100", "average_precision", "full_coverage")


def write_table(directory, name, rows, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (directory / f"{name}.csv").write_text(stream.getvalue())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("NA" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (directory / f"{name}.md").write_text("\n".join(lines) + "\n")


def figures(root, destination, aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    datasets = ("slake", "vqa_rad", "vqav2")
    models = ("llava", "qwen35")
    colors = {"gt": "#176A86", "random": "#D16B40"}
    bins = ("1-16", "17-32", "33-64")
    lookup = {(r["model"], r["dataset"], r["image"], r["length_bin"]): r for r in aggregate}
    for metric, label, filename in (("joint_em", "Joint Exact Match", "joint-em-by-length"),
                                     ("recall_at_100", "Token Recall@100", "token-recall-by-length")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
        for mi, model in enumerate(models):
            for di, dataset in enumerate(datasets):
                ax = axes[mi, di]
                for j, image in enumerate(("gt", "random")):
                    xs, ys, counts = [], [], []
                    for index, length_bin in enumerate(bins):
                        row = lookup.get((model, dataset, image, length_bin))
                        if row is not None and row.get(metric) is not None:
                            xs.append(index + (j - .5) * .34)
                            ys.append(row[metric])
                            counts.append(row["n"])
                    bars = ax.bar(xs, ys, width=.32, color=colors[image], label="GT" if image == "gt" else "Random")
                    for bar, count in zip(bars, counts, strict=True):
                        ax.annotate(f"n={count}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
                ax.set(title=f"{model} · {dataset}", xticks=range(3), xticklabels=bins, ylim=(0, 1.12))
                ax.set_xlabel("Q+A reference token length")
                if di == 0:
                    ax.set_ylabel(label)
                if not any((model, dataset, image, length_bin) in lookup for image in colors for length_bin in bins):
                    ax.text(.5, .5, "Pending", transform=ax.transAxes, ha="center", color="#666666")
                ax.yaxis.grid(True, alpha=.18)
                ax.set_axisbelow(True)
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(destination / f"{filename}.png", dpi=160)
        fig.savefig(destination / f"{filename}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    training_rows = []
    palette = ("#176A86", "#D16B40", "#6B5895")
    for ax, model in zip(axes, models, strict=True):
        for dataset, color in zip(datasets, palette, strict=True):
            path = root / "models" / dataset / model / "training.json"
            if not path.exists():
                continue
            history = read_json(path)["history"]
            ax.plot([r["round"] for r in history], [r["mean_selected_client_loss"] for r in history],
                    marker="o", markersize=3, color=color, label=dataset)
            training_rows.extend({"model": model, "dataset": dataset, "round": row["round"],
                                  "mean_selected_client_loss": row["mean_selected_client_loss"]} for row in history)
        ax.set(title=model, xlabel="Federated round", ylabel="Selected-client loss", xticks=range(1, 11))
        ax.legend(frameon=False)
        ax.yaxis.grid(True, alpha=.18)
    fig.tight_layout()
    fig.savefig(destination / "training-loss.png", dpi=160)
    fig.savefig(destination / "training-loss.svg")
    plt.close(fig)
    write_json(destination / "training-loss.json", training_rows)

    audit = read_json(root / "setup/selection-audit.json")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, dataset in zip(axes, datasets, strict=True):
        for j, model in enumerate(models):
            rows = [next(r for r in audit if r["dataset"] == dataset and r["model"] == model
                         and r["length_bin"] == length_bin) for length_bin in bins]
            means = np.array([r["question_mean"] + r["answer_mean"] for r in rows])
            errors = np.array([[mean - r["total_min"] for mean, r in zip(means, rows, strict=True)],
                               [r["total_max"] - mean for mean, r in zip(means, rows, strict=True)]])
            ax.errorbar(np.arange(3) + (j - .5) * .12, means, yerr=errors,
                        marker="o", capsize=3, color=palette[j], label=model)
        counts = [next(r["n"] for r in audit if r["dataset"] == dataset and r["length_bin"] == value)
                  for value in bins]
        ax.set(title=dataset, xticks=range(3),
               xticklabels=[f"{value}\n(n={n})" for value, n in zip(bins, counts, strict=True)],
               xlabel="Reference length bin")
        ax.yaxis.grid(True, alpha=.18)
    axes[0].set_ylabel("Native Q+A token count (mean and range)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "native-token-lengths.png", dpi=160)
    fig.savefig(destination / "native-token-lengths.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = sorted((args.root / "attacks").glob("*/*/summary.json"))
    aggregate = [row for path in summaries for row in read_json(path)["aggregate"]]
    write_json(args.output / "aggregate.json", {
        "rows": aggregate, "sources": [{"path": str(p.relative_to(args.root)), "sha256": file_hash(p)}
                                       for p in summaries],
        "contains_private_examples": False,
        "producer_source_sha256": source_fingerprint(),
        "metric_policy": "Full-QA ROUGE; exact Question and Answer required for Joint EM; "
                         "budget exhaustion is zero recovery; see per-metric denominator fields",
        "planned_attacks": 288, "completed_combinations": len(summaries), "planned_combinations": 6})
    main_rows = [r for r in aggregate if r["length_bin"] == "All"]
    length_rows = [r for r in aggregate if r["length_bin"] != "All"]
    write_table(args.output, "main", main_rows, MAIN_COLUMNS)
    write_table(args.output, "by-length", length_rows, MAIN_COLUMNS[:3] + ("length_bin",) + MAIN_COLUMNS[3:])
    write_table(args.output, "token-filter", length_rows, FILTER_COLUMNS)
    figures(args.root, args.output, aggregate)
    print(f"Exported {len(summaries)}/6 completed model/dataset combinations to {args.output}")


if __name__ == "__main__":
    main()
