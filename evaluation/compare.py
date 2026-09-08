"""Paired metric deltas against a prior-only control; raw sign is preserved."""
import argparse
from core.artifacts import read_json, write_json


def prior_only_deltas(attacked, baseline):
    if baseline["condition"]["method"] != "prior_only":
        raise ValueError("Baseline must be the prior_only control")
    for report in [attacked, baseline]:
        if report["status"] != "completed":
            raise ValueError("Both runs must complete before computing metric deltas")
    for key in ["observation_id", "evaluation_config"]:
        if attacked[key] != baseline[key]:
            raise ValueError(f"Control comparison differs in {key}")
    if attacked["condition"]["seed"] != baseline["condition"]["seed"]:
        raise ValueError("Pair the same candidate initialization seed")
    other = {row["sample_id"]: row for row in baseline["samples"]}
    if len(other) != len(baseline["samples"]) or set(other) != {r["sample_id"] for r in attacked["samples"]}:
        raise ValueError("Control sample sets differ or contain duplicates")
    rows = []
    for row in attacked["samples"]:
        metrics = {}
        for name, value in row["metrics"].items():
            previous = other[row["sample_id"]]["metrics"].get(name)
            if type(value) in {float, int} and type(previous) in {float, int}:
                metrics[name] = value - previous
        rows.append({"sample_id": row["sample_id"], "delta": metrics})
    return {"definition": "attack minus prior_only; direction depends on metric",
            "observation_id": attacked["observation_id"], "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacked", required=True)
    parser.add_argument("--prior-only", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, prior_only_deltas(read_json(args.attacked), read_json(args.prior_only)))


if __name__ == "__main__":
    main()
