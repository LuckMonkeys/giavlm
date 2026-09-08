"""Explicit YAML jobs, argv-only execution, no shell or GPU occupancy processes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf


def materialize_jobs(path, output_root=None):
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if set(config) - {"output_root", "jobs"}:
        raise ValueError("Schedule supports only output_root and jobs")
    root = Path(output_root or config.get("output_root", "../outputs/batch"))
    root = root if root.is_absolute() else path.parent / root
    jobs, names = [], set()
    for job in config["jobs"]:
        if set(job) - {"name", "config_name", "overrides"}:
            raise ValueError("Unknown job field")
        name = job["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("Job names must be unique, path-safe identifiers")
        names.add(name)
        overrides = job.get("overrides", [])
        if not isinstance(overrides, list) or not all(isinstance(s, str) for s in overrides):
            raise ValueError("overrides must be a list of Hydra argument strings")
        if any(s.startswith(("output_dir=", "hydra.run.dir=", "--")) for s in overrides):
            raise ValueError("Output routing and CLI options belong to the scheduler")
        output = (root / name).resolve()
        argv = [sys.executable, "-m", "examples.run_attack", "--config-name",
                job.get("config_name", "config"), *overrides,
                f"output_dir='{output}'", f"hydra.run.dir='{output / 'hydra'}'"]
        jobs.append({"name": name, "argv": argv})
    return jobs


def available_gpu(gpu_ids, min_free_mib):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    memory = {index.strip(): int(free.strip()) for index, free in
              (line.split(",") for line in output.strip().splitlines())}
    for device in gpu_ids:
        if memory.get(device, -1) >= min_free_mib:
            return device
    raise RuntimeError("No requested GPU meets the memory threshold; no jobs were launched")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd-config-yaml", "--cmd_config_yaml", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-ids", help="Explicit allowed physical GPU indices, comma separated")
    parser.add_argument("--min-free-mib", type=int, default=10000)
    args = parser.parse_args(argv)
    jobs = materialize_jobs(args.cmd_config_yaml, args.output_root)
    print(json.dumps(jobs, indent=2))
    if args.execute:
        for job in jobs:
            env = os.environ.copy()
            if args.gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = available_gpu(args.gpu_ids.split(","), args.min_free_mib)
            # Sequential execution bounds GPU ownership; stop on the first failed experiment.
            subprocess.run(job["argv"], env=env, check=True)


if __name__ == "__main__":
    main()
