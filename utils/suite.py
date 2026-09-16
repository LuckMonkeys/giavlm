"""Materialize experiments before spending compute; execute reviewed argv lists."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

from core.artifacts import file_hash, read_json, write_json
from core.commands import child_environment
from core.config import digest, load_config, validate
from core.data import read_manifest


def materialize(args):
    if args.samples <= 0 or args.batch_size <= 0 or args.local_steps <= 0 or args.gpus_per_run <= 0:
        raise ValueError("Sample, batch, step and GPU counts must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Suite output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    data = str(Path(args.data).resolve())
    data_hash = file_hash(data)
    jobs = []
    for path in args.configs:
        base = load_config(path, args.set)
        for task in args.tasks:
            rows = read_manifest(data, task, args.split, unique_images=True)
            clients = {}
            for row in rows:
                clients.setdefault(row["client"], []).append(row)
            for algorithm in args.algorithms:
                local_steps = 1 if algorithm == "fedsgd" else args.local_steps
                slots = args.batch_size * local_steps
                batches = [(client, offset) for client in sorted(clients)
                           for offset in range(0, len(clients[client]) - slots + 1, slots)]
                required = (args.samples + slots - 1) // slots
                if len(batches) < required:
                    raise ValueError(
                        f"Need {required} complete within-client batches for {task}/{algorithm}; "
                        f"found {len(batches)}")
                # Stable interleaving avoids placing the entire subset on the first client.
                batches.sort(key=lambda pair: (pair[1], pair[0]))
                for mode in args.modes:
                    for knowledge in args.knowledge:
                        if task == "caption" and knowledge == "question_known":
                            continue
                        cfg = replace(
                            base,
                            model=replace(base.model, target_length=64)
                            if task == "caption" and base.model.family != "tiny"
                            else replace(base.model),
                            training=replace(base.training, task=task, mode=mode,
                                             knowledge=knowledge, algorithm=algorithm,
                                             batch_size=args.batch_size,
                                             local_steps=local_steps))
                        validate(cfg)
                        model_cfg = replace(
                            cfg, training=replace(cfg.training, knowledge="private", rounds=0,
                                                  snapshots=[0]))
                        model_key = digest({"model": asdict(model_cfg.model),
                                            "training": asdict(model_cfg.training)})[:20]
                        model_dir = output / "models" / model_key
                        model_config_path = output / "configs" / f"model-{model_key}.json"
                        write_json(model_config_path, asdict(model_cfg))
                        setup_command = ["train", "--config", str(model_config_path), "--data", data,
                                         "--output", str(model_dir), "--resume"]
                        for client, offset in batches[:required]:
                            capture_key = digest({"model": asdict(cfg.model),
                                                  "training": asdict(cfg.training),
                                                  "data": data_hash, "split": args.split,
                                                  "client": client, "offset": offset})[:20]
                            capture_dir = output / "captures" / capture_key
                            capture_cfg = output / "configs" / f"capture-{capture_key}.json"
                            write_json(capture_cfg, asdict(cfg))
                            capture_command = ["capture", "--config", str(capture_cfg), "--data", data,
                                               "--client", str(client), "--offset", str(offset),
                                               "--split", args.split, "--output", str(capture_dir),
                                               "--model", str(model_dir / "round-0000")]
                            for method in args.methods:
                                for seed in args.seeds:
                                    experiment = replace(
                                        cfg, attack=replace(cfg.attack, method=method, seed=seed))
                                    key = digest({"capture": capture_key,
                                                  "attack": asdict(experiment.attack)})[:20]
                                    config_path = output / "configs" / f"attack-{key}.json"
                                    write_json(config_path, asdict(experiment))
                                    reconstruction = output / "attacks" / key
                                    jobs.append({"id": key, "capture_key": capture_key,
                                                 "model_key": model_key, "model_setup": setup_command,
                                                 "model_config_hash": file_hash(model_config_path),
                                                 "model_dir": str(model_dir),
                                                 "capture_config_hash": file_hash(capture_cfg),
                                                 "attack_config_hash": file_hash(config_path),
                                                 "capture": capture_command,
                                                 "attack": ["attack", "--config", str(config_path),
                                                            "--observation", str(capture_dir / "public"),
                                                            "--output", str(reconstruction), "--resume"],
                                                 "evaluate": ["evaluate", "--config", str(config_path),
                                                              "--reconstruction", str(reconstruction),
                                                              "--truth", str(capture_dir / "private")],
                                                 "capture_dir": str(capture_dir),
                                                 "reconstruction_dir": str(reconstruction),
                                                 "max_attack_seconds": experiment.attack.seconds})
    manifest = output / "jobs.jsonl"
    manifest.write_text("".join(json.dumps(job, sort_keys=True) + "\n" for job in jobs))
    budget = {"jobs": len(jobs), "captures": len({j["capture_key"] for j in jobs}),
              "shared_models": len({j["model_key"] for j in jobs}),
              "samples_requested_per_condition": args.samples,
              "samples_actual_per_condition": {
                  algorithm: ((args.samples + args.batch_size
                               * (1 if algorithm == "fedsgd" else args.local_steps) - 1)
                              // (args.batch_size
                                  * (1 if algorithm == "fedsgd" else args.local_steps)))
                  * args.batch_size * (1 if algorithm == "fedsgd" else args.local_steps)
                  for algorithm in args.algorithms},
              "manifest_sha256": file_hash(manifest), "data_sha256": data_hash,
              "attack_wall_hours_cap": sum(j["max_attack_seconds"] for j in jobs) / 3600,
              "gpus_per_run": args.gpus_per_run,
              "gpu_hours_cap": sum(j["max_attack_seconds"] for j in jobs) / 3600 * args.gpus_per_run,
              "pilot_gpu_hours_estimate": len(jobs) * args.pilot_seconds / 3600 * args.gpus_per_run
              if args.pilot_seconds is not None else None,
              "note": "Model loading, capture, checkpoint I/O and evaluation are additional. No jobs executed."}
    write_json(output / "budget.json", budget)
    return budget


def execute(args):
    path = Path(args.manifest)
    budget = read_json(path.parent / "budget.json")
    if file_hash(path) != budget["manifest_sha256"]:
        raise ValueError("Suite manifest changed after materialization")
    jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    for job in jobs[:args.limit]:
        for stage in ["model_setup", "capture", "attack"]:
            config = job[stage][job[stage].index("--config") + 1]
            field = "model_config_hash" if stage == "model_setup" else f"{stage}_config_hash"
            if file_hash(config) != job[field]:
                raise ValueError("A suite configuration changed after materialization")
        data_path = job["capture"][job["capture"].index("--data") + 1]
        if file_hash(data_path) != budget["data_sha256"]:
            raise ValueError("Dataset manifest changed after suite materialization")
        capture = Path(job["capture_dir"])
        if not (capture / "public" / "observation.json").exists():
            commands = [job["capture"], job["attack"], job["evaluate"]]
        else:
            from core.fl import load_observation
            load_observation(capture / "public", "cpu")
            private = read_json(capture / "private" / "capture.json")
            config = load_config(job["capture"][job["capture"].index("--config") + 1])
            if private["data_sha256"] != budget["data_sha256"] or private["config"] != asdict(config):
                raise ValueError("Existing capture does not match the suite")
            commands = [job["attack"], job["evaluate"]]
        if not (Path(job["model_dir"]) / "training.json").exists():
            commands.insert(0, job["model_setup"])
        log_dir = path.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        with (log_dir / f"{job['id']}.log").open("a") as log:
            for command in commands:
                # No shell expansion. Training data paths never appear in the attack argv.
                subprocess.run([sys.executable, "-m", "core.commands", "--threads", str(args.threads),
                                *command], check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=child_environment())
        print(json.dumps({"job": job["id"], "status": "finished"}), flush=True)
