import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from core.artifacts import (environment, file_hash, read_json, source_fingerprint,
                              write_json, write_tensors)
from core.config import digest, load_config


def child_environment():
    """Put the repository root on the child PYTHONPATH so `-m core.commands` resolves.

    The staged CLI is launched as a module rather than an installed console
    script, so a child started from another working directory would otherwise
    fail to import `core`.
    """
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    return env


def emit(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)


def configuration(args):
    return load_config(getattr(args, "config", None), getattr(args, "set", []))


def prepare_data(args):
    from core.data import prepare_coco, prepare_medical_vqa, synthetic
    if args.synthetic:
        manifest = synthetic(args.output, args.count, args.seed, args.clients)
    elif args.medical:
        manifest = prepare_medical_vqa(args.output, tuple(args.medical), args.seed, args.clients,
                                       args.limit, args.cache_dir)
    else:
        if not args.captions or not args.images:
            raise ValueError("Supply --captions and --images, --medical, or --synthetic")
        manifest = prepare_coco(args.captions, args.images, Path(args.output) / "samples.jsonl",
                                args.questions, args.annotations, args.seed, args.clients)
    emit({"manifest": str(manifest), "sha256": file_hash(manifest)})


def inject_canaries(args):
    from core.data import inject_canaries as inject
    manifest = inject(args.data, args.output, args.canary_seed, args.rate, args.field)
    emit({"manifest": str(manifest), "sha256": file_hash(manifest)})


def train(args):
    import numpy as np
    from core.data import load_batch, read_manifest
    from core.fl import fedavg, restore_model, save_model
    from core.vlm_wrapper import build_model
    cfg = configuration(args)
    output = Path(args.output)
    signature = digest({"config": asdict(cfg), "data_sha256": file_hash(args.data),
                        "source_sha256": source_fingerprint()})
    pointer = output / "training.json"
    start = 0
    history = []
    if pointer.exists():
        if not args.resume:
            raise FileExistsError("Training output exists; use --resume with the same configuration")
        saved = read_json(pointer)
        if saved["signature"] != signature:
            raise ValueError("Training resume configuration or data changed")
        adapter = restore_model(output / saved["checkpoint"], cfg.model.device)
        start, history = saved["round"], saved["history"]
    else:
        adapter = build_model(cfg.model, cfg.training)
    clients = {}
    for row in read_manifest(args.data, cfg.training.task, "train"):
        clients.setdefault(row["client"], []).append(row)
    if len(clients) < cfg.training.clients_per_round:
        raise ValueError("Too few nonempty training clients in the prepared manifest")
    if any(client >= cfg.training.clients for client in clients):
        raise ValueError("Prepared manifest has more clients than the training protocol")
    count = cfg.training.batch_size * cfg.training.local_steps
    if start == 0 and not pointer.exists():
        save_model(output / "round-0000", adapter, {"round": 0, "config_hash": signature})
        write_json(pointer, {"signature": signature, "round": 0, "checkpoint": "round-0000", "history": []})
    for round_id in range(start + 1, cfg.training.rounds + 1):
        rng = np.random.default_rng(cfg.training.seed + round_id)
        selected = rng.choice(sorted(clients), cfg.training.clients_per_round, replace=False)
        batches, weights = [], []
        for client in selected:
            rows = clients[client]
            indices = rng.choice(len(rows), count, replace=len(rows) < count)
            batches.append(load_batch([rows[i] for i in indices], adapter))
            weights.append(len(rows))
        fedavg(adapter, batches, weights)
        with torch.no_grad():
            loss = sum(adapter(b.images, b.questions, b.targets).item() for b in batches) / len(batches)
        history.append({"round": round_id, "mean_selected_client_loss": loss,
                        "client_count": len(selected)})
        # Commit every round to make interruption recovery exact; published snapshots are flagged.
        checkpoint = (f"round-{round_id:04d}" if round_id in cfg.training.snapshots
                      else f"latest-{round_id % 2}")
        save_model(output / checkpoint, adapter,
                   {"round": round_id, "config_hash": signature,
                    "published_snapshot": round_id in cfg.training.snapshots})
        write_json(pointer, {"signature": signature, "round": round_id,
                             "checkpoint": checkpoint, "history": history})
        emit(history[-1])
    write_json(output / "environment.json", environment())
    emit({"status": "completed", "rounds": cfg.training.rounds, "output": str(output)})


def capture(args):
    from core.data import load_batch, read_manifest
    from core.fl import capture as capture_update, restore_model, save_model, save_observation
    from core.vlm_wrapper import build_model
    cfg = configuration(args)
    output = Path(args.output)
    if args.client < 0 or args.client >= cfg.training.clients or args.offset < 0:
        raise ValueError("Invalid client index or negative sample offset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Capture output is nonempty; choose a new directory")
    if args.model:
        adapter = restore_model(args.model, cfg.model.device)
        if asdict(adapter.spec) != asdict(cfg.model):
            raise ValueError("Model checkpoint specification differs from capture configuration")
        old = adapter.training_spec
        if (old.mode, old.lora_rank, old.lora_alpha) != (
                cfg.training.mode, cfg.training.lora_rank, cfg.training.lora_alpha):
            raise ValueError("Checkpoint trainable parameterization differs from capture configuration")
        adapter.training_spec = cfg.training
    else:
        adapter = build_model(cfg.model, cfg.training)
    rows = read_manifest(args.data, cfg.training.task, args.split, args.client, unique_images=True)
    count = cfg.training.batch_size * cfg.training.local_steps
    rows = rows[args.offset:args.offset + count]
    if len(rows) != count:
        raise ValueError(f"Need {count} unique-image records for client={args.client}, split={args.split}; got {len(rows)}")
    batch = load_batch(rows, adapter)
    observation = capture_update(adapter, batch,
                                 adapter.decode(batch.questions), adapter.decode(batch.targets))
    defense_config = getattr(args, "defense_config", None)
    if defense_config is not None:
        from defenses.factory import create_defense
        observation.tensors = create_defense(defense_config).apply(observation.tensors)
    observation_id = save_observation(output / "public", observation)
    if defense_config is not None:
        write_json(output / "public" / "upload.json", {
            "defense": defense_config, "attack_policy": "defense_unaware_raw_update_matching"})
    if args.model:
        write_json(output / "public" / "model_ref.json", {"path": str(Path(args.model).resolve())})
    else:
        save_model(output / "public" / "model", adapter)
    write_tensors(output / "private" / "images.safetensors", {"images": batch.images})
    for row, q, y in zip(rows, adapter.decode(batch.questions), adapter.decode(batch.targets)):
        row["model_question"], row["model_target"] = q, y
    write_json(output / "private" / "truth.json", {"observation_id": observation_id,
                                                  "training": asdict(cfg.training), "samples": rows})
    write_json(output / "private" / "capture.json", {"config": asdict(cfg),
                                                     "data_sha256": file_hash(args.data),
                                                     "split": args.split, "client": args.client,
                                                     "offset": args.offset, "environment": environment()})
    emit({"status": "captured", "observation_id": observation_id, "output": str(output),
          "uploaded_parameters": len(observation.tensors),
          "communication_bytes": sum(t.numel() * t.element_size() for t in observation.tensors.values())})


def attack(args):
    from attacks.factory import create_attacker
    from core.knowledge import AdversaryKnowledge
    from core.fl import load_observation, restore_model
    cfg = configuration(args)
    public, output = Path(args.observation), Path(args.output)
    obs = load_observation(public, args.device or cfg.model.device)
    meta = read_json(public / "observation.json")
    signature = digest({"observation_id": meta["observation_id"], "attack": asdict(cfg.attack),
                        "source_sha256": source_fingerprint(),
                        "upload_sha256": file_hash(public / "upload.json")
                        if (public / "upload.json").exists() else None,
                        "wrong_observation": file_hash(Path(args.wrong_observation) / "update.safetensors")
                        if args.wrong_observation else None})
    result_file = output / "result.json"
    if result_file.exists():
        result = read_json(result_file)
        if args.resume and result.get("run_signature") == signature:
            emit({"status": "already_completed", "output": str(output)})
            return
        raise FileExistsError("Result already exists; choose another output or resume the exact run")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("Partial attack output exists; use --resume")
    model_path = args.model or (public / "model")
    if not args.model and (public / "model_ref.json").exists():
        model_path = read_json(public / "model_ref.json")["path"]
    adapter = restore_model(model_path, obs.model.device)
    adapter.training_spec = obs.training
    if args.wrong_observation:
        wrong = load_observation(args.wrong_observation, obs.model.device)
        if wrong.model_fingerprint != obs.model_fingerprint or asdict(wrong.training) != asdict(obs.training):
            raise ValueError("Wrong-gradient control requires an independent observation under the same protocol/state")
        if file_hash(Path(args.wrong_observation) / "update.safetensors") == meta["update_sha256"]:
            raise ValueError("Wrong-gradient control was given the original update")
        obs.tensors = wrong.tensors
    from core.types import Reconstruction
    try:
        result = create_attacker(adapter, cfg.attack).attack(
            obs.tensors, obs, AdversaryKnowledge(obs.training.knowledge),
            directory=output, resume=args.resume)
    except torch.OutOfMemoryError as error:
        if getattr(args, "raise_oom", False):
            raise
        result = Reconstruction("resource_unavailable", f"Out of memory: {error}")
    except (FileNotFoundError, ImportError, OSError) as error:
        result = Reconstruction("resource_unavailable", str(error))
    info = {k: v for k, v in asdict(result).items() if k != "images"}
    info.update({"observation_id": meta["observation_id"], "run_signature": signature,
                 "condition": {"model": obs.model.name, "revision": obs.model.revision,
                               "mode": obs.training.mode, "task": obs.training.task,
                               "observation": obs.training.observation,
                               "knowledge": obs.training.knowledge, "batch_size": obs.training.batch_size,
                               "local_steps": obs.training.local_steps, "lora_rank": obs.training.lora_rank,
                               "model_state": obs.model_fingerprint, "method": cfg.attack.method,
                               "source_sha256": source_fingerprint(),
                               "text_method": cfg.attack.text_method, "seed": cfg.attack.seed,
                               "model_protocol_hash": digest({k: v for k, v in asdict(obs.model).items()
                                                               if k not in {"device", "device_map", "max_memory"}}),
                               "local_lr": obs.training.lr, "lora_alpha": obs.training.lora_alpha,
                               "attack_protocol_hash": digest({k: v for k, v in asdict(cfg.attack).items() if k != "seed"}),
                               "control": "wrong_update" if args.wrong_observation else "none"},
                 "environment": environment()})
    if (public / "upload.json").exists():
        info["condition"].update(read_json(public / "upload.json"))
    if result.images is not None:
        write_tensors(output / "images.safetensors", {"images": result.images})
        from PIL import Image
        import numpy as np
        for index, image in enumerate(result.images):
            pixels = (image.permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(output / f"image-{index:03d}.png")
    write_json(result_file, info)
    emit({"status": result.status, "reason": result.reason, "costs": result.costs, "output": str(output)})


def evaluate(args):
    from evaluation.reconstruction import evaluate as run_evaluation
    cfg = configuration(args)
    report = run_evaluation(args.reconstruction, args.truth, cfg.evaluation, args.device)
    path = Path(args.output) if args.output else Path(args.reconstruction) / "evaluation.json"
    write_json(path, report)
    emit({"status": report["status"], "sample_count": len(report["samples"]), "output": str(path)})


def report(args):
    from evaluation.reconstruction import summarize
    paths = sorted(Path(args.input).rglob("evaluation.json"))
    if not paths:
        raise FileNotFoundError("No evaluation.json files found")
    records = [read_json(path) for path in paths]
    summary = summarize(records, args.bootstrap, args.seed)
    write_json(args.output, summary)
    jsonl = Path(args.output).with_suffix(".jsonl")
    jsonl.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    emit({"reports": len(paths), "groups": len(summary["groups"]), "output": args.output})


def doctor(args):
    from attacks import METHODS, supports
    cfg = configuration(args)
    result = {"environment": environment(), "config": asdict(cfg), "config_hash": digest(cfg),
              "methods": METHODS}
    if args.resolve_revision:
        from huggingface_hub import model_info
        result["resolved_revision"] = model_info(args.resolve_revision).sha
    if args.probe:
        from core.vlm_wrapper import build_model
        from core.fl import capture as capture_update, simulate_update
        adapter = build_model(cfg.model, cfg.training)
        n = cfg.training.batch_size * cfg.training.local_steps
        g = torch.Generator(device=adapter.device).manual_seed(314)
        images = torch.rand(n, 3, cfg.model.image_size, cfg.model.image_size,
                            generator=g, device=adapter.device, dtype=adapter.dtype)
        batch = adapter.batch(images, ["what color is the object"] * n, ["red"] * n)
        obs = capture_update(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
        replay = simulate_update(adapter, batch, cfg.training)
        error = max((replay[k] - obs.tensors[k]).abs().max().item() for k in replay)
        from attacks import Candidate, matching_loss
        candidate = Candidate(adapter, obs, 123)
        predicted = simulate_update(adapter, candidate.batch(), cfg.training, True)
        loss = matching_loss(predicted, obs.tensors, "l2")
        grads = torch.autograd.grad(loss, tuple(candidate.parameters()), allow_unused=True)
        norms = {name: None if grad is None else grad.norm().item()
                 for (name, _), grad in zip(candidate.named_parameters(), grads)}
        if error > 1e-5 or any(grad is None or not torch.isfinite(grad).all() for grad in grads):
            raise RuntimeError(f"Replay/second-order probe failed: error={error}, norms={norms}")
        result.update({"probe": "passed", "replay_max_error": error, "candidate_gradient_norms": norms,
                       "model": adapter.description(),
                       "capabilities": {method: asdict(supports(method, adapter, obs)) for method in METHODS},
                       "zero_update_parameters": [k for k, v in obs.tensors.items() if v.count_nonzero() == 0]})
    emit(result)
    if args.output:
        write_json(args.output, result)


def smoke(args):
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Smoke output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, "-m", "core.commands", "--threads", "1"]

    env = child_environment()

    def run(*command):
        subprocess.run(prefix + list(command), check=True, env=env)

    run("prepare-data", "--synthetic", "--count", "96", "--clients", "1", "--output", str(root / "data"))
    modes = ["full"] if args.quick else ["full", "lora_llm"]
    tasks = ["vqa"] if args.quick else ["vqa", "caption"]
    for mode in modes:
        for task in tasks:
            for observation in (["gradient"] if args.quick else ["gradient", "client_delta"]):
                folder = root / f"{mode}-{task}-{observation}"
                overrides = [f"training.mode={mode}", f"training.task={task}",
                             f"training.observation={observation}", "training.clients=1",
                             "training.clients_per_round=1",
                             f"training.local_steps={2 if observation == 'client_delta' else 1}",
                             "attack.iterations=12", "attack.checkpoint_interval=4",
                             "attack.max_evaluations=50"]
                opts = [item for setting in overrides for item in ["--set", setting]]
                run("capture", "--data", str(root / "data" / "samples.jsonl"), "--client", "0",
                    "--output", str(folder / "capture"), *opts)
                run("attack", "--observation", str(folder / "capture" / "public"),
                    "--output", str(folder / "attack"), *opts)
                run("evaluate", "--reconstruction", str(folder / "attack"),
                    "--truth", str(folder / "capture" / "private"))
    run("report", "--input", str(root), "--output", str(root / "report.json"), "--bootstrap", "100")
    emit({"status": "smoke_completed", "output": str(root)})


def utility(args):
    from core.data import read_manifest
    from core.fl import restore_model
    from evaluation.utility import evaluate_utility
    adapter = restore_model(args.model, args.device)
    rows = read_manifest(args.data, adapter.training_spec.task, args.split, unique_images=True)[:args.limit]
    if not rows:
        raise ValueError("No held-out utility samples")
    result = evaluate_utility(adapter, rows, args.batch_size)
    result.update({"model_fingerprint": adapter.fingerprint(), "split": args.split,
                   "data_sha256": file_hash(args.data), "samples": len(rows),
                   "protocol": adapter.protocol_version})
    write_json(args.output, result)
    emit({"samples": len(rows), "metrics": result["metrics"], "output": args.output})


def suite(args):
    from utils.suite import materialize
    emit(materialize(args))


def run_suite(args):
    from utils.suite import execute
    execute(args)


def build_parser():
    parser = argparse.ArgumentParser(prog="giavlm", description="Federated VLM gradient leakage benchmark")
    parser.add_argument("--threads", type=int, default=1)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, func, configured=False):
        p = commands.add_parser(name)
        p.set_defaults(func=func)
        if configured:
            p.add_argument("--config")
            p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        return p

    from core.data import MEDICAL_VQA_SOURCES

    p = command("prepare-data", prepare_data)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--medical", nargs="+", choices=sorted(MEDICAL_VQA_SOURCES),
                   help="Normalize the named medical VQA corpora into one manifest")
    p.add_argument("--limit", type=int, help="Rows per upstream split; for quick CPU checks")
    p.add_argument("--cache-dir", help="Dataset cache directory; keep it off the NAS mount")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clients", type=int, default=10)
    for name in ["captions", "images", "questions", "annotations"]:
        p.add_argument("--" + name)
    p.add_argument("--output", required=True)
    p = command("inject-canaries", inject_canaries)
    p.add_argument("--data", required=True, help="Prepared manifest to rewrite")
    p.add_argument("--output", required=True)
    p.add_argument("--field", default="question", choices=["question", "target"])
    p.add_argument("--rate", type=float, default=1.0)
    p.add_argument("--canary-seed", type=int, default=42,
                   help="Entity generation only; partitioning is inherited from --data")

    p = command("train", train, True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p = command("capture", capture, True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    p.add_argument("--client", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--split", choices=["tune", "eval", "train"], default="eval")
    p = command("attack", attack, True)
    p.add_argument("--observation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    p.add_argument("--device")
    p.add_argument("--wrong-observation")
    p.add_argument("--resume", action="store_true")
    p = command("evaluate", evaluate, True)
    p.add_argument("--reconstruction", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output")
    p.add_argument("--device", default="cpu")
    p = command("report", report)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p = command("doctor", doctor, True)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--resolve-revision", metavar="HF_MODEL_ID")
    p.add_argument("--output")
    p = command("smoke", smoke)
    p.add_argument("--output", required=True)
    p.add_argument("--quick", action="store_true")
    p = command("utility", utility)
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", choices=["tune", "eval"], default="eval")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p = command("suite", suite)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--split", choices=["tune", "eval"], default="eval")
    p.add_argument("--tasks", nargs="+", choices=["vqa", "caption"], default=["vqa", "caption"])
    p.add_argument("--modes", nargs="+", choices=["full", "llm_full", "lora_llm"], default=["full", "lora_llm"])
    p.add_argument("--knowledge", nargs="+", choices=["private", "question_known", "text_known"],
                   default=["private", "text_known"])
    p.add_argument("--methods", nargs="+", default=["dlg_adapted", "ig_adapted", "april_adapted",
                                                   "gradvit_adapted", "gi_dqa_adapted"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--observation", choices=["gradient", "client_delta"], default="gradient")
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--pilot-seconds", type=float)
    p.add_argument("--gpus-per-run", type=int, default=1)
    p = command("run-suite", run_suite)
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"giavlm: {error}\n")


if __name__ == "__main__":
    main()
