"""Canonical Hydra lifecycle, with immutable protocol and run-ID based recovery."""
from dataclasses import asdict
import gc
from pathlib import Path
import shutil
from types import SimpleNamespace

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch

from core import commands
from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import Config, digest, validate
from core.fl import NamedGradientAccumulator, canonicalize_lora_parameter_name  # noqa: F401
from core.knowledge import AdversaryKnowledge
from defenses.factory import create_defense


def protocol_config(config: DictConfig) -> Config:
    """Translate Hydra groups to the strict, serialized wire protocol."""
    value = OmegaConf.to_container(config, resolve=True)
    knowledge = AdversaryKnowledge(**value["knowledge"]).validate()
    model = dict(value["model"])
    model["name"] = model.pop("checkpoint")
    fed = dict(value["fed"])
    fed.pop("name")
    if fed.pop("secure_aggregation"):
        raise NotImplementedError("Aggregate inversion is not an individual-client observation")
    if fed.pop("upload_parameters"):
        raise NotImplementedError("Upload masks are not yet integrated into attack replay")
    fed.update(task=value["data"]["task"], knowledge=knowledge.name)
    attack = dict(value["attack"])
    attack["method"] = attack.pop("name")
    evaluation = dict(value["evaluation"])
    evaluation.pop("name")
    create_defense(value["defense"])
    merged = OmegaConf.merge(OmegaConf.structured(Config), {
        "model": model, "training": fed, "attack": attack, "evaluation": evaluation})
    return validate(OmegaConf.to_object(merged))


class ExperimentRunner:
    def __init__(self, config: DictConfig):
        self.config = config
        self.protocol = protocol_config(config)
        if not 0 <= config.start_run_id < config.num_runs:
            raise ValueError("Require 0 <= start_run_id < num_runs (exclusive stop run ID)")
        if config.oom_recovery.max_retries < 0:
            raise ValueError("oom_recovery.max_retries must be nonnegative")
        if config.model_snapshot and self.protocol.training.rounds:
            raise ValueError("Choose model_snapshot or fed.rounds, not both")
        if config.output_dir:
            self.output = Path(to_absolute_path(config.output_dir))
        elif HydraConfig.initialized():
            self.output = Path(HydraConfig.get().runtime.output_dir)
        else:
            raise ValueError("Set output_dir when calling run_experiment without Hydra")
        self.output.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def _prepare(self):
        from core.data import synthetic
        data = self.config.data
        if data.name == "synthetic":
            if self.protocol.model.family != "tiny":
                raise ValueError("Synthetic fixture is restricted to the tiny model")
            manifest = self.output / "data" / "samples.jsonl"
            if not manifest.exists():
                synthetic(self.output / "data", data.count, data.seed, self.protocol.training.clients)
        elif data.name in {"vqav2", "coco_captions", "medical_vqa"}:
            if not data.manifest:
                raise ValueError(f"data.manifest is required for {data.name}; use a prepared JSONL manifest")
            manifest = Path(to_absolute_path(data.manifest))
        else:
            raise ValueError(f"Unknown dataset: {data.name}")
        from core.data import read_manifest
        rows = read_manifest(manifest, self.protocol.training.task, data.split,
                             data.client, unique_images=True)
        count = self.protocol.training.batch_size * self.protocol.training.local_steps
        if data.offset < 0 or data.client < 0 or data.client >= self.protocol.training.clients:
            raise ValueError("Invalid data client or offset")
        if len(rows) < data.offset + self.config.num_runs * count:
            raise ValueError("Not enough unique-image records for requested run IDs")
        frozen = OmegaConf.to_container(self.config, resolve=True)
        for key in ["output_dir", "num_runs", "start_run_id", "resume", "oom_recovery"]:
            frozen.pop(key, None)
        snapshot_hashes = {}
        if self.config.model_snapshot:
            snapshot = Path(to_absolute_path(self.config.model_snapshot))
            for name in ["model.json", "model.safetensors"]:
                snapshot_hashes[name] = file_hash(snapshot / name)
        signature = digest({"config": frozen, "data_sha256": file_hash(manifest),
                            "snapshot_hashes": snapshot_hashes,
                            "source_sha256": source_fingerprint()})
        state = self.output / "experiment.json"
        if state.exists():
            saved = read_json(state)
            if not self.config.resume:
                raise FileExistsError("Experiment exists; resume=true is required")
            if saved["signature"] != signature:
                raise ValueError("Resume protocol, data, or source changed")
            self.results = {r["run_id"]: r for r in saved["runs"]}
        self.manifest, self.signature = manifest, signature
        write_json(self.output / "config.resolved.json", OmegaConf.to_container(self.config, resolve=True))
        self._save()

    def _save(self):
        write_json(self.output / "experiment.json", {
            "signature": self.signature, "num_runs": self.config.num_runs,
            "runs": [self.results[k] for k in sorted(self.results)]})

    def _run_one(self, run_id):
        directory = self.output / f"run-{run_id:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "protocol.json"
        write_json(config_path, asdict(self.protocol))
        model_path = self._prepare_model(config_path)
        captured = directory / "capture"
        if not captured.exists():
            staging = directory / ".capture-pending"
            # Only this runner's uncommitted staging directory may be discarded.
            if staging.exists():
                shutil.rmtree(staging)
            count = self.protocol.training.batch_size * self.protocol.training.local_steps
            commands.capture(SimpleNamespace(
                config=config_path, set=[], output=staging, data=self.manifest,
                model=model_path,
                split=self.config.data.split, client=self.config.data.client,
                offset=self.config.data.offset + run_id * count,
                defense_config=OmegaConf.to_container(self.config.defense, resolve=True)))
            staging.rename(captured)
        reconstruction = directory / "attack"
        commands.attack(SimpleNamespace(
            config=config_path, set=[], observation=captured / "public", output=reconstruction,
            device=self.protocol.model.device, model=None, wrong_observation=None,
            resume=True, raise_oom=True))
        commands.evaluate(SimpleNamespace(
            config=config_path, set=[], reconstruction=reconstruction, truth=captured / "private",
            device=self.protocol.model.device, output=None))
        result = read_json(reconstruction / "result.json")
        return {"run_id": run_id, "status": result["status"], "reason": result["reason"],
                "result": str((reconstruction / "result.json").relative_to(self.output)),
                "result_sha256": file_hash(reconstruction / "result.json"),
                "evaluation_sha256": file_hash(reconstruction / "evaluation.json")}

    def _prepare_model(self, config_path):
        if self.config.model_snapshot:
            return to_absolute_path(self.config.model_snapshot)
        if self.protocol.training.rounds:
            directory = self.output / "federation"
            commands.train(SimpleNamespace(config=config_path, set=[], data=self.manifest,
                                            output=directory, resume=True))
            return directory / read_json(directory / "training.json")["checkpoint"]
        from core.fl import save_model
        from core.vlm_wrapper import build_model
        directory = self.output / "model"
        if not directory.exists():
            staging = self.output / ".model-pending"
            if staging.exists():
                shutil.rmtree(staging)
            save_model(staging, build_model(self.protocol.model, self.protocol.training))
            staging.rename(directory)
        return directory

    def run_experiments(self):
        torch.set_num_threads(self.config.threads)
        self._prepare()
        for run_id in range(self.config.start_run_id, self.config.num_runs):
            saved = self.results.get(run_id)
            if saved and saved["status"] not in {"oom", "error"}:
                result_path = self.output / saved["result"]
                if (file_hash(result_path) != saved["result_sha256"] or
                        file_hash(result_path.with_name("evaluation.json")) != saved["evaluation_sha256"]):
                    raise ValueError("Completed run artifact integrity check failed")
                continue
            retries = 0
            while True:
                retry = False
                try:
                    self.results[run_id] = {**self._run_one(run_id), "oom_retries": retries}
                except torch.OutOfMemoryError as error:
                    self.results[run_id] = {"run_id": run_id, "status": "oom", "reason": str(error),
                                            "oom_retries": retries}
                    self._save()
                    if not self.config.oom_recovery.enabled or retries >= self.config.oom_recovery.max_retries:
                        raise
                    retries += 1
                    retry = True
                except Exception as error:
                    self.results[run_id] = {"run_id": run_id, "status": "error", "reason": str(error)}
                    self._save()
                    raise
                # Release the exception traceback before retrying the exact same protocol.
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not retry:
                    break
            self._save()
        return read_json(self.output / "experiment.json")


def run_experiment(config):
    return ExperimentRunner(config).run_experiments()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig):
    run_experiment(config)


if __name__ == "__main__":
    main()
