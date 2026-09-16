"""Canonical Hydra lifecycle, with immutable protocol and run-ID based recovery."""
from dataclasses import asdict
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
    # Hydra group labels are provenance; strict protocol dataclasses contain behavior only.
    fed.pop("name")
    if fed.pop("secure_aggregation"):
        raise NotImplementedError("Aggregate inversion is not an individual-client observation")
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

        # Get verifed config
        self.protocol = protocol_config(config)
        if not 0 <= config.start_run_id < config.num_runs:
            raise ValueError("Require 0 <= start_run_id < num_runs (exclusive stop run ID)")
        if config.output_dir:
            self.output = Path(to_absolute_path(config.output_dir))
        elif HydraConfig.initialized():
            self.output = Path(HydraConfig.get().runtime.output_dir)
        else:
            raise ValueError("Set output_dir when calling run_experiment without Hydra")
        self.output.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def _prepare(self):
        """Prepare experiment-wide inputs and restore resumable state.

        This runs once before the run-ID loop. It resolves the shared data
        manifest, verifies that every requested run has a deterministic sample
        window, computes the experiment identity used for safe resume, and
        restores the index of completed runs. Model preparation and all
        private/public capture work happen later.
        """
        from core.data import synthetic
        data = self.config.data

        # Synthetic data is a self-contained correctness fixture. Real datasets
        # must already be normalized into the benchmark's JSONL manifest format.
        if data.name == "synthetic":
            if self.protocol.model.family != "tiny":
                raise ValueError("Synthetic fixture is restricted to the tiny model")
            manifest = self.output / "data" / "samples.jsonl"
            if not manifest.exists():
                synthetic(self.output / "data", data.count, data.seed, self.protocol.training.clients)
        elif data.name in {"vqav2", "coco_captions", "medical_vqa", "vqa_rad", "slake"}:
            if not data.manifest:
                raise ValueError(f"data.manifest is required for {data.name}; use a prepared JSONL manifest")
            manifest = Path(to_absolute_path(data.manifest))
        else:
            raise ValueError(f"Unknown dataset: {data.name}")

        # A run consumes a fixed, non-overlapping window of unique images. Check
        # the full requested ID range now so a sweep cannot fail halfway through.
        from core.data import read_manifest
        rows = read_manifest(manifest, self.protocol.training.task, data.split,
                             data.client, unique_images=True)
        count = self.protocol.training.batch_size * self.protocol.training.local_steps
        if data.offset < 0 or data.client < 0 or data.client >= self.protocol.training.clients:
            raise ValueError("Invalid data client or offset")
        if len(rows) < data.offset + self.config.num_runs * count:
            raise ValueError("Not enough unique-image records for requested run IDs")

        # The signature identifies the scientific condition, not its scheduling.
        # Excluding run controls permits extending num_runs or resuming at another
        # run ID without allowing protocol, data, weights, or source code to drift.
        frozen = OmegaConf.to_container(self.config, resolve=True)
        for key in ["output_dir", "num_runs", "start_run_id", "resume"]:
            frozen.pop(key, None)
        snapshot_hashes = {}
        if self.config.model_snapshot:
            # Hash contents as well as retaining the configured path, so replacing
            # files in-place is detected during resume.
            snapshot = Path(to_absolute_path(self.config.model_snapshot))
            for name in ["model.json", "model.safetensors"]:
                snapshot_hashes[name] = file_hash(snapshot / name)
        signature = digest({"config": frozen, "data_sha256": file_hash(manifest),
                            "snapshot_hashes": snapshot_hashes,
                            "source_sha256": source_fingerprint()})

        # Existing output is reusable only when explicitly requested and when it
        # belongs to this exact experiment identity.
        state = self.output / "experiment.json"
        if state.exists():
            saved = read_json(state)
            if not self.config.resume:
                raise FileExistsError("Experiment exists; resume=true is required")
            if saved["signature"] != signature:
                raise ValueError("Resume protocol, data, or source changed")
            self.results = {r["run_id"]: r for r in saved["runs"]}

        # Persist provenance before the first run so failures still leave enough
        # state to diagnose and safely resume the experiment.
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
        if self.protocol.training.rounds:
            directory = self.output / "federation"
            initial_model = (to_absolute_path(self.config.model_snapshot)
                             if self.config.model_snapshot else None)
            commands.train(SimpleNamespace(config=config_path, set=[], data=self.manifest,
                                            output=directory, resume=True,
                                            initial_model=initial_model))
            return directory / read_json(directory / "training.json")["checkpoint"]
        if self.config.model_snapshot:
            return to_absolute_path(self.config.model_snapshot)
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
        torch.set_num_threads(self.config.threads) # Really Needed?
        self._prepare()
        for run_id in range(self.config.start_run_id, self.config.num_runs):
            # Reuse completed runs only after verifying their committed artifacts.
            saved = self.results.get(run_id)
            if saved and saved["status"] not in {"oom", "error"}:
                result_path = self.output / saved["result"]
                if (file_hash(result_path) != saved["result_sha256"] or
                        file_hash(result_path.with_name("evaluation.json")) != saved["evaluation_sha256"]):
                    raise ValueError("Completed run artifact integrity check failed")
                continue

            try:
                self.results[run_id] = self._run_one(run_id)
            except Exception as error:
                self.results[run_id] = {"run_id": run_id, "status": "error", "reason": str(error)}
                self._save()
                raise
            self._save()
        return read_json(self.output / "experiment.json")


def run_experiment(config):
    return ExperimentRunner(config).run_experiments()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig):
    run_experiment(config)


if __name__ == "__main__":
    main()
