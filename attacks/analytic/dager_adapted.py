"""Public-image LLaVA adaptation of DAGER; no claim of original-paper fidelity."""
from dataclasses import asdict, fields, replace
import hashlib
from pathlib import Path
import time
import uuid

import torch

from attacks.base import BaseAttacker
from attacks.analytic.dager_search import expansion_batch, retain_best
from attacks.analytic.dager_subspace import SpanFilter, select_tokens
from attacks.objectives import matching_loss
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import DAGEROptions, digest, validate_attack_config
from core.fl import simulate_update
from core.types import Batch, Reconstruction, Support


def gradient_groups(adapter, observation, projections):
    parameters = dict(adapter.named_parameters())
    groups = []
    for layer in (0, 1):
        names = []
        for projection in projections:
            prefix = f"backend.language_model.model.layers.{layer}.self_attn.{projection}_proj.lora_A."
            found = [name for name in observation.tensors
                     if name.startswith(prefix) and name.endswith(".weight")]
            if len(found) != 1:
                raise ValueError(f"Required uploaded LoRA-A gradient missing or ambiguous: layer {layer} {projection}")
            name = found[0]
            if name not in parameters or observation.tensors[name].shape != parameters[name].shape:
                raise ValueError(f"Uploaded gradient shape differs from model: {name}")
            names.append(name)
        groups.append(names)
    return groups


def dager_support(adapter, observation, options=None):
    training = observation.training
    if observation.model.family != "llava":
        return Support("not_applicable", "DAGER adaptation currently requires LLaVA/LLaMA")
    if observation.model.device_map:
        return Support("not_applicable", "DAGER partial forward currently requires a single device")
    if training.task != "vqa" or training.knowledge not in {"image_known", "image_question_known"}:
        return Support("not_applicable", "DAGER requires public images and private VQA text")
    if not training.token_lengths_known:
        return Support("not_applicable", "DAGER currently requires declared public token lengths")
    if training.algorithm != "fedsgd" or observation.sample_count != 1:
        return Support("not_applicable", "DAGER currently requires single-sample FedSGD gradients")
    if training.fine_tuning_strategy not in {"f_l", "f_cl"}:
        return Support("not_applicable", "DAGER requires F-L or F-CL LoRA uploads")
    decoder = adapter.backend.language_model.model
    if decoder.config.model_type != "llama" or len(decoder.layers) < 2:
        return Support("not_applicable", "DAGER requires at least two LLaMA decoder blocks")
    try:
        groups = gradient_groups(adapter, observation, (options or DAGEROptions()).projections)
    except ValueError as error:
        return Support("not_applicable", str(error))
    if any(not any(observation.tensors[name].count_nonzero().item() for name in group)
           for group in groups):
        return Support("not_applicable", "Required LoRA-A gradients are zero; initial LoRA-B may be zero")
    return Support("supported")


class SearchBudgetExhausted(Exception):
    pass


def observation_signature(observation, spec):
    h = hashlib.sha256()
    for name, tensor in sorted(observation.tensors.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    public = {field.name: getattr(observation, field.name) for field in fields(observation)
              if field.name not in {"tensors", "public_images", "model", "training"}}
    public.update(model=asdict(observation.model), training=asdict(observation.training))
    images = observation.public_images
    if images is not None:
        h.update(images.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest({"observation": public, "tensor_hash": h.hexdigest(),
                   "attack": asdict(spec), "source": source_fingerprint()})


class DAGERAdaptedAttacker(BaseAttacker):
    def attack(self, gradients, batch_info, knowledge, *, directory=None, resume=False,
               initial_images=None, initial_tokens=None, upload_metadata=None):
        validate_attack_config(self.spec)
        observation = replace(batch_info, tensors=gradients)
        knowledge.validate_observation(observation)
        if upload_metadata is not None and upload_metadata.get("defense", {}).get("name", "none") != "none":
            return Reconstruction("not_applicable", "DAGER currently requires undefended raw gradients")
        if initial_images is not None or initial_tokens is not None:
            raise ValueError("DAGER does not accept candidate initialization")
        support = dager_support(self.adapter, observation, self.spec.dager)
        if support.status != "supported":
            return Reconstruction(support.status, support.reason)
        return DAGERSearch(self.adapter, observation, self.spec, directory).run(resume)


class DAGERSearch:
    SCHEMA = 1

    def __init__(self, adapter, observation, spec, directory):
        self.adapter, self.observation, self.spec = adapter, observation, spec
        self.options = spec.dager
        self.directory = Path(directory) if directory is not None else None
        self.started = time.monotonic()
        self.elapsed_before = 0.0
        self.signature = observation_signature(observation, spec)
        self.state = {"stage": "tokens", "vocab_cursor": 0, "candidates": [],
                      "beam": [{"tokens": [], "score": 0.0}], "depth": 0,
                      "expansion_cursor": 0, "next_beam": [], "rerank_cursor": 0,
                      "reranked": [], "history": [], "provenance": {},
                      "costs": {"vocab_tokens_scored": 0, "prefix_candidates_evaluated": 0,
                                "prefix_forward_batches": 0, "full_gradient_replays": 0,
                                "ambiguous_prefix_checks": 0}}
        self.scores = torch.zeros(adapter.vocab_size, dtype=getattr(torch, self.options.analysis_dtype))
        self.ambiguous = torch.zeros(adapter.vocab_size, dtype=torch.bool)

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    def check_budget(self):
        if self.elapsed() >= self.spec.seconds:
            raise SearchBudgetExhausted()

    def available(self, requested):
        self.check_budget()
        costs = self.state["costs"]
        used = costs["prefix_candidates_evaluated"] + costs["full_gradient_replays"]
        if self.spec.max_evaluations is not None:
            requested = min(requested, self.spec.max_evaluations - used)
        if requested <= 0:
            raise SearchBudgetExhausted()
        return requested

    def save(self):
        if self.directory is None:
            return
        generation = "generation-" + uuid.uuid4().hex
        folder = self.directory / "checkpoints" / generation
        write_tensors(folder / "state.safetensors", {"scores": self.scores, "ambiguous": self.ambiguous})
        write_json(folder / "state.json", {"schema": self.SCHEMA, "signature": self.signature,
                                          "tensor_sha256": file_hash(folder / "state.safetensors"),
                                          "elapsed": self.elapsed(), "state": self.state})
        write_json(self.directory / "checkpoint.json", {"generation": generation})

    def restore(self):
        if self.directory is None or not (self.directory / "checkpoint.json").exists():
            raise ValueError("DAGER resume requires an existing checkpoint")
        generation = read_json(self.directory / "checkpoint.json")["generation"]
        if Path(generation).name != generation:
            raise ValueError("Invalid checkpoint generation")
        folder = self.directory / "checkpoints" / generation
        meta = read_json(folder / "state.json")
        if meta["schema"] != self.SCHEMA or meta["signature"] != self.signature:
            raise ValueError("DAGER checkpoint schema or source/protocol/observation differs")
        if file_hash(folder / "state.safetensors") != meta["tensor_sha256"]:
            raise ValueError("DAGER checkpoint tensor integrity check failed")
        tensors = read_tensors(folder / "state.safetensors")
        self.state = meta["state"]
        self.elapsed_before = meta["elapsed"]
        self.scores, self.ambiguous = tensors["scores"], tensors["ambiguous"]

    def result(self, status, reason="", prefix=None):
        questions, targets = [], []
        if prefix is not None:
            ids = self.view.token_ids(prefix)
            questions = self.adapter.decode(ids["questions"])
            targets = self.adapter.decode(ids["targets"])
        costs = dict(self.state["costs"], seconds=self.elapsed())
        return Reconstruction(status, reason, images=None, questions=questions, targets=targets,
                              costs=costs, history=self.state["history"],
                              provenance=self.state["provenance"])

    def prepare(self):
        self.check_budget()
        self.view = LlavaTextView(self.adapter, self.observation)
        groups = gradient_groups(self.adapter, self.observation, self.options.projections)
        self.spans = []
        for names, public in zip(groups, (self.view.public_layer0, self.view.public_layer1), strict=True):
            self.check_budget()
            self.spans.append(SpanFilter.build(
                [self.observation.tensors[name] for name in names], public, self.options))
        self.state["provenance"].update({
            "method": "dager_adapted", "options": asdict(self.options),
            "gradient_names": groups, "subspaces": [s.diagnostics for s in self.spans],
            "candidate_selection": "observable subspace residuals; optional uploaded-update reranking",
            "exact_recovery_claimed": False,
            "evaluation_unit": "one private prefix extension or one full gradient replay",
            "private_slot_count": len(self.view.slots)})

    @torch.no_grad()
    def filter_tokens(self):
        options = self.options
        while self.state["vocab_cursor"] < self.adapter.vocab_size:
            self.check_budget()
            start = self.state["vocab_cursor"]
            stop = min(start + options.vocab_chunk_size, self.adapter.vocab_size)
            embeds = self.adapter.embedding().weight[start:stop]
            features = self.view.layer_input(embeds, 0)
            scores, ambiguous = self.spans[0].score(features)
            self.scores[start:stop] = scores.cpu()
            self.ambiguous[start:stop] = ambiguous.cpu()
            self.state["vocab_cursor"] = stop
            self.state["costs"]["vocab_tokens_scored"] += stop - start
            self.save()
        forbidden = set(self.adapter.tokenizer.all_special_ids) | {self.adapter.eos, self.adapter.pad}
        candidates, details = select_tokens(self.scores, self.ambiguous, forbidden, options)
        self.state["candidates"] = candidates
        self.state["provenance"]["token_filter"] = details
        if self.directory is not None:
            write_tensors(self.directory / "token_candidates.safetensors", {
                "token_ids": torch.arange(self.adapter.vocab_size), "scores": self.scores,
                "ambiguous": self.ambiguous,
                "selected_ids": torch.tensor(candidates, dtype=torch.long)})
            write_json(self.directory / "token_candidates.json", {
                "schema_version": 1, "unit": "token_id", "scope": "private_text_union",
                "forbidden_ids": sorted(forbidden), "selection": details,
                "note": "Ambiguous public-overlap tokens are retained, not detected positives."})
        self.state["stage"] = "search"
        self.save()

    @torch.no_grad()
    def search(self):
        while self.state["depth"] < len(self.view.slots):
            beam, candidates = self.state["beam"], self.state["candidates"]
            total = len(beam) * len(candidates)
            while self.state["expansion_cursor"] < total:
                count = self.available(min(self.options.prefix_batch_size,
                                           total - self.state["expansion_cursor"]))
                additions = expansion_batch(beam, candidates, self.state["expansion_cursor"], count)
                self.state["costs"]["prefix_candidates_evaluated"] += len(additions)
                self.state["costs"]["prefix_forward_batches"] += 1
                features = self.view.prefix_features([row["tokens"] for row in additions])
                scores, ambiguous = self.spans[1].score(features)
                self.state["costs"]["ambiguous_prefix_checks"] += int(ambiguous.sum())
                for row, score in zip(additions, scores.cpu().tolist(), strict=True):
                    row["score"] += score
                self.state["next_beam"] = retain_best(
                    self.state["next_beam"], additions, self.options.beam_width)
                self.state["expansion_cursor"] += len(additions)
                if self.state["costs"]["prefix_forward_batches"] % self.spec.checkpoint_interval == 0:
                    self.save()
            self.state["beam"] = self.state["next_beam"]
            self.state["next_beam"] = []
            self.state["expansion_cursor"] = 0
            self.state["depth"] += 1
            self.state["history"].append({"private_tokens_filled": self.state["depth"],
                                          "beam_size": len(self.state["beam"]),
                                          "best_span_score": self.state["beam"][0]["score"]})
            self.save()
        self.state["stage"] = "rerank"
        self.save()

    def rerank(self):
        limit = min(self.options.rerank_candidates, len(self.state["beam"]))
        while self.state["rerank_cursor"] < limit:
            self.available(1)
            row = self.state["beam"][self.state["rerank_cursor"]]
            ids = self.view.token_ids(row["tokens"])
            batch = Batch(self.observation.public_images.to(self.adapter.device, self.adapter.dtype),
                          ids["questions"], ids["targets"])
            self.state["costs"]["full_gradient_replays"] += 1
            update = simulate_update(self.adapter, batch, self.observation.training, False)
            update = {name: update[name] for name in self.observation.tensors}
            loss = matching_loss(update, self.observation.tensors, "l2")
            denominator = sum(value.float().square().sum().to(loss.device)
                              for value in self.observation.tensors.values())
            score = float(loss / denominator.clamp_min(1e-20))
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("Nonfinite DAGER candidate update score")
            self.state["reranked"].append({"tokens": row["tokens"], "score": score})
            self.state["rerank_cursor"] += 1
            self.save()
        self.state["stage"] = "done"
        self.save()

    def run(self, resume):
        if (not resume and self.directory is not None and self.directory.exists()
                and any(self.directory.iterdir())):
            raise FileExistsError("DAGER output is nonempty; use resume or a new directory")
        has_checkpoint = (self.directory is not None
                          and (self.directory / "checkpoint.json").exists())
        checkpoint_valid = not resume or not has_checkpoint
        try:
            # ExperimentRunner passes resume=True for a fresh attack so it can
            # safely continue an interrupted capture/attack pipeline. Restore
            # only after this attack has actually committed a checkpoint.
            if resume and has_checkpoint:
                self.restore()
                checkpoint_valid = True
            self.prepare()
            if any(not len(span.basis) for span in self.spans) and self.view.slots:
                self.save()
                return self.result("no_signal", "Selected gradient subspace has zero numerical rank")
            if self.state["stage"] == "tokens":
                self.filter_tokens()
            if not self.state["candidates"] and self.view.slots:
                return self.result("no_candidates", "No vocabulary candidates passed the declared filter")
            if self.state["stage"] == "search":
                self.search()
            if self.state["stage"] == "rerank":
                self.rerank()
            ranked = self.state["reranked"] or self.state["beam"]
            winner = retain_best([], ranked, 1)[0]
            self.state["provenance"]["final_score"] = winner["score"]
            self.state["provenance"]["final_score_type"] = (
                "relative_l2_update" if self.state["reranked"] else "summed_prefix_span_distance")
            if self.directory is not None:
                write_tensors(self.directory / "text_tokens.safetensors", self.view.token_ids(winner["tokens"]))
            return self.result("completed", prefix=winner["tokens"])
        except SearchBudgetExhausted:
            self.save()
            return self.result("budget_exhausted", "Search stopped at the declared time/evaluation budget")
        except Exception as error:
            # A rejected resume must not overwrite the checkpoint it rejected.
            if self.directory is not None:
                if checkpoint_valid:
                    self.save()
                write_json(self.directory / "failed.json", {
                    "status": "failed", "error_type": type(error).__name__, "reason": str(error),
                    "signature": self.signature, "stage": self.state["stage"],
                    "costs": self.state["costs"]})
            raise
