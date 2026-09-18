"""Explicit VLM adaptations. Original paper implementations are not aliases."""
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from core.artifacts import read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import AttackSpec, digest
from core.fl import simulate_update
from core.vlm_wrapper import canonical_probabilities
from attacks.priors import BatchNormPrior, TextPrior, total_variation
from core.types import Batch, Reconstruction


from attacks.objectives import matching_loss
from attacks.registry import METHODS, supports

class Candidate(nn.Module):
    def __init__(self, adapter, observation, seed):
        super().__init__()
        self.adapter = [adapter]  # The victim is not part of candidate parameters/state.
        self.observation = observation
        g = torch.Generator(device=adapter.device).manual_seed(seed)
        n, m = observation.sample_count, observation.model
        self.images = nn.Parameter(torch.rand(n, 3, m.image_size, m.image_size,
                                              generator=g, device=adapter.device, dtype=adapter.dtype))
        self.private_q = observation.training.task == "vqa" and observation.training.knowledge == "private"
        self.private_y = observation.training.knowledge != "text_known"
        for field, length, private, public in [
            ("questions", m.question_length, self.private_q, observation.public_questions),
            ("targets", m.target_length, self.private_y, observation.public_targets)]:
            if private:
                values = torch.randn(n, length, adapter.vocab_size, generator=g,
                                     device=adapter.device, dtype=torch.float32) * 0.1
                self.register_parameter(field, nn.Parameter(values))
            else:
                public_ids = observation.public_question_ids if field == "questions" else observation.public_target_ids
                ids = torch.tensor(public_ids, device=adapter.device, dtype=torch.long) if public_ids else adapter.encode(public or [""] * n, length)
                self.register_buffer(field, F.one_hot(ids, adapter.vocab_size).to(adapter.dtype))
        forbidden = sorted(set(adapter.tokenizer.all_special_ids) - {adapter.eos})
        self.register_buffer("forbidden", torch.tensor(forbidden, device=adapter.device, dtype=torch.long))

    def distribution(self, name, discrete=False):
        adapter = self.adapter[0]
        value = getattr(self, name)
        private = self.private_q if name == "questions" else self.private_y
        if private:
            value = value.index_fill(-1, self.forbidden, -1e4)
            # A public maximum-length EOS, never the private stopping position.
            last = value.new_full(value[:, -1:].shape, -1e4)
            last[:, :, adapter.eos] = 0
            value = torch.cat([value[:, :-1], last], dim=1).softmax(-1)
        if discrete:
            value = F.one_hot(value.argmax(-1), adapter.vocab_size).to(adapter.dtype)
        value = value.to(adapter.dtype)
        return canonical_probabilities(value, adapter.eos, adapter.pad)[0] if discrete else value

    def batch(self, discrete=False):
        return Batch(self.images.clamp(0, 1), self.distribution("questions", discrete),
                     self.distribution("targets", discrete))

    def decoded(self):
        adapter = self.adapter[0]
        batch = self.batch(True)
        return adapter.decode(batch.questions), adapter.decode(batch.targets)


class BudgetExhausted(Exception):
    pass


class AttackRunner:
    def __init__(self, adapter, observation, spec: AttackSpec):
        self._validate_spec(spec)
        self.adapter, self.observation, self.spec = adapter, observation, spec
        self.evaluations = 0
        self.local_backward_evaluations = 0
        self.prior_evaluations = 0
        self.iterations_completed = 0
        self.restarts_started = 0
        self.elapsed_before = 0.0
        self.history = []
        self.best = None
        self.best_score = math.inf
        self.text_prior = None
        self.image_prior = None

    @staticmethod
    def _validate_spec(spec):
        """Validate optimizer details where they are actually consumed."""
        if spec.lr <= 0 or spec.text_lr <= 0:
            raise ValueError("Attack learning rates must be positive")
        if spec.method != "random" and spec.checkpoint_interval <= 0:
            raise ValueError("attack.checkpoint_interval must be positive")
        if spec.text_method == "lamp_adapted" and spec.prior_interval <= 0:
            raise ValueError("attack.prior_interval must be positive for LAMP")

    def check_budget(self):
        if self.evaluations >= self.spec.max_evaluations or self.elapsed() >= self.spec.seconds:
            raise BudgetExhausted()

    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started

    def replay(self, batch, differentiable):
        """Candidate update restricted to what the client actually uploaded.

        The local replay necessarily computes every trainable parameter, but the
        attacker may only match the uploaded subset. Filtering here keeps that
        boundary in one place and preserves the key-set equality matching_loss
        checks.
        """
        self.check_budget()
        self.evaluations += 1
        self.local_backward_evaluations += (
            self.observation.training.local_steps
            * self.observation.training.gradient_accumulation_steps)
        update = simulate_update(self.adapter, batch, self.observation.training, differentiable)
        return {name: update[name] for name in self.observation.tensors}

    def text_score(self, candidate):
        if self.text_prior is None:
            return 0.0
        self.prior_evaluations += 1
        questions, targets = candidate.decoded()
        texts = (questions if candidate.private_q else []) + (targets if candidate.private_y else [])
        return self.text_prior.score(texts)

    def discrete_score(self, candidate):
        if self.spec.method in {"random", "prior_only"}:
            score = self.spec.tv * total_variation(candidate.images).item()
        else:
            predicted = self.replay(candidate.batch(True), False)
            # Common observable residual selects restart/iterate, never a reference metric.
            squared = matching_loss(predicted, self.observation.tensors, "l2")
            denominator = sum(x.float().square().sum().to(squared.device) for x in self.observation.tensors.values())
            score = (squared / denominator.clamp_min(1e-20)).item()
        return score + self.spec.text_prior * self.text_score(candidate)

    def remember(self, candidate, score):
        if math.isfinite(score) and score < self.best_score:
            self.best_score = score
            q, y = candidate.decoded()
            self.best = (candidate.images.detach().clamp(0, 1).cpu().clone(), q, y)

    def objective(self, candidate, iteration, text_phase):
        batch = candidate.batch()
        method = self.spec.method
        if method == "prior_only":
            value = self.spec.tv * total_variation(batch.images)
            # Text optimization still has a graph when only a discrete prior is available.
            return value + sum(p.sum() * 0 for p in candidate.parameters())
        predicted = self.replay(batch, True)
        from attacks.optim import dlg, inverting_gradients, gidqa, gradvit
        strategies = {"dlg_adapted": dlg, "ig_adapted": inverting_gradients,
                      "gi_dqa_adapted": gidqa, "gradvit_adapted": gradvit}
        strategy = strategies.get(method)
        kind = strategy.GRADIENT_OBJECTIVE if strategy else "l2"
        if text_phase and self.spec.text_method == "tag_adapted":
            kind = "tag"
        if kind == "tag":
            from attacks.text.tag import gradient_objective
            value = gradient_objective(predicted, self.observation.tensors).to(batch.images.device)
        else:
            value = matching_loss(predicted, self.observation.tensors, kind).to(batch.images.device)
        fraction = iteration / max(1, self.spec.iterations)
        if method == "april_adapted":
            names = self.adapter.position_gradient_names
            value = value + self.spec.april_weight * matching_loss(
                {k: predicted[k] for k in names}, {k: self.observation.tensors[k] for k in names}, "cosine").to(value.device)
        if strategy and hasattr(strategy, "regularize"):
            value = strategy.regularize(self, value, batch.images, fraction)
        return value

    def permutation_step(self, candidate):
        from attacks.text.lamp import permutation_step
        return permutation_step(self, candidate)

    def save_checkpoint(self, directory, candidate, optimizer, restart, iteration, signature):
        if not directory:
            return
        # Each immutable generation is committed by replacing a small pointer last.
        generation = Path(directory) / "checkpoints" / f"r{restart}-i{iteration}-e{self.evaluations}"
        tensors = {f"candidate.{k}": v for k, v in candidate.state_dict().items()}
        tensors["rng.cpu"] = torch.get_rng_state()
        if self.adapter.device.type == "cuda":
            tensors["rng.cuda"] = torch.cuda.get_rng_state(self.adapter.device)
        meta = {"schema_version": 3, "signature": signature,
                "restart": restart, "iteration": iteration,
                "iterations_completed": self.iterations_completed,
                "restarts_started": self.restarts_started,
                "evaluations": self.evaluations, "local_backward_evaluations": self.local_backward_evaluations,
                "prior_evaluations": self.prior_evaluations, "elapsed": self.elapsed(), "history": self.history,
                "best_score": self.best_score if math.isfinite(self.best_score) else None}
        if self.best is not None:
            tensors["best.images"] = self.best[0]
            meta["best_questions"], meta["best_targets"] = self.best[1:]
        meta["optimizer"] = _pack(optimizer.state_dict(), tensors)
        write_tensors(generation / "state.safetensors", tensors)
        write_json(generation / "state.json", meta)
        write_json(Path(directory) / "checkpoint.json",
                   {"schema_version": 3, "generation": generation.name})

    def run(self, directory=None, resume=False):
        support = supports(self.spec.method, self.adapter, self.observation)
        if support.status != "supported":
            return Reconstruction(support.status, support.reason)
        self.observation.validate()
        trainable = set(self.adapter.trainable())
        observed = set(self.observation.tensors)
        if not observed <= trainable:
            raise ValueError("Observed parameters are not a subset of the configured trainable parameters")
        if not self.observation.training.upload_parameters and observed != trainable:
            raise ValueError("Observed parameters do not match the configured trainable parameters")
        if self.adapter.fingerprint() != self.observation.model_fingerprint:
            raise ValueError("Attack model differs from the observed model state")
        parameters = self.adapter.trainable()
        self.observation.tensors = {k: v.to(parameters[k].device) for k, v in self.observation.tensors.items()}
        if self.spec.text_method == "lamp_adapted" and self.observation.training.knowledge != "text_known":
            if self.spec.prior_model == "tiny-public-bigram" and self.adapter.spec.family != "tiny":
                raise ValueError("The tiny bigram prior is a fixture, not a VLM benchmark prior")
            self.text_prior = TextPrior(self.spec.prior_model, self.spec.prior_revision,
                                        self.adapter.device, self.spec.allow_prior_download)
        if self.spec.method == "gradvit_adapted":
            self.image_prior = BatchNormPrior(self.spec.gradvit_prior_checkpoint, self.adapter.device)
        self.started = time.monotonic()
        if self.adapter.device.type == "cuda":
            for device in {p.device for p in self.adapter.parameters()}:
                torch.cuda.reset_peak_memory_stats(device)
        signature = digest({"attack": asdict(self.spec), "fingerprint": self.observation.model_fingerprint,
                            "source_sha256": source_fingerprint(),
                            "training": asdict(self.observation.training),
                            "observed": _tensor_digest(self.observation.tensors),
                            "questions": self.observation.public_questions,
                            "targets": self.observation.public_targets,
                            "question_ids": self.observation.public_question_ids,
                            "target_ids": self.observation.public_target_ids})
        saved = None
        pointer = Path(directory) / "checkpoint.json" if directory else None
        if resume and pointer and pointer.exists():
            checkpoint = read_json(pointer)
            if checkpoint.get("schema_version") != 3:
                raise ValueError("Unsupported attack checkpoint schema; expected schema v3")
            generation = Path(directory) / "checkpoints" / checkpoint["generation"]
            saved = read_json(generation / "state.json")
            if saved.get("schema_version") != 3:
                raise ValueError("Unsupported attack state schema; expected schema v3")
            if saved["signature"] != signature:
                raise ValueError("Resume configuration/observation differs from checkpoint")
            state = read_tensors(generation / "state.safetensors", str(self.adapter.device))
            self.evaluations, self.elapsed_before = saved["evaluations"], saved["elapsed"]
            self.local_backward_evaluations = saved["local_backward_evaluations"]
            self.prior_evaluations = saved["prior_evaluations"]
            self.iterations_completed = saved["iterations_completed"]
            self.restarts_started = saved["restarts_started"]
            self.history = saved["history"]
            if saved["best_score"] is not None:
                self.best_score = saved["best_score"]
                self.best = (state["best.images"].cpu(), saved["best_questions"], saved["best_targets"])
        reason = "iterations_completed"
        for restart in range(saved["restart"] if saved else 0, self.spec.restarts):
            self.restarts_started = max(self.restarts_started, restart + 1)
            torch.manual_seed(self.spec.seed + restart)
            candidate = Candidate(self.adapter, self.observation, self.spec.seed + restart)
            image_params = [candidate.images]
            text_params = [p for n, p in candidate.named_parameters() if n != "images"]
            if self.spec.method == "dlg_adapted":
                optimizer = torch.optim.LBFGS(list(candidate.parameters()), lr=self.spec.lr,
                                              max_iter=1, history_size=10, line_search_fn=None)
            else:
                optimizer = torch.optim.Adam([{"params": image_params, "lr": self.spec.lr},
                                               {"params": text_params, "lr": self.spec.text_lr}])
            start = 0
            if saved:
                candidate.load_state_dict({k.removeprefix("candidate."): v for k, v in state.items()
                                           if k.startswith("candidate.")})
                optimizer.load_state_dict(_unpack(saved["optimizer"], state))
                start = saved["iteration"]
                torch.set_rng_state(state["rng.cpu"].cpu())
                if "rng.cuda" in state:
                    torch.cuda.set_rng_state(state["rng.cuda"].cpu(), self.adapter.device)
                saved = None
            try:
                if self.spec.method == "random":
                    self.remember(candidate, 0.0)
                    break
                if start == 0:
                    self.remember(candidate, self.discrete_score(candidate))
                for iteration in range(start, self.spec.iterations):
                    self.check_budget()
                    text_phase = bool(text_params) and iteration % 2 == 1 and self.spec.method != "dlg_adapted"

                    def closure():
                        optimizer.zero_grad(set_to_none=True)
                        value = self.objective(candidate, iteration, text_phase)
                        if not torch.isfinite(value):
                            raise FloatingPointError("Nonfinite attack objective")
                        params = list(candidate.parameters())
                        grads = torch.autograd.grad(value, params, allow_unused=True)
                        for p, grad in zip(params, grads):
                            if grad is not None and not torch.isfinite(grad).all():
                                raise FloatingPointError("Nonfinite candidate gradient")
                            p.grad = grad
                        if self.spec.method != "dlg_adapted":
                            if text_phase:
                                candidate.images.grad = None
                            else:
                                for p in text_params:
                                    p.grad = None
                                if self.spec.method == "ig_adapted" and candidate.images.grad is not None:
                                    candidate.images.grad.sign_()
                        return value

                    loss = optimizer.step(closure)
                    self.iterations_completed += 1
                    with torch.no_grad():
                        candidate.images.clamp_(0, 1)
                    if self.text_prior and (iteration + 1) % self.spec.prior_interval == 0:
                        self.permutation_step(candidate)
                    if (iteration + 1) % self.spec.checkpoint_interval == 0 or iteration + 1 == self.spec.iterations:
                        score = self.discrete_score(candidate)
                        self.remember(candidate, score)
                        self.history.append({"restart": restart, "iteration": iteration + 1,
                                             "objective": float(loss.detach()), "discrete_score": score,
                                             "evaluations": self.evaluations})
                        self.save_checkpoint(directory, candidate, optimizer, restart, iteration + 1, signature)
                reason = "iterations_completed"
            except BudgetExhausted:
                reason = "budget_exhausted"
                break
            except FloatingPointError as error:
                reason = str(error)
                self.history.append({"restart": restart, "failure": reason})
                continue
        from metrics.cost import resource_costs
        costs = resource_costs(self.adapter, self.elapsed(), self.evaluations,
                               self.local_backward_evaluations, self.prior_evaluations)
        costs.update(iterations_completed=self.iterations_completed, restarts_started=self.restarts_started)
        provenance = {"method": self.spec.method, "text_method": self.spec.text_method,
                      "description": METHODS[self.spec.method], "attack": asdict(self.spec),
                      "signature": signature, "protocol": self.adapter.protocol_version,
                      "best_observable_score": self.best_score if self.best else None,
                      "text_prior": self.text_prior.provenance if self.text_prior else None,
                      "image_prior": self.image_prior.provenance if self.image_prior else None}
        if self.best is None:
            return Reconstruction("attack_failed", reason, costs=costs, history=self.history, provenance=provenance)
        status = "completed" if reason in {"iterations_completed", "budget_exhausted"} else "attack_failed"
        return Reconstruction(status, reason, *self.best, costs, self.history, provenance)


def _tensor_digest(tensors):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _pack(value, tensors):
    if isinstance(value, torch.Tensor):
        key = f"optimizer.{len(tensors)}"
        tensors[key] = value
        return {"tensor": key}
    if isinstance(value, dict):
        return {"mapping": [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"sequence": [_pack(v, tensors) for v in value], "tuple": isinstance(value, tuple)}
    return value


def _unpack(value, tensors):
    if isinstance(value, dict):
        if "tensor" in value:
            return tensors[value["tensor"]]
        if "mapping" in value:
            return {_unpack(k, tensors): _unpack(v, tensors) for k, v in value["mapping"]}
        items = [_unpack(v, tensors) for v in value["sequence"]]
        return tuple(items) if value["tuple"] else items
    return value
