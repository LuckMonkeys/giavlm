"""Public-slot views for one pooled upload; never split the observed update.

The beam fills all sample slots jointly. Sample boundaries reset the causal
context, while the geometric and language-prior scores remain cumulative.
There is no oracle assignment, deduplication or assumption of unique text.
"""
from dataclasses import replace

import torch


def text_view(adapter, observation):
    if observation.model.family == "qwen3_5":
        from core.adapters.qwen35 import Qwen35TextView
        return Qwen35TextView(adapter, observation)
    from core.adapters.llava import LlavaTextView
    return LlavaTextView(adapter, observation)


class PooledTextView:
    @torch.no_grad()
    def __init__(self, adapter, observation):
        self.views, self.offsets, self.slots = [], [], []
        for index in range(observation.sample_count):
            # This is a layout-only view of declared public fields. Search keeps
            # the original observation, including its pooled upload/training.
            fields = {key: getattr(observation, key)[index:index + 1] for key in (
                "public_questions", "public_targets", "public_question_ids", "public_target_ids",
                "public_question_lengths", "public_target_lengths")}
            slot = replace(observation, **fields,
                           training=replace(observation.training, batch_size=1, local_steps=1,
                                            gradient_accumulation_steps=1),
                           public_images=observation.public_images[index:index + 1])
            view = text_view(adapter, slot)
            self.offsets.append(len(self.slots))
            self.slots.extend(view.slots)
            self.views.append(view)
        # Remove known inputs from every sample, not just the slot being searched.
        self.public_layer0 = torch.cat([view.public_layer0 for view in self.views])
        self.public_layer1 = torch.cat([view.public_layer1 for view in self.views])

    def _active(self, depth):
        for view, offset in zip(self.views, self.offsets, strict=True):
            if offset <= depth < offset + len(view.slots):
                return view, offset
        raise ValueError("No private slot at this pooled prefix depth")

    def layer_input(self, embeddings, layer):
        return self.views[0].layer_input(embeddings, layer)

    def prefix_features(self, prefixes):
        view, offset = self._active(len(prefixes[0]) - 1)
        return view.prefix_features([prefix[offset:] for prefix in prefixes])

    def text_prior_nll(self, prefixes, candidates):
        view, offset = self._active(len(prefixes[0]))
        return view.text_prior_nll([prefix[offset:] for prefix in prefixes], candidates)

    def token_ids(self, prefix):
        if len(prefix) != len(self.slots):
            raise ValueError("A pooled reconstruction must fill every private slot")
        rows = [view.token_ids(prefix[offset:offset + len(view.slots)])
                for view, offset in zip(self.views, self.offsets, strict=True)]
        return {key: torch.cat([row[key] for row in rows]) for key in ("questions", "targets")}
