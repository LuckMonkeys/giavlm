import torch

from core.adapters.hf import HFAdapter


def llava_input_pieces(adapter, visual, question, response):
    """One fixed-slot layout shared by victim forward and discrete prefix attacks."""
    batch = len(visual)
    instruction = (adapter.public_embeddings("\nDescribe the image.", batch)
                   if adapter.training_spec.task == "caption"
                   else torch.cat([adapter.public_embeddings("\n", batch), question], 1))
    return [adapter.public_embeddings("USER: ", batch), visual, instruction,
            adapter.public_embeddings(" ASSISTANT: ", batch), response]


class LlavaTextView:
    """Public-image, known-length inputs and exact first-block LLaMA replay.

    Unknown slots are placeholders only. Prefix replay never includes an unknown
    future slot. No private batch, loss mask or dataset is accepted here.
    """

    @torch.no_grad()
    def __init__(self, adapter, observation):
        self.adapter = adapter
        self.decoder = adapter.backend.language_model.model
        self.device = adapter.embedding().weight.device
        model, training = observation.model, observation.training
        self.question_ids = self._slots(model.question_length, adapter.eos, adapter.pad,
                                        observation.public_question_lengths[0])
        if training.question_public:
            self.question_ids = list(observation.public_question_ids[0])
            if adapter.content_lengths(torch.tensor([self.question_ids])) != observation.public_question_lengths:
                raise ValueError("Public question IDs differ from declared public lengths")
        self.target_ids = self._slots(model.target_length, adapter.eos, adapter.pad,
                                      observation.public_target_lengths[0])
        visual, _ = adapter.visual_embeddings(observation.public_images.to(
            device=self.device, dtype=adapter.dtype))
        visual = visual.to(self.device)
        question = adapter.embedding()(torch.tensor([self.question_ids], device=self.device))
        response = adapter.embedding()(torch.tensor([self.target_ids[:-1]], device=self.device))
        pieces = llava_input_pieces(adapter, visual, question, response)
        self.embeddings = torch.cat(pieces, 1)
        question_start = pieces[0].shape[1] + visual.shape[1] + adapter.public_embeddings(
            "\n", 1).shape[1]
        response_start = sum(piece.shape[1] for piece in pieces[:-1])
        self.slots = []
        if training.question_private:
            self.slots.extend(("questions", i, question_start + i)
                              for i in range(observation.public_question_lengths[0]))
        self.slots.extend(("targets", i, response_start + i)
                          for i in range(observation.public_target_lengths[0]))
        known = torch.ones(self.embeddings.shape[1], dtype=torch.bool, device=self.device)
        for _, _, position in self.slots:
            known[position] = False
        self.public_layer0 = self.decoder.layers[0].input_layernorm(self.embeddings[:, known])[0]
        # Only this causal prefix is known at deeper layers. Public tokens after
        # an unknown question are context-dependent and cannot be projected away.
        end = self.slots[0][2] if self.slots else self.embeddings.shape[1]
        self.public_layer1 = self.layer_input(self.embeddings[:, :end], 1)[0]

    @staticmethod
    def _slots(length, eos, pad, content_length):
        return [pad] * content_length + [eos] + [pad] * (length - content_length - 1)

    @torch.no_grad()
    def layer_input(self, embeddings, layer):
        if layer == 0:
            return self.decoder.layers[0].input_layernorm(embeddings)
        if layer != 1:
            raise ValueError("LLaVA text view exposes only the first two attention inputs")
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        position_ids = positions.unsqueeze(0)
        mask = self.decoder._update_causal_mask(None, embeddings, positions, None, False)
        hidden = self.decoder.layers[0](
            embeddings, attention_mask=mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=positions,
            position_embeddings=self.decoder.rotary_emb(embeddings, position_ids))[0]
        return self.decoder.layers[1].input_layernorm(hidden)

    @torch.no_grad()
    def prefix_features(self, prefixes):
        depth = len(prefixes[0])
        if not depth or any(len(p) != depth for p in prefixes) or depth > len(self.slots):
            raise ValueError("Prefixes must fill the same positive number of private slots")
        end = self.slots[depth - 1][2] + 1
        values = self.embeddings[:, :end].expand(len(prefixes), -1, -1).clone()
        positions = [slot[2] for slot in self.slots[:depth]]
        ids = torch.tensor(prefixes, device=self.device, dtype=torch.long)
        values[:, positions] = self.adapter.embedding()(ids)
        return self.layer_input(values, 1)[:, -1]

    def token_ids(self, prefix):
        if len(prefix) != len(self.slots):
            raise ValueError("A reconstruction must fill every private content slot")
        fields = {"questions": self.question_ids.copy(), "targets": self.target_ids.copy()}
        for (field, index, _), token in zip(self.slots, prefix, strict=True):
            fields[field][index] = token
        return {key: torch.tensor([ids], device=self.device) for key, ids in fields.items()}


class LlavaAdapter(HFAdapter):
    def visual_embeddings(self, images):
        pixels = ((images - self.image_mean) / self.image_std).to(self.dtype)
        return self.backend.get_image_features(
            pixels, self.backend.config.vision_feature_layer,
            self.backend.config.vision_feature_select_strategy), None
