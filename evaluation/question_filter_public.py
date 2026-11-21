"""Public authored calibration, text-only proposals and exact shallow features."""
import json
from pathlib import Path

import torch
from PIL import Image

from core.artifacts import file_hash, write_json, write_tensors
from core.fl import capture, simulate_update
from defenses.gaussian_noise import GaussianNoiseDefense
from defenses.topk_sparsify import TopKSparsifyDefense


PICTURES = ('astronaut.png', 'camera.png', 'coffee.png', 'chelsea.png', 'rocket.jpg',
            'coins.png', 'moon.png', 'hubble_deep_field.jpg', 'motorcycle_left.png',
            'grass.png', 'brick.png', 'gravel.png')
QUESTIONS = ('What objects can be seen in the photograph?', 'What color is the camera?',
             'Where is the cup in the image?', 'Which animal appears in this picture?',
             'What vehicle is shown in the image?', 'How many coins are visible?',
             'What shape is the bright region?', 'What objects appear in the dark sky?',
             'Which vehicle is visible in the picture?', 'What covers the ground?',
             'What color are the blocks?', 'What texture does the surface have?')
ANSWERS = ('A person wearing a helmet.', 'Black.', 'Beside a plate.', 'A cat.',
           'A rocket.', 'Several coins.', 'Round.', 'Many stars.', 'A motorcycle.',
           'Green grass.', 'Red.', 'Rough.')
PROMPTS = (
    'The following are noisy candidate answer pieces for an image question: {pieces}.\n'
    'Some pieces may be incorrect or incomplete. Write one possible question:',
    'Possible answer pieces, possibly noisy: {pieces}.\n'
    'A question about the image that could have this answer is:',
    'Candidate answer fragments: {pieces}.\n'
    'Generate a single image question consistent with some of these fragments:',
)
CONTEXT_MODULES = {'l1_o': 'layers.0.self_attn.o_proj', 'l1_gate': 'layers.0.mlp.gate_proj',
                   'l1_up': 'layers.0.mlp.up_proj', 'l1_down': 'layers.0.mlp.down_proj',
                   'l2_k': 'layers.1.self_attn.k_proj', 'l2_v': 'layers.1.self_attn.v_proj'}


def public_manifest():
    import skimage
    folder = Path(skimage.__file__).parent / 'data'
    rows = []
    for index, name in enumerate(PICTURES):
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(path)
        variants = [(QUESTIONS[index], ANSWERS[index]),
                    (QUESTIONS[index], 'The image is shown.'),
                    ('What can be seen in this photograph?', ANSWERS[index]),
                    ('Which objects are visible in the picture?', ANSWERS[index])]
        for variant, (question, answer) in enumerate(variants):
            rows.append({'family': index, 'variant': variant, 'path': str(path),
                         'sha256': file_hash(path), 'question': question, 'answer': answer,
                         'split': 'fit' if index < 8 else 'audit'})
    return rows


def capture_public(adapter, rows, names, output):
    """Same native client update; no fingerprint recomputation for every sample."""
    for index, row in enumerate(rows):
        with Image.open(row['path']) as im:
            image = adapter.prepare_image(im.convert('RGB')).unsqueeze(0)
        for text, capacity in ((row['question'], adapter.spec.question_length),
                               (row['answer'], adapter.spec.target_length)):
            if len(adapter.tokenizer.encode(text, add_special_tokens=False)) + 1 > capacity:
                raise ValueError('Public calibration text exceeds native capacity')
        batch = adapter.batch(image, [row['question']], [row['answer']])
        update = simulate_update(adapter, batch, adapter.training_spec)
        if index == 0:
            native = capture(adapter, batch, [], [])
            if native.tensors.keys() != update.keys() or not all(
                    torch.equal(native.tensors[n], v) for n, v in update.items()):
                raise ValueError('Public update differs from original capture')
            write_json(output / 'native-gate.json', {'all_tensors_exact': True,
                'tensors': len(update), 'model_fingerprint': native.model_fingerprint})
            del native
        value = {key: update[name].detach().cpu() for key, name in names.items()}
        write_tensors(output / f'{index:02d}.safetensors', value)
        write_json(output / f'{index:02d}.json', {**row, 'q_ids': batch.questions[0].tolist(),
            'a_ids': batch.targets[0].tolist(), 'native_gradient_count': len(update),
            'sha256': file_hash(output / f'{index:02d}.safetensors')})
        del update, value, batch
        print('public capture', index + 1, '/', len(rows), flush=True)


def defended(values, condition, seed):
    if condition == 'fedsgd-b1':
        return values
    if condition == 'noise':
        return GaussianNoiseDefense(.01).apply(values, generator=torch.Generator().manual_seed(seed))
    if condition == 'sparsify':
        return TopKSparsifyDefense(.1).apply(values)
    raise ValueError(condition)


@torch.no_grad()
def question_prior(adapter, pieces):
    means, traces = [], []
    lm, tokenizer = adapter.backend.language_model, adapter.tokenizer
    device = adapter.embedding().weight.device
    for template in PROMPTS:
        prompt = template.format(pieces=json.dumps(pieces, ensure_ascii=True))
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
        accumulated = torch.zeros(adapter.vocab_size, dtype=torch.float64, device=device)
        past, generated = None, []
        for _ in range(16):
            out = lm(input_ids=ids, past_key_values=past, use_cache=True, return_dict=True)
            probability = out.logits[0, -1].float().softmax(-1)
            if not torch.isfinite(probability).all():
                raise ValueError('Nonfinite public LM probability')
            accumulated += probability.double()
            token = int(probability.argmax())
            generated.append(token)
            if token == adapter.eos:
                break
            past, ids = out.past_key_values, torch.tensor([[token]], device=device)
        means.append(accumulated / len(generated))
        traces.append({'prompt': prompt, 'generated_ids': generated})
    return -torch.stack(means).mean(0).cpu(), traces


@torch.no_grad()
def first_pool(adapter, forbidden):
    tok = adapter.tokenizer
    ids = tok.encode('USER: ', add_special_tokens=False) + tok.encode('\n', add_special_tokens=False)
    logits = adapter.backend.language_model(input_ids=torch.tensor([ids], device=adapter.device),
        use_cache=False, return_dict=True).logits[0, -1].float()
    for index in forbidden:
        logits[index] = -torch.inf
    # Stable ID ordering resolves ties, matching the global candidate policy.
    pool = torch.argsort(logits, descending=True, stable=True)[:256]
    return pool.cpu(), logits.cpu()


@torch.no_grad()
def shallow_inputs(view, pool, chunk=16):
    """Hook actual module inputs, stop after the second block (no LM-head replay)."""
    decoder = view.decoder
    prefix_end = view.slots[0][2]
    prefix = view.embeddings[:, :prefix_end]
    captured, handles = {}, []
    for key, path in CONTEXT_MODULES.items():
        module = decoder.get_submodule(path)
        def hook(module, args, key=key):
            captured[key] = args[0].detach()
        handles.append(module.register_forward_pre_hook(hook))
    def forward(values):
        positions = torch.arange(values.shape[1], device=values.device)
        mask = decoder._update_causal_mask(None, values, positions, None, False)
        position_ids = positions[None]
        rotary = decoder.rotary_emb(values, position_ids)
        for layer in decoder.layers[:2]:
            values = layer(values, attention_mask=mask, position_ids=position_ids,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=positions, position_embeddings=rotary)[0]
    try:
        forward(prefix)
        public = {key: value[0].double().clone() for key, value in captured.items()}
        features = {key: [] for key in CONTEXT_MODULES}
        for start in range(0, len(pool), chunk):
            ids = pool[start:start + chunk].to(view.device)
            embeds = view.adapter.embedding()(ids)[:, None]
            values = torch.cat([prefix.expand(len(ids), -1, -1), embeds], dim=1)
            forward(values)
            for key in features:
                features[key].append(captured[key][:, -1].double().clone())
        return public, {key: torch.cat(value) for key, value in features.items()}
    finally:
        for handle in handles:
            handle.remove()
