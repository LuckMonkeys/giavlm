"""Post-commit E1 token-gain audit; detailed token identities remain in outputs."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import statistics
import time
import traceback

import torch
from safetensors import safe_open

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.question_filter_math import rank


WORD_GROUPS = {
    'Question word': 'what which where when why how who whom whose'.split(),
    'Function word': ("a an the this that these those it its they their there here is are was were be been being "
        "do does did can could would should will may might have has had of in on at to for from with without "
        "by as and or but if than then any some all not no yes i you your we our he she his her them "
        "most more least less only other same such many much please don't doesn't isn't aren't").split(),
    'General image term': ('image images picture pictures photo photos photograph photographs shown show '
        'shows showing seen see visible depicted displayed display present appears appear object objects scene').split(),
    'General VQA task term': ('type kind color colour size shape number location located position region '
        'side view plane area part parts describe identify indicate').split(),
    'Medical task term': ('body organ organs modality modalities medical anatomy anatomical imaging '
        'abnormality abnormalities normal abnormal').split(),
    'Medical modality': ('ct mri xray x-ray radiograph radiographs scan scans ultrasound t1 t2 weighted').split(),
    'Anatomical content': ('brain head lung lungs liver kidney kidneys spleen heart abdomen abdominal chest '
        'thorax pelvis bone bones vertebra vertebrae spine breast neck arm leg stomach pancreas intestine '
        'bowel aorta vein artery vessel arterial venous hepatic renal pulmonary femur hip skull lobe').split(),
    'Pathology content': ('lesion lesions tumor tumour cancer fracture pneumonia edema effusion hemorrhage '
        'bleeding nodule nodules opacity enlarged enlargement atrophy mass masses cyst cysts stone stones').split(),
    'Spatial content': ('left right anterior posterior superior inferior upper lower lateral medial '
                        'bilateral axial coronal sagittal proximal distal').split(),
}
WORD_LOOKUP = {word: group for group, words in WORD_GROUPS.items() for word in words}
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|[0-9]+|[^\w\s]")
CONTRASTS = (
    ('blind_vs_residual', 'bridge_q_blind', 'baseline'),
    ('conditioned_vs_residual', 'bridge_q_matched', 'baseline'),
    ('blind_vs_b_anchor', 'bridge_q_blind', 'b_anchor'),
    ('conditioned_vs_b_anchor', 'bridge_q_matched', 'b_anchor'),
    ('conditioned_vs_blind', 'bridge_q_matched', 'bridge_q_blind'),
    ('conditioned_vs_wrong', 'bridge_q_matched', 'bridge_q_wrong'),
)
GENERIC = {'Question word', 'Function word', 'General image term'}


def word_category(word):
    if word.isdecimal():
        return 'Number'
    if not any(c.isalnum() for c in word):
        return 'Punctuation'
    return WORD_LOOKUP.get(word.casefold(), 'Other lexical content')


def contextual_words(text, offsets):
    """Map BPE pieces to complete lexical contexts; e.g. I in MRI is not a pronoun."""
    spans = [(m.start(), m.end(), m.group()) for m in WORD_PATTERN.finditer(text)]
    result = []
    for start, stop in offsets:
        words = {word for left, right, word in spans if max(start, left) < min(stop, right)}
        result.append(words)
    return result


def changes(reference, current, previous):
    return (reference & (current - previous), reference & (previous - current))


def source_map(base_order, b_order, lm_order, ambiguous):
    result = {}
    for name, order, count in (('original60', base_order, 60), ('b20', b_order, 20), ('lm20', lm_order, 20)):
        before = len(result)
        for token in order:
            if token not in result:
                result[token] = name
            if len(result) == before + count:
                break
        if len(result) != before + count:
            raise ValueError('Insufficient observable candidate source')
    if set(result) & ambiguous:
        raise ValueError('Ambiguous IDs in informative sources')
    result.update({token: 'ambiguous9' for token in ambiguous})
    return result


def load_bundle(path):
    meta = read_json(path.with_suffix('.json'))
    if file_hash(path) != meta['sha256']:
        raise ValueError('Committed artifact changed')
    return read_tensors(path), meta


def order_from_positions(positions):
    ids = torch.nonzero(positions > 0).flatten().tolist()
    return sorted(ids, key=lambda token: int(positions[token]))


def csv_file(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    from transformers import AutoTokenizer

    final = read_json(root / 'finalized.json')
    if final['status'] != 'completed' or file_hash(root / 'summary.json') != final['summary_sha256']:
        raise ValueError('Prior target evaluation is incomplete or changed')
    plan = read_json(root / 'plan.json')
    model = read_json(Path(plan['model']) / 'model.json')['model']
    tokenizer = AutoTokenizer.from_pretrained(model['name'], revision=model['revision'],
                                             local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError('Exact word attribution needs fast-tokenizer offsets')
    groups, token_counts, token_origins, token_context, events = [], defaultdict(Counter), defaultdict(Counter), defaultdict(set), []
    alignment, committed = {}, []
    for condition in ('fedsgd-b1', 'noise', 'sparsify'):
        for image in ('random', 'gt'):
            for index in range(30):
                job = f'{condition}-{index:02d}'
                path = root / 'candidates' / image / (job + '.safetensors')
                candidate, meta = load_bundle(path)
                positions, _ = load_bundle(root / 'ranks' / image / (job + '.safetensors'))
                committed.append({'group': job + ':' + image, 'candidate_sha256': meta['sha256']})
                forbidden = set(meta['forbidden_ids'])
                original_path = Path(plan['source']) / 'runs' / job / 'attacks' / image / 'token_candidates.safetensors'
                if file_hash(original_path) != plan['baselines'][job + ':' + image]:
                    raise ValueError('Original baseline changed')
                flags = read_tensors(original_path)['ambiguous']
                ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
                b_path = Path(plan['previous']) / 'scores' / job / 'scores.safetensors'
                with safe_open(str(b_path), framework='pt', device='cpu') as stream:
                    b_order = rank(stream.get_tensor('l31_down'), flags, forbidden)
                origins = {}
                for variant, prior in (('bridge_q_blind', 'q_blind'), ('bridge_q_matched', 'q_matched')):
                    origins[variant] = source_map(order_from_positions(positions['baseline']), b_order,
                                                  order_from_positions(positions[prior]), ambiguous)
                    if set(origins[variant]) != set(candidate[variant].tolist()):
                        raise ValueError('Source attribution does not reproduce the committed candidate')
                # References enter only after the existing candidates and their sources were checked.
                refs = read_tensors(Path(plan['source']) / 'runs' / job / 'captures' / image /
                                    'private/text_tokens.safetensors')
                q_ids = [t for t in refs['questions'][0].tolist() if t not in forbidden]
                a_ids = [t for t in refs['targets'][0].tolist() if t not in forbidden]
                q, a = set(q_ids), set(a_ids)
                key = tuple(q_ids)
                if key not in alignment:
                    text = tokenizer.decode(q_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                    exact = encoded['input_ids'] == q_ids
                    contexts = contextual_words(text, encoded['offset_mapping']) if exact else [set()] * len(q_ids)
                    by_token = defaultdict(set)
                    for token, words in zip(q_ids, contexts, strict=True):
                        by_token[token].update(words)
                    alignment[key] = (exact, by_token)
                exact, words = alignment[key]
                for contrast, method, control in CONTRASTS:
                    selected, before = set(candidate[method].tolist()), set(candidate[control].tolist())
                    for part, truth in (('question', q), ('exclusive', q-a), ('answer', a), ('joint', q|a)):
                        gained, lost = changes(truth, selected, before)
                        group = (condition, image, contrast, part)
                        after_hit, before_hit = len(truth & selected), len(truth & before)
                        if after_hit - before_hit != len(gained) - len(lost):
                            raise ValueError('Gain/loss identity failed')
                        groups.append({'condition': condition, 'image': image, 'contrast': contrast,
                            'part': part, 'index': index, 'reference_count': len(truth), 'new_hits': len(gained),
                            'lost_hits': len(lost), 'net_hits': len(gained)-len(lost),
                            'before_recall': before_hit/len(truth) if truth else None,
                            'after_recall': after_hit/len(truth) if truth else None,
                            'lm_owned_new_hits': sum(origins[method].get(t)=='lm20' for t in gained)})
                        for direction, tokens in (('gain', gained), ('loss', lost)):
                            for token in tokens:
                                token_counts[group + (direction,)][token] += 1
                                origin = origins[method].get(token, 'removed') if direction == 'gain' else 'removed'
                                token_origins[group + (direction,)][token, origin] += 1
                                context_words = words.get(token, set()) if token in q else set()
                                token_context[token].update(context_words)
                                categories = {word_category(word) for word in context_words}
                                category = (next(iter(categories)) if len(categories)==1 else
                                            'Mixed word contexts' if categories else 'Unaligned or answer-only')
                                events.append({'condition': condition, 'image': image, 'contrast': contrast,
                                    'part': part, 'direction': direction, 'index': index, 'token_id': token,
                                    'piece': tokenizer.convert_ids_to_tokens(token), 'category': category,
                                    'word_forms': ' | '.join(sorted(context_words)), 'source': origin})
    buckets = defaultdict(list)
    for row in groups:
        buckets[row['condition'], row['image'], row['contrast'], row['part']].append(row)
    summary = []
    for group, values in buckets.items():
        condition, image, contrast, part = group
        gains = token_counts[group + ('gain',)]
        losses = token_counts[group + ('loss',)]
        row = dict(condition=condition, image=image, contrast=contrast, part=part, n=len(values),
            new_hits=sum(v['new_hits'] for v in values), lost_hits=sum(v['lost_hits'] for v in values),
            net_hits=sum(v['net_hits'] for v in values), new_unique_ids=len(gains), lost_unique_ids=len(losses),
            lm_owned_new_hits=sum(v['lm_owned_new_hits'] for v in values))
        valid = [v for v in values if v['reference_count']]
        row['before_recall'] = statistics.fmean(v['before_recall'] for v in valid) if valid else None
        row['after_recall'] = statistics.fmean(v['after_recall'] for v in valid) if valid else None
        summary.append(row)
    categories = Counter((e['condition'], e['image'], e['contrast'], e['part'], e['direction'], e['category']) for e in events)
    category_rows = [dict(zip(('condition','image','contrast','part','direction','category'), key, strict=True), count=count)
                     for key, count in sorted(categories.items())]
    detailed = []
    for group, counts in token_counts.items():
        for token, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
            detailed.append(dict(zip(('condition','image','contrast','part','direction'), group, strict=True),
                token_id=token, piece=tokenizer.convert_ids_to_tokens(token),
                decoded=tokenizer.decode([token], clean_up_tokenization_spaces=False),
                sample_count=count, lm_owned_count=token_origins[group][token, 'lm20'],
                word_forms=' | '.join(sorted(token_context[token]))))
    csv_file(output / 'token-frequency.csv', detailed)
    csv_file(output / 'sample-token-events.csv', events)
    csv_file(output / 'summary.csv', summary)
    csv_file(output / 'categories.csv', category_rows)
    write_json(output / 'aggregate.json', {'summary': summary, 'categories': category_rows,
        'alignment_unique_questions': len(alignment), 'alignment_exact': sum(v[0] for v in alignment.values()),
        'generic_definition': sorted(GENERIC), 'classification': WORD_GROUPS,
        'unit': 'one deduplicated token ID per sample; no sequence reconstruction',
        'contains_private_tokens': False, 'uncommitted': True})
    write_json(output / 'manifest.json', {'files': {p.name: file_hash(p) for p in sorted(output.iterdir())
        if p.is_file() and p.name not in ('state.json', 'manifest.json')},
        'source_summary_sha256': final['summary_sha256'], 'committed_candidate_groups': committed,
        'code_sha256': file_hash(Path(__file__)), 'contains_private_tokens': True,
        'note': 'Detailed CSV token pieces/contexts and sample-level gains stay local; aggregate contains no private tokens',
        'private_reference_used_for_scoring': False, 'no_new_predictions': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    state = args.output / 'state.json'
    write_json(state, {'status': 'running', 'started': time.time()})
    try:
        analyze(args.root, args.output)
        write_json(state, {'status': 'completed', 'finished': time.time()})
    except BaseException:
        write_json(state, {'status': 'failed', 'traceback': traceback.format_exc(), 'finished': time.time()})
        raise


if __name__ == '__main__':
    main()
