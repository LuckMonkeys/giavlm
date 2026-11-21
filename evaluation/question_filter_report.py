"""Aggregate-only report; never imports a target dataset or private token file."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.artifacts import file_hash, read_json, write_json


METHODS = [('baseline', 'Residual'), ('b_anchor', 'B anchor'),
           ('bridge_q_blind', 'Blind question prior'), ('bridge_q_matched', 'Answer-conditioned prior'),
           ('mix_energy_kv', 'K/V energy'), ('mix_ridge', 'Masked ridge'),
           ('mix_b_forward', 'B forward probe'), ('mix_b_input', 'B input probe')]
COLORS = dict(zip((k for k, _ in METHODS),
    ('#667685', '#287da3', '#b7c4cc', '#d28c3f', '#4e998a', '#aa6c8e', '#8c8cba', '#a4aa59'),
    strict=True))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    source = root / 'summary.json'
    summary = read_json(source)
    if summary['status'] != 'completed' or not summary['all_scores_before_truth']:
        raise ValueError('Cannot report incomplete or uncommitted target predictions')
    dest = root / 'report'
    dest.mkdir(exist_ok=False)
    published = {k: summary[k] for k in ('aggregate', 'by_length', 'first', 'first_informative', 'gates',
                                        'paired', 'independent_validation', 'source_sha256')}
    published.update(contains_private_tokens=False, uncommitted=True,
                     source_summary_sha256=file_hash(source), all_scores_before_truth=True)
    write_json(dest / 'aggregate.json', published)
    for name in ('aggregate', 'by_length', 'first', 'first_informative', 'paired'):
        write_csv(dest / (name + '.csv'), summary[name])
    lookup = {(r['condition'], r['image'], r['method']): r for r in summary['aggregate']}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for i, condition in enumerate(('noise', 'sparsify')):
        active = [(key, label) for key, label in METHODS if (condition, 'random', key) in lookup]
        for j, (metric, title) in enumerate((('exclusive_recall', 'Question-exclusive candidate recall'),
                                           ('answer_recall', 'Answer candidate recall'))):
            ax = axes[i, j]
            values = [lookup[condition, 'random', key][metric] for key, _ in active]
            y = np.arange(len(active))
            ax.barh(y, values, color=[COLORS[k] for k, _ in active])
            ax.set_yticks(y, [label for _, label in active], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.08)
            ax.set_title(('Noise' if condition == 'noise' else 'Top-k 10%') + ' | ' + title)
            for k, value in enumerate(values):
                ax.text(value + .012, k, f'{value:.3f}', va='center', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SLAKE30 | Random image | Fixed candidate budget 109', fontsize=15)
    fig.savefig(dest / 'question-answer.png', dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, condition in zip(axes, ('noise', 'sparsify'), strict=True):
        rows = [r for r in summary['aggregate'] if r['condition'] == condition and r['image'] == 'random'
                and (r['method'].startswith('quota_') or r['method'] == 'baseline')]
        y = np.arange(len(rows))
        ax.barh(y-.16, [r['question_recall'] for r in rows], height=.3, label='Question')
        ax.barh(y+.16, [r['answer_recall'] for r in rows], height=.3, label='Answer')
        ax.set_yticks(y, [r['method'].replace('quota_', '') for r in rows], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title('Noise' if condition == 'noise' else 'Top-k 10%')
        ax.legend()
    fig.suptitle('Budget allocation | Candidate recall', fontsize=15)
    fig.savefig(dest / 'budget.png', dpi=155)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for ax, condition in zip(axes, ('noise', 'sparsify'), strict=True):
        rows = [r for r in summary['first_informative'] if r['condition'] == condition and
                r['image'] == 'random' and not r['method'].startswith('wrong_')]
        y = np.arange(len(rows))
        ax.barh(y, [r['hit_5'] for r in rows], color='#287da3')
        ax.set_yticks(y, [r['method'] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title('Noise' if condition == 'noise' else 'Top-k 10%')
    fig.suptitle('First question token | Top-5 hit | Fixed proposal pool 256', fontsize=15)
    fig.savefig(dest / 'first-token.png', dpi=155)
    plt.close(fig)
    lines = ['# Question-filter Stage-1 results', '', 'Development diagnostics; not sequence recovery.', '',
             '| Defense | Method | Image | Q recall | Q-exclusive recall | A recall | Joint recall | Joint full |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for condition in ('noise', 'sparsify'):
        for image in ('random', 'gt'):
            for key, label in METHODS:
                row = lookup.get((condition, image, key))
                if row:
                    lines.append(f"| {condition} | {label} | {image} | {row['question_recall']:.4f} | "
                        f"{row['exclusive_recall']:.4f} | {row['answer_recall']:.4f} | "
                        f"{row['joint_recall']:.4f} | {row['joint_full_coverage_count']}/{row['n']} |")
    (dest / 'summary.md').write_text('\n'.join(lines) + '\n')
    files = {str(p.relative_to(dest)): file_hash(p) for p in sorted(dest.iterdir()) if p.is_file()}
    write_json(dest / 'manifest.json', {'files': files, 'report_code_sha256': file_hash(Path(__file__)),
        'source_summary_sha256': file_hash(source), 'uncommitted': True, 'contains_private_tokens': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    report(parser.parse_args().output)
