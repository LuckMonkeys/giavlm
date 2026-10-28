"""Evaluator-only summaries after every property-geometry decision is committed."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from pathlib import Path

import numpy as np

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.generic_head_report import clustered_interval
from evaluation.image_property_scoring import classification_stats
from evaluation.property_geometry_study import ROOT, SOURCE, BANKS, verify
from evaluation.property_geometry import centered_blocks, class_matrix, SUPPRESSION


def plot_report(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = report['selection']['selected']
    methods = ['centered', 'centered_equal_tensor', selected['weighted'],
               'prototype_equal', selected['suppression'], selected['fisher'], 'model_only']
    names = ['Centered\nTop-3', 'Equal tensor\nTop-3', 'Selected\nweighting',
             'Prototype', 'Selected\nsuppression', 'Selected\nFisher', 'Model-only']
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, phase, title in zip(axes, ['reconstructed', 'oracle'],
                               ['Reconstructed text', 'Correct text'], strict=True):
        p = report['phases'][phase]
        x = np.arange(len(methods))
        generic = [p['generic_mean'][m]['accuracy'] for m in methods]
        medical = [p['banks']['3']['metrics'][m]['accuracy'] for m in methods]
        ax.bar(x - .19, generic, width=.36, color='#3a77ad', label='Generic: mean of 3 banks')
        ax.bar(x + .19, medical, width=.36, color='#d59250', label='Medical: 1 bank')
        ax.axhline(.5, color='#777777', linestyle='--', linewidth=1)
        for xx, yy in zip(x - .19, generic, strict=True):
            ax.text(xx, yy + .015, f'{yy:.2f}', ha='center', fontsize=8)
        ax.set(title=title, ylim=(0, 1.12), xticks=x, xticklabels=names)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Balanced accuracy')
    axes[0].legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    fig.suptitle('Head property: fixed 20 targets; candidate-only parameter selection')
    fig.tight_layout()
    path = Path('docs/experiments/figures/synthetic-head-property/geometry-accuracy.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def public_transfer_diagnostics(keys):
    """Use public heldout labels only; measure transfer beyond the fitting bank."""
    from scipy.stats import spearmanr
    correlations, retained = [], {f'k{k}_a{alpha:g}': [] for k, alpha in SUPPRESSION}
    for key in keys:
        path = ROOT / 'grams' / f'{key}.safetensors'
        g = read_tensors(path)['gram'].numpy()
        rows = read_json(path.with_suffix('.json'))['bank_rows']
        scores = read_json(ROOT / 'scores' / f'{key}.json')
        for b in range(3):
            fit = list(range(24 * b, 24 * (b + 1)))
            query = [i for i in range(72) if i not in fit]
            y = np.array([int(rows[i]['label'] == 'non-head') for i in query])
            labels = [int(rows[i]['label'] == 'non-head') for i in fit]
            k, d, qn, _, _ = centered_blocks(g, fit, query)
            a = class_matrix(labels)
            margin = d @ (a[:, 0] - a[:, 1])
            accuracy = ((margin < 0) == y[None, :]).mean(1)
            r = scores[str(b)]['diagnostics']['tensor_ratio']
            value = spearmanr(r, accuracy).statistic
            if np.isfinite(value):
                correlations.append(float(value))
            h = np.eye(len(fit)) - a[:, labels].T
            within = h @ k @ h.T
            eig, vec = np.linalg.eigh((within + within.transpose(0, 2, 1)) / 2)
            eig, vec = np.maximum(eig[:, ::-1], 0), vec[:, :, ::-1]
            valid = eig > np.maximum(eig[:, :1] * 1e-10, 1e-60)
            inv = np.divide(1, np.sqrt(eig), out=np.zeros_like(eig), where=valid)
            coeff = h.T @ (vec * inv[:, None, :])
            f, q = k @ coeff, d @ coeff
            means = a[:, y].T
            residual = np.maximum(qn - 2 * np.einsum('tqi,qi->tq', d, means)
                                  + np.einsum('qi,tij,qj->tq', means, k, means), 0)
            rz = q - np.einsum('qi,tik->tqk', means, f)
            for rank, alpha in SUPPRESSION:
                after = np.maximum(residual - alpha * (2 - alpha) * np.square(rz[:, :, :rank]).sum(2), 0)
                retained[f'k{rank}_a{alpha:g}'].append(float(after.sum() / max(residual.sum(), 1e-60)))
    return {'tensor_ratio_vs_heldout_accuracy_spearman': float(np.mean(correlations)),
            'correlation_contexts': len(correlations),
            'heldout_within_variation_retained': {m: float(np.mean(v)) for m, v in retained.items()},
            'limit': 'Heldout seeds in the same photographic prompt families; not unseen domains'}


def credit(scores, labels, classes):
    return [float(s is not None and classes[int(np.argmin(s))] == label)
            for s, label in zip(scores, labels, strict=True)]


def summarize():
    complete = read_json(ROOT / 'predictions_complete.json')
    verify(complete['hashes'])
    plan = read_json(ROOT / 'plan.json')
    verify(plan['frozen_hashes'])
    selection = read_json(ROOT / 'selection.json')
    if complete['selection_sha256'] != file_hash(ROOT / 'selection.json'):
        raise ValueError('Selection changed')
    # This is the first use of target labels in the new experiment.
    truth = read_json(SOURCE / 'private/targets.json')
    labels = [r['label'] for r in truth]
    classes = plan['classes']
    original = read_json(SOURCE / 'report.json')
    report = {'selection': selection, 'phases': {}, 'input_checks': {},
              'statistical_unit': '20 reused targets; three repeated generic banks',
              'scope': 'diagnostic; no independent test cohort',
              'cost': read_json(ROOT / 'collect_complete.json')}
    del report['cost']['hashes']
    for phase in plan['phases']:
        results, correct = {}, {}
        for b in range(4):
            preds = [read_json(ROOT / 'predictions' / f'{phase}-{b}-{i:02d}.json') for i in range(20)]
            old = [read_json(BANKS[b] / phase / 'predictions' / f'{i:05d}-{phase}.json') for i in range(20)]
            for p, previous in zip(preds, old, strict=True):
                if p['text_sha256'] != previous['text_sha256']:
                    raise ValueError('Shared text differs from prior experiment')
            scores = {m: [p['class_scores'][m] for p in preds] for m in preds[0]['class_scores']}
            scores.update({m: [p[m] for p in old] for m in ['model_only', 'text_only']})
            metrics = {m: classification_stats(labels, s, classes) for m, s in scores.items()}
            correct[b] = {m: credit(s, labels, classes) for m, s in scores.items()}
            max_error = 0.
            for m in ['global', 'centered', 'equal_tensor', 'centered_equal_tensor']:
                prev = np.array([p['class_scores'][m] for p in old])
                current = np.array(scores[m])
                max_error = max(max_error, float(np.max(np.abs(current - prev))))
                if np.max(np.abs(current - prev)) > 3e-5:
                    raise ValueError(f'Replayed baseline drift: {phase} bank {b} {m}')
            wrong = {}
            for m in scores:
                if m in ['model_only', 'text_only']:
                    continue
                per_target = []
                for i, pred in enumerate(preds):
                    wrong_scores = [s for j, s in enumerate(pred['all_upload_scores'][m]) if i != j]
                    per_target.append(float(np.mean(credit(wrong_scores, [labels[i]] * 19, classes))))
                wrong[m] = float(np.mean(per_target))
            groups = {name: [i for i, t in enumerate(truth) if t['modality'] == name]
                      for name in ['CT', 'MRI', 'X-Ray']}
            groups.update({f'head_terms_{v}': [i for i, t in enumerate(truth) if t['head_terms_present'] == v]
                           for v in [False, True]})
            groups.update({f'joint_em_{v}': [i for i, t in enumerate(original['text_recovery']['per_target'])
                                           if t['joint_em'] == v] for v in [False, True]})
            results[str(b)] = {'metrics': metrics, 'wrong_gradient_accuracy': wrong,
                              'max_old_score_error': max_error,
                              'groups': {group: {m: classification_stats([labels[i] for i in ids],
                                          [s[i] for i in ids], classes) for m, s in scores.items()}
                                         for group, ids in groups.items()}}
        mean = {}
        for m in correct[0]:
            means = {metric: float(np.mean([results[str(b)]['metrics'][m][metric] for b in range(3)]))
                     for metric in ['accuracy', 'balanced_accuracy', 'macro_f1']}
            means['head_recall'] = float(np.mean([results[str(b)]['metrics'][m]['recall']['head'] for b in range(3)]))
            means['head_auroc'] = float(np.mean([results[str(b)]['metrics'][m]['class_auroc']['head'] for b in range(3)]))
            means['coverage'] = float(np.mean([1 - results[str(b)]['metrics'][m]['invalid'] / 20 for b in range(3)]))
            means['accuracy_interval'] = clustered_interval([correct[b][m] for b in range(3)], labels)
            # Cluster entire bank-repeated target, preserving pairs.
            for baseline in ['centered', 'centered_equal_tensor', 'prototype_equal', 'model_only']:
                means[f'paired_vs_{baseline}'] = clustered_interval(
                    [np.asarray(correct[b][m]) - np.asarray(correct[b][baseline]) for b in range(3)], labels)
            if m not in ['model_only', 'text_only']:
                means['wrong_gradient_accuracy'] = float(np.mean([results[str(b)]['wrong_gradient_accuracy'][m]
                                                                 for b in range(3)]))
            mean[m] = means
        report['phases'][phase] = {'banks': results, 'generic_mean': mean}
    # Candidate-only geometry and fitted-weight stability, grouped per context/bank.
    refs = read_json(ROOT / 'references.json')
    keys = sorted({r['text_sha256'] for r in refs if r['phase'] == 'reconstructed'})
    ratios, concentrations, diagnostics = [], [], {}
    tensor_profiles = {str(b): [] for b in range(4)}
    for key in keys:
        data = read_json(ROOT / 'scores' / f'{key}.json')
        for b in range(4):
            d = data[str(b)]['diagnostics']
            w = np.array(d['weights_b0.5'])
            tensor_profiles[str(b)].append(w)
            concentrations.append({'bank': b, 'top10_mass': float(np.sort(w)[-10:].sum()),
                                   'effective_tensors': float(1 / np.square(w).sum())})
            for m, v in d['suppression'].items():
                diagnostics.setdefault(f'{b}:{m}', []).append(v)
        from scipy.stats import spearmanr
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            ratios.append(float(spearmanr(data[str(a)]['diagnostics']['tensor_ratio'],
                                         data[str(b)]['diagnostics']['tensor_ratio']).statistic))
    report['geometry'] = {name: {'between_retained': float(np.mean([r['between_retained'] for r in rows])),
                                 'within_retained': float(np.mean([r['within_retained'] for r in rows])),
                                 'query_energy_retained': float(np.mean([np.mean(r['query_energy_retained']) for r in rows]))}
                          for name, rows in diagnostics.items()}
    report['weights'] = {'ratio_rank_spearman_mean': float(np.mean(ratios)),
                         'banks': {str(b): {m: float(np.mean([r[m] for r in concentrations if r['bank'] == b]))
                                           for m in ['top10_mass', 'effective_tensors']} for b in range(4)}}
    report['weights']['tensor_names'] = read_json(ROOT / 'grams' / f'{keys[0]}.json')['tensor_names']
    report['weights']['mean_weights_by_bank'] = {b: np.mean(v, axis=0).tolist()
                                                for b, v in tensor_profiles.items()}
    report['public_transfer'] = public_transfer_diagnostics(keys)
    report['input_checks'] = {'old_scores_reproduced_atol': 3e-5, 'private_labels_after_commit': True,
                               'predictions': complete['predictions'], 'prediction_hashes_verified': True,
                               'no_new_target_text_search': True}
    write_json(ROOT / 'report.json', report)
    public = {k: v for k, v in report.items() if k != 'cost'}
    public['cost'] = {k: v for k, v in report['cost'].items() if k != 'references_sha256'}
    out = Path('docs/experiments/data/synthetic-head-property/geometry-aggregate.json')
    write_json(out, public)
    figure = plot_report(report)
    write_json(ROOT / 'report_complete.json', {'report_sha256': file_hash(ROOT / 'report.json'),
               'aggregate_sha256': file_hash(out), 'aggregate': str(out),
               'figure': str(figure), 'figure_sha256': file_hash(figure)})
    print({'selected': selection['selected'], 'generic': {p: {m: v['accuracy'] for m, v in r['generic_mean'].items()}
           for p, r in report['phases'].items()}})
    return report


def publish(report):
    """Export a phase draft without overwriting the consolidated experiment page.

    The canonical record joins geometry scoring and concept probing; one phase
    cannot regenerate it alone. Frozen historical snapshots remain unchanged.
    """
    canonical = 'docs/experiments/2026-10-28-image-property-ideas.md'
    lines = ['# 候选统计评分导出', '', f'统一实验记录：{canonical}', '',
             '| Method | Reconstructed accuracy | Correct accuracy |', '|---|---:|---:|']
    for name in report['phases']['reconstructed']['generic_mean']:
        rec = report['phases']['reconstructed']['generic_mean'][name]['accuracy']
        oracle = report['phases']['oracle']['generic_mean'][name]['accuracy']
        lines.append(f'| {name} | {rec:.3f} | {oracle:.3f} |')
    draft = ROOT / 'scoring-report-draft.md'
    draft.write_text('\n'.join(lines) + '\n')
    write_json(ROOT / 'draft_export.json', {'draft': str(draft),
               'draft_sha256': file_hash(draft), 'canonical_record': canonical,
               'documentation_written': False})


if __name__ == '__main__':
    try:
        publish(summarize())
    except BaseException:
        import traceback
        write_json(ROOT / 'report_failed.json', {'traceback': traceback.format_exc()})
        raise
