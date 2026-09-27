"""Offline summaries: bootstrap images, never individual perturbations or checkpoints."""
from itertools import combinations

import numpy as np

from core.artifacts import read_json, write_json
from evaluation.discriminability_metrics import (PRIMARY, RADII, THRESHOLDS, bootstrap,
                                                candidate_statistics)


def strata(rows):
    base = [r for r in rows if r['family'] not in {'truth', 'historical'}]
    result = {'all': base, 'natural': [r for r in base if r['family'].startswith('natural_')],
              'synthetic': [r for r in base if not r['family'].startswith('natural_')],
              'historical': [r for r in rows if r['family'] == 'historical'],
              'near': [r for r in base if r['image']['mse'] <= 1e-3],
              'far': [r for r in base if r['image']['mse'] >= 1e-2]}
    result.update({family: [r for r in base if r['family'] == family]
                   for family in sorted({r['family'] for r in base})})
    return result


def summarize_images(images):
    metrics = sorted({metric for value in images.values() for metric in value})
    result = {}
    for metric in metrics:
        groups = sorted({group for value in images.values() for group in value.get(metric, {})})
        result[metric] = {}
        for group in groups:
            stats = [value[metric][group] for value in images.values() if metric in value]
            out = {'near_win': bootstrap([s['near_win']['value'] for s in stats]),
                   'spearman': {key: bootstrap([s['spearman'][key]['value'] for s in stats])
                                for key in ['mse', 'one_minus_ssim', 'lpips']}}
            for kind, keys in [('thresholds', [str(t) for t in THRESHOLDS]),
                               ('bottom_fraction', ['0.01', '0.05', '0.1', '0.2'])]:
                out[kind] = {key: {field: bootstrap([s[kind][key][field] for s in stats])
                                   for field in ['far_fraction', 'mse_median', 'mse_p90']}
                             for key in keys}
                for key in keys:
                    out[kind][key]['accepted_candidates'] = sum(s[kind][key]['count'] for s in stats)
            result[metric][group] = out
    return result


def paired_comparisons(conditions):
    result = {}
    for left, right in combinations(conditions, 2):
        a, b = conditions[left]['images'], conditions[right]['images']
        common = sorted(set(a) & set(b))
        comparisons = {}
        for metric in PRIMARY:
            for statistic in ['near_win', 'spearman_mse']:
                def value(record):
                    s = record[metric]['all']
                    return s['near_win']['value'] if statistic == 'near_win' else s['spearman']['mse']['value']
                delta = [value(b[k]) - value(a[k]) for k in common
                         if value(a[k]) is not None and value(b[k]) is not None]
                comparisons[f'{metric}/{statistic}'] = bootstrap(delta)
        result[f'{right}_minus_{left}'] = comparisons
    return result


def direction_summary(directory):
    per_image = {}
    for path in sorted(directory.glob('image-*/directions/*.json')):
        record = read_json(path)
        image = path.parent.parent.name
        for direction in ['gradient', 'sign']:
            key = f'{record["control"]}/{record["objective"]}/{direction}/{record["alpha"]}'
            per_image.setdefault(key, {})[image] = record[f'{direction}_alignment']
            for step in record['steps']:
                if step['direction'] == direction:
                    success = float(step['loss_decreased'] and step['mse_decreased'])
                    per_image.setdefault(f'{key}/step-{step["step_rms"]}', {})[image] = success
    return {key: bootstrap(list(values.values())) for key, values in per_image.items()}


def report(root, study, cohort, precision):
    conditions, scatter = {}, {}
    for condition in study['spec']['conditions']:
        name = condition['name']
        directory = root / 'records' / cohort / precision / name
        marker = read_json(directory / 'score-complete.json')
        direction_marker = read_json(directory / 'directions-complete.json')
        if marker['signature'] != study['signature'] or marker['status'] != 'completed':
            raise ValueError('Report requires completed matching scoring jobs')
        if (direction_marker['signature'] != study['signature'] or
                direction_marker['status'] != 'completed'):
            raise ValueError('Report requires completed direction jobs')
        images, points = {}, []
        for index_path in sorted(directory.glob('image-*/candidate_index.json')):
            index = read_json(index_path)
            if index['signature'] != study['signature']:
                raise ValueError('Candidate index fingerprint differs')
            rows = []
            for identifier in index['candidate_ids']:
                saved = read_json(index_path.parent / 'scores' / f'{identifier}.json')
                rows.append({k: saved[k] for k in ['family', 'losses', 'image']})
            subsets = strata(rows)
            metrics = sorted({k for row in rows for k in row['losses']})
            images[index_path.parent.name] = {
                metric: {group: candidate_statistics(values, metric)
                         for group, values in subsets.items()} for metric in metrics}
            points.extend(rows)
        if len(images) != marker['images']:
            raise ValueError('Report is missing completed image records')
        conditions[name] = {'images': images, 'summary': summarize_images(images),
                            'directions': direction_summary(directory), 'resources': marker}
        # Resource records contain no source paths, public text, or private identifiers.
        scatter[name] = points
    result = {'schema_version': 1, 'uses_private_reference': True,
              'cohort': cohort, 'precision': precision, 'signature': study['signature'],
              'note': 'Finite candidate-set diagnostics; no guarantee of unique recovery.',
              'conditions': conditions, 'paired': paired_comparisons(conditions)}
    output = root / 'reports' / cohort / precision
    write_json(output / 'report.json', result)
    plots(output, conditions, scatter)
    lines = ['# Gradient discriminability', '',
             'Private-reference diagnostic; not an attack benchmark.', '',
             f'Cohort: {cohort}; precision: {precision}.', '',
             '| Condition | Metric | Image count | Near-win mean | 95% CI |',
             '|---|---|---:|---:|---|']
    for condition, data in conditions.items():
        for metric in PRIMARY:
            value = data['summary'][metric]['all']['near_win']
            lines.append(f'| {condition} | {metric} | {value["images"]} | '
                         f'{value["mean"]} | {value["ci95"]} |')
    lines += ['', 'Bootstrap units are images; missing near/far sets remain undefined.',
              'All candidate-family results and paired differences are in report.json.',
              'No observed counterexample does not establish identifiability.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return result


def plots(output, conditions, scatter):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = len(conditions)

    def finish(fig, name):
        fig.suptitle('Private-reference diagnostic')
        fig.tight_layout()
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'{name}.{suffix}', dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4), squeeze=False)
    for ax, (name, rows) in zip(axes[0], scatter.items()):
        for family in sorted({r['family'] for r in rows} - {'truth'}):
            values = [r for r in rows if r['family'] == family and r['losses']['cosine'] is not None]
            ax.scatter([r['image']['mse'] for r in values],
                       [r['losses']['cosine'] for r in values], s=7, alpha=.4, label=family)
        ax.set(xscale='log', xlabel='Image MSE', ylabel='Cosine loss', title=name)
    axes[0][-1].legend(fontsize=6)
    finish(fig, 'distance_loss')

    fig, ax = plt.subplots(figsize=(max(7, count * 2), 4))
    for i, metric in enumerate(PRIMARY):
        stats = [data['summary'][metric]['all']['near_win'] for data in conditions.values()]
        x = np.arange(count) + (i - 1.5) * .18
        y = [s['mean'] if s['mean'] is not None else np.nan for s in stats]
        ax.scatter(x, y, label=metric)
        for position, s in zip(x, stats):
            if s['ci95']:
                ax.vlines(position, *s['ci95'])
    ax.axhline(.5, color='gray', linestyle='--')
    ax.set(xticks=np.arange(count), xticklabels=list(conditions), ylabel='Near-win probability')
    ax.legend(fontsize=7)
    finish(fig, 'near_win')

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4), squeeze=False)
    for ax, (name, data) in zip(axes[0], conditions.items()):
        for metric in PRIMARY:
            threshold = data['summary'][metric]['all']['thresholds']
            y = [threshold[str(t)]['far_fraction']['mean'] for t in THRESHOLDS]
            ax.plot(THRESHOLDS, [v if v is not None else np.nan for v in y], label=metric)
        ax.set(xscale='log', xlabel='Loss threshold', ylabel='Far-image fraction', title=name)
    axes[0][-1].legend(fontsize=7)
    finish(fig, 'low_loss_errors')

    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4), squeeze=False)
    for ax, (name, data) in zip(axes[0], conditions.items()):
        for kind in ['cosine', 'relative_l2']:
            for direction in ['gradient', 'sign']:
                values = [data['directions'].get(f'correct/{kind}/{direction}/{r}', {}).get('mean')
                          for r in RADII]
                ax.plot(RADII, [v if v is not None else np.nan for v in values],
                        label=f'{kind}/{direction}')
        ax.set(xscale='log', xlabel='Mix alpha', ylabel='Direction alignment', title=name)
    axes[0][-1].legend(fontsize=6)
    finish(fig, 'local_directions')

    groups = sorted({k for data in conditions.values() for k in data['summary'] if '/' in k})
    if groups:
        fig, ax = plt.subplots(figsize=(max(8, len(groups) / 4), max(3, count)))
        values = [[data['summary'].get(k, {}).get('all', {}).get('spearman', {}).get(
            'mse', {}).get('mean') for k in groups] for data in conditions.values()]
        matrix = np.array([[v if v is not None else np.nan for v in row] for row in values])
        plot = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm', aspect='auto')
        ax.set(xticks=range(len(groups)), xticklabels=groups,
               yticks=range(count), yticklabels=list(conditions))
        ax.tick_params(axis='x', labelrotation=90, labelsize=5)
        fig.colorbar(plot, ax=ax, label='Image-level mean Spearman (MSE)')
        finish(fig, 'module_discriminability')
