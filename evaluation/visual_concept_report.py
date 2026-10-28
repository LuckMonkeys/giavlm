"""Post-commit evaluation of public-image concept geometry."""
from pathlib import Path

import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import classification_stats, binary_auc
from evaluation.visual_concept_probe import ROOT, BANKS, verify


def interval(labels, scores, clusters=None, repeats=5000):
    y = np.asarray([v == 'head' for v in labels])
    if any(s is None for s in scores) or not y.any() or y.all():
        return {'balanced_accuracy_ci95': None, 'head_auc_ci95': None}
    s = np.asarray(scores)
    margin = s[:, 1] - s[:, 0]
    correct = (s.argmin(1) == (~y).astype(int)).astype(float)
    rng = np.random.default_rng(20261028)
    if clusters is None:
        indices = [rng.choice(np.flatnonzero(y == value), (repeats, int((y == value).sum())), replace=True)
                   for value in [True, False]]
    else:
        groups = {c: np.flatnonzero(np.asarray(clusters) == c) for c in sorted(set(clusters))}
        indices = []
        for value in [True, False]:
            members = [v for v in groups.values() if bool(y[v[0]]) == value]
            if any(not (y[v] == value).all() for v in members):
                raise ValueError('Prompt families must have one frozen label')
            choices = rng.integers(0, len(members), size=(repeats, len(members)))
            indices.append(np.stack(members)[choices].reshape(repeats, -1))
    pos, neg = indices
    accuracy = (correct[pos].mean(1) + correct[neg].mean(1)) / 2
    a, b = margin[pos][:, :, None], margin[neg][:, None, :]
    auc = ((a > b).mean((1, 2)) + .5 * (a == b).mean((1, 2)))
    return {'balanced_accuracy_ci95': np.quantile(accuracy, [.025, .975]).tolist(),
            'head_auc_ci95': np.quantile(auc, [.025, .975]).tolist(),
            'unit': 'image' if clusters is None else 'bank and prompt family'}


def summarize():
    done = read_json(ROOT / 'predictions_complete.json')
    verify(done['hashes'])
    verify(done['basis_hashes'])
    plan = read_json(ROOT / 'plan.json')
    verify(plan['hashes'])
    predictions = [read_json(ROOT / 'predictions' / f'{i:03d}.json') for i in range(96)]
    # Labels are joined here only; there are no private victim references.
    bank_rows = [row for bank in BANKS for row in read_json(bank / 'public/bank.json')]
    labels = [r['label'] for r in bank_rows]
    clusters = [f'{i//24}:{(i%24)//4}' for i in range(96)]
    groups = {'generic': list(range(72)), 'medical': list(range(72, 96)),
              **{f'generic_{b+1}': list(range(b*24, (b+1)*24)) for b in range(3)}}
    methods = list(predictions[0]['scores'])
    all_scores = {m: [p['scores'][m] for p in predictions] for m in methods}
    all_scores['other_image'] = [predictions[i]['scores']['neutral_r96_primary']
                                 for i in plan['other_image_indices']]
    all_scores.update({name: [p[name] for p in predictions] for name in ['readout', 'zero_visual_readout']})
    results = {}
    for group, ids in groups.items():
        y = [labels[i] for i in ids]
        results[group] = {m: classification_stats(y, [s[i] for i in ids], ['head', 'non-head'])
                          for m, s in all_scores.items()}
        for m in ['neutral_r96_primary', 'neutral_r96_primary_random', 'other_image', 'readout', 'zero_visual_readout']:
            scores = [all_scores[m][i] for i in ids]
            results[group][m]['image_interval'] = interval(y, scores)
            results[group][m]['family_interval'] = interval(y, scores, [clusters[i] for i in ids])
    gate = all(results[g]['neutral_r96_primary']['balanced_accuracy'] >= .65 and
               results[g]['neutral_r96_primary']['image_interval']['head_auc_ci95'] is not None and
               results[g]['neutral_r96_primary']['image_interval']['head_auc_ci95'][0] > .5
               for g in ['generic', 'medical'])
    gate = gate and all(results[f'generic_{i}']['neutral_r96_primary']['class_auroc']['head'] > .5 for i in [1, 2, 3])
    check = [p['interface_check'] for p in predictions if 'interface_check' in p]
    coverage = {}
    for context in ['raw', 'neutral', 'head_word']:
        rows = [p['diagnostics'][f'{context}_r96_primary'] for p in predictions]
        coverage[context] = {'valid_word_fraction': float(np.mean([r['valid_words'] for r in rows])),
                             'valid_by_word': np.mean([r['valid_words'] for r in rows], axis=0).tolist()}
    wordwise = {}
    for group, ids in groups.items():
        y = np.array([labels[i] == 'head' for i in ids])
        wordwise[group] = {}
        for cluster, words in plan['words'].items():
            rows = [predictions[i]['diagnostics'][f'neutral_r96_{cluster}'] for i in ids]
            values = np.array([row['distances'] for row in rows])
            for j, word in enumerate(sum(words, [])):
                # All columns use head-positive orientation, including negative words.
                wordwise[group][f'{cluster}:{word}'] = binary_auc(y, -values[:, j] if j < 4 else values[:, j])
    numerical = {name: {'rank_min': min(p['diagnostics'][name]['rank'] for p in predictions),
                        'rank_max': max(p['diagnostics'][name]['rank'] for p in predictions),
                        'max_orthogonality_error': max(p['diagnostics'][name]['orthogonality_error'] for p in predictions),
                        'mean_energy_retained': float(np.mean([p['diagnostics'][name]['energy_retained'] for p in predictions]))}
                 for name in predictions[0]['diagnostics'] if name.endswith(('_r32', '_r96', '_r576'))}
    report = {'scope': plan['scope'], 'images': 96, 'results': results,
              'primary': 'neutral_r96_primary', 'd1_gate_passed': gate,
              'e1_interface_passed': len(check) == 4,
              'interface_checks': check, 'word_coverage': coverage,
              'per_word_head_oriented_auroc': wordwise, 'space_diagnostics': numerical,
              'limits': plan['limits'], 'cost': {k: v for k, v in done.items() if k not in ['hashes', 'basis_hashes']},
              'sources': {'plan_sha256': file_hash(ROOT / 'plan.json'),
                          'predictions_complete_sha256': file_hash(ROOT / 'predictions_complete.json')}}
    write_json(ROOT / 'report.json', report)
    public = Path('docs/experiments/data/visual-concept-probe/aggregate.json')
    write_json(public, report)
    write_json(ROOT / 'report_complete.json', {'report_sha256': file_hash(ROOT / 'report.json'),
               'aggregate': str(public), 'aggregate_sha256': file_hash(public)})
    print('D1 gate:', gate)
    for g in ['generic', 'medical']:
        print(g, {m: (results[g][m]['balanced_accuracy'], results[g][m]['class_auroc']['head'])
                  for m in ['neutral_r96_primary', 'neutral_r96_synonyms', 'other_image', 'readout', 'zero_visual_readout']})
    return report


def publish(report):
    """Export this phase as a local draft; keep the unified record untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = 'neutral_r96_primary'
    results = report['results']
    methods = [selected, 'neutral_r96_synonyms', selected + '_random', 'neutral_r96_unrelated',
               'other_image', 'readout', 'zero_visual_readout']
    labels = ['Concept\nprimary', 'Concept\nsynonyms', 'Random\nspace', 'Unrelated\nwords',
              'Other\nimage', 'Model\nreadout', 'Zero\nvisual']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(methods))
    for ax, metric, title in zip(axes, ['balanced_accuracy', 'class_auroc'],
                                 ['Balanced accuracy', 'Head AUROC'], strict=True):
        for offset, group, caption, color in [(-.18, 'generic', 'Generic (72 public images)', '#3a77ad'),
                                                (.18, 'medical', 'Medical (24 public images)', '#d59250')]:
            values = [results[group][m][metric] if metric != 'class_auroc'
                      else results[group][m][metric]['head'] for m in methods]
            ax.bar(x + offset, values, width=.34, label=caption, color=color)
        ax.axhline(.5, color='#777', linestyle='--', linewidth=1)
        ax.set(title=title, ylim=(0, 1.12), xticks=x, xticklabels=labels)
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Public-image geometry screen: no private victim data or gradient recovery')
    fig.tight_layout()
    figure = Path('docs/experiments/figures/visual-concept-probe/geometry-readout.png')
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)

    def fmtci(values):
        return '不可用' if values is None else f'[{values[0]:.3f}, {values[1]:.3f}]'

    generic, medical = [results[g][selected] for g in ['generic', 'medical']]
    verdict = ('预定主条件通过本轮推进门槛，但仍需真实公开图像确认。' if report['d1_gate_passed'] else
               '预定主条件未通过推进门槛，当前词向量构造不足以支持继续进行梯度概念探针。')
    checks = report['interface_checks']
    maxlogit = max(r['max_logit_abs_error'] for r in checks)
    maxgrad = max(r['gradient_relative_l2'] for r in checks)
    text = f'''---
id: visual-concept-probe
title: 视觉子空间概念探针
date: 2026-10-28
track: property
importance: supporting
data: public-synthetic
model: llava-1.5-7b
tuning: F-L
round: 10
knowledge:
- image_known
n: 96
evidence_level: diagnostic
outcome: {'mixed' if not report['d1_gate_passed'] else 'positive'}
uses_private_reference: false
contains_private_tokens: false
uncommitted: true
commits:
- aba0a9f
code:
- evaluation/visual_concept_probe.py
- evaluation/visual_concept_report.py
artifacts:
- path: docs/experiments/data/visual-concept-probe/aggregate.json
  note: 九十六张公开合成候选的概念几何、空间对照、接口读出及区间，不含私有目标
- path: outputs/diagnostics/visual_concept_probe_20261028/plan.json
  note: 执行前冻结的词簇、秩、文本上下文、随机映射与输入哈希
- path: outputs/diagnostics/visual_concept_probe_20261028/runtime_source.json
  note: 冻结源码、环境、WN-A100 GPU UUID 与基础提交
- path: outputs/diagnostics/visual_concept_probe_20261028/predictions_complete.json
  note: 全部九十六份预测和公开子空间保存后才加入公开标签评分
- path: outputs/diagnostics/visual_concept_probe_20261028/report_complete.json
  note: 聚合报告来源哈希
related:
- synthetic-property
---

## 要点

- **概念几何{'通过初步检查' if report['d1_gate_passed'] else '尚未通过'}**：{verdict}
- **视觉 tokens 接口一致**：四个公开样本的 logits 最大差异 {maxlogit:.3g}，LoRA 梯度最大相对误差 {maxgrad:.3g}；视觉 tokens 的反向梯度有效。
- **读出与恢复分开评价**：本页直接提供公开真图的视觉 tokens，不是从梯度恢复图像或属性。
- **证据限于合成候选**：重用了此前公开生成库，没有读取私有目标或训练新对齐层，不能替代真实图像验证。

## 关键数字

| Public group | Images | Coverage | Primary balanced accuracy | Head AUROC | AUROC image CI95 | Model readout accuracy |
|---|---:|---:|---:|---:|---|---:|
'''
    for g, caption in [('generic', 'Generic'), ('medical', 'Medical')]:
        r = results[g][selected]
        text += f"| {caption} | {r['n']} | {1-r['invalid']/r['n']:.3f} | {r['balanced_accuracy']:.3f} | {r['class_auroc']['head']:.3f} | {fmtci(r['image_interval']['head_auc_ci95'])} | {results[g]['readout']['accuracy']:.3f} |\n"
    text += '''
## 研究问题

- **概念几何**：直接视觉子空间能否区分预先固定的头部与非头部概念？
- **空间对照**：结果能否排除随机高秩、无关概念和文本重叠的影响？
- **接口读出**：真实视觉 tokens 注入后能否复现原前向与梯度，原模型能否读出属性？

## 实验设置

执行前的完整方案见[视觉概念检查](../plans/2026-10-28-image-property-ideas.md#concept-design)。

| Item | Setting |
|---|---|
| Public images | 三套通用摄影候选及一套医学候选，各二十四张，类别各半；复用已审核生成图 |
| Private inputs | 不读私有目标图像、问答、Observation 或重构结果 |
| Model | 原 LLaVA-1.5-7B，F-L，Rank 32，Round 10；不更新参数 |
| Primary probe | 首层 input RMSNorm 后的视觉特征；减去固定中性文本方向，前九十六个视觉方向 |
| Primary concepts | head、brain、skull、face 对 chest、abdomen、arm、leg |
| Synonyms | head、cranium、forehead、scalp 对 torso、belly、hand、foot |
| Unrelated concepts | table、chair、door、window 对 river、mountain、tree、cloud |
| Multi-token words | 各 token 经同一 RMSNorm 后取均值；词簇冻结，不按结果挑词 |
| Score | 概念在视觉空间外的相对残差；每类至少两个有效词，类别内均值较低者胜出 |
| Other conditions | 秩三十二、完整视觉秩；原始空间；固定 head 词面重叠上下文 |
| Controls | 等秩随机商空间、预定同库其他图像映射、无关词簇、概念分组置换 |
| Readout | 固定头部 yes/no 问句，比较原生答案 NLL，含 EOS；另输入全零视觉槽位 |
| Interface | 四个预定样本比较 logits、loss 和全部 LoRA 梯度，容差 1e-6；检查视觉变量梯度 |

所有概念、秩和映射在预测前冻结，不拟合阈值。进入 D2 的门槛是：摄影组与医学组的主探针 Balanced accuracy 均至少 0.65，图像级 Head AUROC 区间下限均高于 0.5，且三套摄影库分数方向一致。D1 不是梯度攻击，也不是其严格性能上界。

## 结果

### 概念几何

'''
    text += f'**回答**：{verdict}以下保留全部预定秩和词簇，不用其中最好的条件替换主结果。\n\n'
    text += '| Context | Rank | Concepts | Generic balanced accuracy | Generic Head AUROC | Medical balanced accuracy | Medical Head AUROC |\n|---|---:|---|---:|---:|---:|---:|\n'
    for context in ['neutral', 'raw', 'head_word']:
        for rank in [32, 96, 576]:
            for cluster in ['primary', 'synonyms']:
                a, b = [results[g][f'{context}_r{rank}_{cluster}'] for g in ['generic', 'medical']]
                text += f"| {context} | {rank} | {cluster} | {a['balanced_accuracy']:.3f} | {a['class_auroc']['head']:.3f} | {b['balanced_accuracy']:.3f} | {b['class_auroc']['head']:.3f} |\n"
    text += '\n图像级区间将每张生成图视为采样单位；下面再按库内每个部位提示家族成组重采样，反映相似提示种子的依赖。医学组只有六个提示家族，区间仍不覆盖任意新领域。\n\n'
    text += '| Public group | BA image CI95 | BA family CI95 | Head AUROC family CI95 |\n|---|---|---|---|\n'
    for g in ['generic', 'medical']:
        r = results[g][selected]
        text += f"| {g} | {fmtci(r['image_interval']['balanced_accuracy_ci95'])} | {fmtci(r['family_interval']['balanced_accuracy_ci95'])} | {fmtci(r['family_interval']['head_auc_ci95'])} |\n"
    text += '''
### 空间对照

'''
    random_auc = [results[g][selected + '_random']['class_auroc']['head'] for g in ['generic', 'medical']]
    other_auc = [results[g]['other_image']['class_auroc']['head'] for g in ['generic', 'medical']]
    text += (f'**回答**：等秩随机空间在摄影与医学组的 Head AUROC 分别为 {random_auc[0]:.3f}、{random_auc[1]:.3f}；'
             f'改用其他图像空间后分别为 {other_auc[0]:.3f}、{other_auc[1]:.3f}。无关词和分组置换另列，不能只凭距离低就解释为正确语义。\n\n')
    text += '| Condition | Generic balanced accuracy | Generic Head AUROC | Medical balanced accuracy | Medical Head AUROC |\n|---|---:|---:|---:|---:|\n'
    for name, m in [('Primary', selected), ('Random space', selected + '_random'),
                    ('Other image', 'other_image'), ('Unrelated concepts', 'neutral_r96_unrelated'),
                    ('Permuted concepts', selected + '_permuted'), ('Synonyms', 'neutral_r96_synonyms')]:
        a, b = [results[g][m] for g in ['generic', 'medical']]
        text += f"| {name} | {a['balanced_accuracy']:.3f} | {a['class_auroc']['head']:.3f} | {b['balanced_accuracy']:.3f} | {b['class_auroc']['head']:.3f} |\n"
    valid = report['word_coverage']
    text += f"\n- 中性文本条件的有效词比例为 {valid['neutral']['valid_word_fraction']:.3f}；包含 head 的上下文中为 {valid['head_word']['valid_word_fraction']:.3f}。近零残差不会作为正证据，单类少于两个有效概念时拒答。\n"
    text += '- 随机空间对相同秩和上下文固定，不携带图像信息；Other image 使用预测前固定的同库随机循环映射，不按类别选图。\n- 完整逐词结果和数值秩、正交误差在聚合产物中。逐词结果只作敏感性诊断，不据此重新选择词簇。\n'
    text += '\n### 接口读出\n\n'
    text += (f'**回答**：四项接口一致性检查通过，logits 最大误差 {maxlogit:.3g}，LoRA 梯度最大相对误差 {maxgrad:.3g}；'
             f"真实视觉 tokens 的属性读出准确率在摄影与医学组分别为 {results['generic']['readout']['accuracy']:.3f}、{results['medical']['readout']['accuracy']:.3f}。"
             '这只检查接口与识别能力，没有验证从上传梯度恢复它们。\n\n')
    text += '| Public group | Real visual tokens accuracy | Real visual tokens Head AUROC | Zero visual accuracy | Zero visual Head AUROC |\n|---|---:|---:|---:|---:|\n'
    for g in ['generic', 'medical']:
        a, b = results[g]['readout'], results[g]['zero_visual_readout']
        text += f"| {g} | {a['accuracy']:.3f} | {a['class_auroc']['head']:.3f} | {b['accuracy']:.3f} | {b['class_auroc']['head']:.3f} |\n"
    text += '''
读出使用真实公开图像的视觉 tokens，是模型识别能力检查，完全不使用目标梯度。Zero visual 保留原视觉槽位和位置但填零，属于消融，不等于标准 Text-only 或合法图像输入。未进行自由 tokens 优化，也没有生成或恢复私有图像。

[![公开图像概念探针与读出比较](figures/visual-concept-probe/geometry-readout.png)](figures/visual-concept-probe/geometry-readout.png)

## 结论

'''
    text += f'- {verdict}\n'
    text += '''- 接口一致性与属性恢复是两个问题：原生前向可复现，只建立了后续表征优化的工程入口。
- 不能将当前词向量探针的失败扩大解释为视觉表征或梯度不含属性信息；跨模态几何、词片段聚合与子空间构造均可能影响结果。

## 局限

- 全部图像来自已有合成库，标签为大类外观核验；没有新真实公开图像，更不是私有目标确认集。
- 医学候选的模态按提示记录，部分 MRI 外观偏 CT；医学真实性没有专家复核。
- 非头部包含不同部位，词簇覆盖不完整；多 token 词片段均值只是启发式，同维度不意味着语义对齐。
- 高秩空间可以让无关概念也接近，不能只看距离低；主条件、随机空间与无关词必须一起解释。
- 秩九十六是预定筛查条件；真实 LoRA 梯度行空间不必等于视觉特征的前九十六主方向，不能把此截断空间当作严格上界。
- 接口中的非零视觉梯度是语言损失对 tokens 的一阶梯度；尚未验证梯度匹配优化所需的二阶链和收敛性。
- 图像级和提示家族区间都不能覆盖生成器、概念选择及真正未知领域的不确定性。

## 下一步

'''
    text += ('- 在真实公开图像上冻结同一探针确认，再进行 D2 的直接视觉空间与梯度残差空间配对比较。\n' if report['d1_gate_passed'] else
             '- 暂停按当前原始词向量直接推进 D2；如研究对齐层，单独声明公开图文训练数据与开发集，重新独立验证。\n')
    launch = read_json(ROOT / 'launch.json')
    text += '''- 视觉 tokens 路线先依据本页读出检查确认属性问题是否可用，再设计少量公开样本的 E2 优化；仅按可观察梯度误差选择，不按真实属性挑初始化。
- 本轮未运行文本引导生成、D2 梯度探针或 E2 表征恢复，不把前提检查标作整条方案完成。

??? note "运行记录"

    - 服务器 WN-A100，GPU 1，UUID `GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09`；启动前显存近零、利用率零，是当时最空闲的卡。
'''
    text += f"    - 使用共享 gia；controller {launch['controller_pid']}、worker {launch['worker_pid']}、watcher {launch['watcher_pid']}，源码 source-v2、输入与计划全部冻结，基础 commit aba0a9f，新增代码未提交。\n"
    text += '    - 首次接口检查的 logits 和梯度均完全一致，但融合交叉熵与原实现的词表维度归约相差 1.43e-6，触发既定容差并停止，未提交逐图预测。保留 attempt-1 与源码；对齐评估 loss 的原生归约后执行第二次检查，不放宽容差、不改模型或科学设置，无 OOM。\n'
    text += f"    - 全部九十六份预测保存后才加入公开标签；无私有参考，模型参数未改变。运行墙钟 {report['cost']['seconds']:.2f} 秒，峰值 allocated {report['cost']['peak_mib']:.2f} MiB。\n"
    text += '    - 四项数学及损失测试、Ruff 通过；原始公开图像和逐图基只保留在 outputs，文档只纳入聚合表图。\n'
    path = ROOT / 'concept-report-draft.md'
    path.write_text(text)
    write_json(ROOT / 'draft_export.json', {'draft': str(path), 'sha256': file_hash(path),
               'canonical_record': 'docs/experiments/2026-10-28-image-property-ideas.md',
               'documentation_written': False,
               'figure': str(figure), 'figure_sha256': file_hash(figure)})


if __name__ == '__main__':
    publish(summarize())
