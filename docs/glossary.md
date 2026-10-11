# 术语表

页面中带虚线下划线的术语，鼠标悬停可以看到简短解释；这里给出完整说明。

实验条件的命名：`Method` 表示方法变体（如 Residual、Raw），`Image` 表示图像条件（GT 为真实图像，Random 为 random surrogate）；正文中写作 Residual (GT)、Raw (Random) 等。

## 联邦设置与模型

| Term | Description |
|---|---|
| F-C / F-L / F-CL | FedVLMBench 的三种微调策略：只训练连接器（connector）/ 只训练 LLM 的 LoRA / 两者都训练。决定客户端上传哪些参数的梯度。 |
| round N | 联邦训练进行到第 N 轮时的全局模型快照。第 0 轮是未训练的初始模型，LoRA-B 为零。 |
| FedSGD / FedAvg | 客户端上传一步梯度 / 上传多步本地训练后的参数更新。本项目的攻击捕获默认用 FedSGD、batch 1。 |
| LoRA rank | LoRA 低秩矩阵的秩 r（本项目为 8）。 |
| bf16 / fp32 | 受害模型或候选图像使用的数值精度。bf16 会把很小的变化舍入掉。 |

## 威胁模型

| Term | Description |
|---|---|
| 攻击者知识（knowledge） | 攻击者事先知道哪些字段。`private`：都不知道；`image_known`：知道图像；`text_known`：知道问题和答案；`question_known` / `target_known`：只知道问题 / 只知道答案；`image_question_known`：知道图像和问题。 |
| lengths known | 攻击者知道每个样本问题和答案的 token 数（`token_lengths_known`）。 |
| 使用私有数据 | 实验用到了攻击者在威胁模型下拿不到的数据，例如真实图像或真实文本（比如从真实图像附近出发）。这类实验用来诊断攻击为什么失败，结论**不能**当作攻击结果。论文里常称为 oracle 实验。 |
| random surrogate | 图像私有时，攻击者自己生成的随机像素图，用来近似"视觉信息张成的方向"；不针对私有图像优化或挑选。 |

## 攻击方法

| Term | Description |
|---|---|
| IG | Inverting Gradients：通过优化候选图像，让它产生的梯度与观测梯度尽量接近。 |
| TAG | 文本侧的梯度匹配攻击，优化连续的文本表示再离散化。 |
| DAGER / `dager_adapted` | DAGER 利用梯度的低秩结构，先判断哪些 token 出现过（第一阶段），再搜索排列成句子（第二阶段）。`dager_adapted` 是本项目针对 LoRA 的改编版，解码器不是论文原版。 |
| public residual / raw | DAGER 第一阶段的两种打分：先从梯度中减去公开信息（模板文本、图像）张成的子空间再打分 / 直接打分。 |
| informative / ambiguous 候选 | 第一阶段筛出的 token：与公开文本不重叠的 / 与公开文本重叠、无法判断是否来自私有文本的。 |
| matching loss | 候选输入产生的梯度与观测梯度之间的距离（余弦或相对 L2）。 |
| α（近真值起点） | 使用私有数据的诊断：起点 = (1−α)·真实图像 + α·均匀噪声。α 越小越接近真实图像。 |

## 评估指标

| Term | Description |
|---|---|
| Top-K / K | 按第一阶段打分取排名前 K 的 token 作为候选。 |
| Recall@K | 私有文本的去重 token 中，有多少比例落在前 K 个候选里。 |
| full coverage | 候选集包含了某个样本的**全部**私有 token。这是第二阶段有可能完全恢复句子的前提。 |
| full coverage (actual / K=100) | 分别按实际搜索用的候选集、按排名前 100 的候选集计算 full coverage。两者在 residual 条件下不同，因为实际候选集额外包含 9 个 ambiguous ID。 |
| AP | Average Precision，衡量真实 token 是否排得靠前；越高越好。 |
| EM | Exact Match，重建文本与原文完全一致的比例。 |
| Joint EM | 问题和答案**都**与原文完全一致的样本比例，即整条样本被完整恢复。 |
| ROUGE-1 / 2 / L | 重建文本与原文在单词、二元词组、最长公共子序列上的重叠程度。 |
| WER | 词错误率，越低越好；可以超过 1。 |
| PSNR / SSIM | 图像重建质量：峰值信噪比（dB，越高越好）/ 结构相似度（0–1，越高越好）。SSIM ≈ 0.007 就是噪声水平。 |
| near-win | 两个候选中，离真实图像更近（图像误差更小）的那个匹配损失也更低的概率；0.5 等于随机。 |
| Spearman rho | 匹配损失排名与图像误差排名之间的秩相关系数；1 为完全一致，0 为无关，负数为相反。 |
| Token Top-1 credit | 正确 token 的图像排序分数是否在候选中最高；并列第一时按并列数分摊分数。一个正确候选对三个错误候选时，随机基线为 0.25。 |
| Token pairwise win | 正确 token 的图像排序分数高于错误 token 的比例；并列计 0.5，随机基线为 0.5。 |

## 证据等级

| Level | Description |
|---|---|
| smoke | 只验证流程能跑通。 |
| diagnostic | 通常是单个样本，或使用了私有数据，用来找原因，不能推广。 |
| development | 多个样本，但分析方法可能还在调整。 |
| validation | 分析方法先在开发集上冻结，再在新的独立样本上检验。 |
| benchmark | 可用于论文的正式结果（目前还没有）。 |
