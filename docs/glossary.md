# 术语表

页面中带虚线下划线的术语，鼠标悬停可以看到简短解释；这里给出完整说明。

实验条件的命名：`Method` 表示方法变体（如 Residual、Raw），`Image` 表示图像条件（GT 为真实图像，Random 为 random surrogate）；正文中写作 Residual (GT)、Raw (Random) 等。

## 联邦设置与模型

| Term | Description |
|---|---|
| F-C / F-L / F-CL | FedVLMBench 的三种微调策略：只训练连接器（connector）/ 只训练 LLM 的 LoRA / 两者都训练。决定客户端上传哪些参数的梯度。 |
| round N | 联邦训练进行到第 N 轮时的全局模型快照。第 0 轮是未训练的初始模型，LoRA-B 为零。 |
| FedSGD / FedAvg | 客户端上传一步梯度 / 上传多步本地训练后的参数更新。本项目的攻击捕获默认用 FedSGD、batch 1。 |
| LoRA rank | LoRA 低秩矩阵的秩 r；当前配对文本实验比较了 8、16、32。 |
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
| public residual / raw | `dager_adapted` 的两种空间打分：Residual 从梯度和候选特征中同时去掉已知输入方向；Raw 直接使用梯度行空间。两个阶段分别使用对应层的特征和梯度。见 [Residual v1](methodology/dager/residual-v1.md)。 |
| informative / ambiguous 候选 | 去掉已知输入方向后仍有可评分分量的候选 / 投影后几乎为零、成员关系无法判断的候选。已知方向可包含图像与模板，不只是公开文本；ambiguous 另行保留，不算检测阳性。 |
| 梯度行空间 | 梯度矩阵各行的所有线性组合组成的空间；它可能只覆盖实际输入空间的一部分。 |
| 正交行基 | 用相互正交的单位行向量表示一个空间；Residual 分别用 U 表示已知方向、S 表示残余梯度方向。 |
| SVD | 奇异值分解；这里用于提取数值上独立的方向及其正交行基。 |
| RMSNorm | 模型的输入归一化操作；第一阶段对词表 embedding 应用对应层的 RMSNorm 后才进行空间打分。 |
| matching loss | 候选输入产生的梯度与观测梯度之间的距离（余弦或相对 L2）。 |
| Public prototype bank | 攻击者明确持有的带属性标签的辅助图像库；必须与目标图像隔离，不能默认将整个 tune 划分视为公开。 |
| Centered cosine | 从候选和观测中减去同一个基准梯度后计算余弦距离；模态 pilot 的基准仅由当前公开候选库计算，不能解释为精确分离了文本贡献。 |
| Equal-tensor cosine | 每个有效上传参数张量先计算余弦距离，再等权平均；避免高范数张量支配，但可能放大小梯度噪声。 |
| Centered equal-tensor cosine | 先应用公开库中心化，再对有效张量的余弦距离等权平均；对同一目标的全部候选使用相同有效张量集合。 |
| α（近真值起点） | 使用私有数据的诊断：起点 = (1−α)·真实图像 + α·均匀噪声。α 越小越接近真实图像。 |

## 评估指标

| Term | Description |
|---|---|
| Top-K / K | 按第一阶段打分取排名前 K 的 token 作为候选。 |
| Recall@K | 私有文本的去重 token 中，有多少比例落在前 K 个候选里。 |
| full coverage | 候选集包含了某个样本的**全部**私有 token。这是第二阶段有可能完全恢复句子的前提。 |
| full coverage (actual / K=100) | 分别按实际搜索用的候选集、按排名前 100 的候选集计算 full coverage。两者在 residual 条件下不同，因为实际候选集额外包含 9 个 ambiguous ID。 |
| AP | Average Precision，衡量真实 token 是否排得靠前；越高越好。 |
| Balanced accuracy | 每个目标类别的 Recall 的等权平均；模态三分类中，缺少任一目标类别时不报告完整三分类值。 |
| Supported-class BA | 仅对当前队列有目标样本的类别取 Recall 等权平均；缺 MRI 的验证队列只包含 CT 与 X-Ray，不能当作三分类 Balanced accuracy。 |
| Macro-F1 | 对各类别的 F1 等权平均；完整类别指标要求每类均有目标样本。 |
| AUROC | ROC 曲线下面积；模态 pilot 使用该类相对最强竞争类的得分优势，按 one-versus-rest 计算；没有正例或负例时不可评估。 |
| Calibrated Text-only | 无图语言模型对类别名称的续写 NLL 加上在旧开发集冻结的类别偏置；新评估标签不参与校准。 |
| McNemar | 比较同一批样本上两个分类器的配对检验；精确版本仅使用两者预测正确性不一致的样本。 |
| EM | Exact Match，重建文本与原文完全一致的比例。 |
| Joint EM | 问题和答案**都**与原文完全一致的样本比例，即整条样本被完整恢复。 |
| Beam hit rate | 正确 token 序列是否在指定深度保留的 beam 中；最终 beam 命中不等于最后选出的序列正确。 |
| Joint beam hit | 最终 beam 中包含正确完整问答 token 序列的样本比例；只表示候选被保留，不表示最终被选中。 |
| Prefix survival fraction | 一次攻击中真实完整前缀仍在 beam 的私有深度数，除以全部私有 token 数；仅在重构保存后用真值计算。 |
| NLL | Negative log-likelihood，负对数似然。纯文本先验使用下一个内容 token 在完整词表上的条件概率计算 `−log p`；越小表示语言模型越偏好。 |
| Bag recall | 最终重构序列保留的真实内容 token 比例：逐 token ID 取重构与参考出现次数的较小值，求和后除以参考长度；不计顺序，排除 EOS 与 padding。不同于第一阶段候选集的 Recall@K。 |
| Position accuracy | 最终重构序列中，内容 token ID 与参考在同一位置相等的比例；排除 EOS 与 padding。 |
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
