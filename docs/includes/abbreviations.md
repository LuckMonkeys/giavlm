*[F-C]: 微调策略：只训练连接器
*[F-L]: 微调策略：只训练 LLM 的 LoRA
*[F-CL]: 微调策略：连接器和 LoRA 都训练
*[FedSGD]: 客户端上传一步梯度
*[FedAvg]: 客户端上传多步本地训练后的参数更新
*[IG]: Inverting Gradients：优化候选图像使其梯度匹配观测梯度
*[TAG]: 文本侧梯度匹配攻击
*[DAGER]: 利用梯度低秩结构：先找出现过的 token，再搜索成句
*[AP]: Average Precision：真实 token 是否排得靠前，越高越好
*[EM]: Exact Match：与原文完全一致的比例
*[WER]: 词错误率，越低越好，可超过 1
*[PSNR]: 峰值信噪比（dB），越高越接近原图
*[SSIM]: 结构相似度（0–1），≈0.007 即噪声水平
*[residual]: 从梯度和候选特征中同时去掉已知输入方向，再计算相对距离
*[Residual]: 从梯度和候选特征中同时去掉已知输入方向，再计算相对距离
*[ambiguous]: 候选投影后几乎为零，成员关系无法判断；另行保留，不算检测阳性
*[SVD]: 奇异值分解；这里用来提取空间的正交行基
*[RMSNorm]: 模型的输入归一化操作
*[Recall@100]: 私有 token 中落在前 100 个候选里的比例
*[near-win]: 误差更小的候选，匹配损失也更低的概率；0.5 为随机
*[surrogate]: 图像私有时，攻击者自己生成的随机像素图，用来代替真实图像
*[full coverage]: 候选集包含了样本的全部私有 token
*[GT]: Ground truth：攻击者知道真实图像
*[Joint EM]: 问题和答案都完全正确的样本比例
*[NLL]: 负对数似然；纯文本先验按完整词表条件概率计算内容 token 的 −log p
*[Beam hit rate]: 正确 token 序列仍在指定深度的 beam 中的比例，不等于最终输出正确
*[Joint beam hit]: 最终 beam 中包含正确完整问答 token 序列的样本比例
*[Prefix survival fraction]: 真实前缀仍在 beam 的私有深度数除以全部私有 token 数
*[Bag recall]: 最终序列保留真实 token 的比例，按重复次数计数，不计顺序，排除 EOS 与 padding
*[Position accuracy]: 最终序列中内容 token 在正确位置的比例，排除 EOS 与 padding
*[Token Top-1 credit]: 正确 token 的图像排序分数在候选中最高时得分；并列按并列数分摊
*[Token pairwise win]: 正确 token 的图像排序分数超过错误 token 的比例，并列计 0.5
*[Public prototype bank]: 攻击者持有的带属性标签辅助图像库，与全部目标图像隔离
*[Centered cosine]: 从候选和观测减去同一个基准梯度后计算余弦距离
*[Equal-tensor cosine]: 每个有效上传张量的余弦距离等权平均
*[Centered equal-tensor cosine]: 公开库中心化后，对有效张量的余弦距离等权平均
*[Balanced accuracy]: 各目标类别 Recall 的等权平均，完整指标要求各类均有目标样本
*[Supported-class BA]: 仅对当前队列有目标样本的类别计算平均 Recall，不能替代缺失类别的完整指标
*[Macro-F1]: 各类别 F1 的等权平均
*[AUROC]: ROC 曲线下面积；某类别没有正例或负例时不可评估
*[Calibrated Text-only]: 无图语言模型类别分数加上旧开发集冻结的类别偏置
*[McNemar]: 比较两个分类器在同一批样本上预测正确性差异的配对检验
