*[F-C]: 微调策略：只训练连接器
*[F-L]: 微调策略：只训练 LLM 的 LoRA
*[F-CL]: 微调策略：连接器和 LoRA 都训练
*[FedSGD]: 客户端上传一步梯度
*[FedAvg]: 客户端上传多步本地训练后的参数更新
*[IG]: Inverting Gradients：优化候选图像使其梯度匹配观测梯度
*[TAG]: 文本侧梯度匹配攻击
*[DAGER]: 利用梯度低秩结构：先找出现过的 token，再搜索成句
*[AP]: Average Precision：真实 token 是否排得靠前，越高越好
*[Informative AP]: 在排名与参考 token 集合中共同排除 ambiguous ID 后计算的 Average Precision
*[EM]: Exact Match：与原文完全一致的比例
*[WER]: 词错误率，越低越好，可超过 1
*[PSNR]: 峰值信噪比（dB），越高越接近原图
*[SSIM]: 结构相似度（0–1），≈0.007 即噪声水平
*[residual]: 从梯度和候选特征中同时去掉已知输入方向，再计算相对距离
*[Residual]: 从梯度和候选特征中同时去掉已知输入方向，再计算相对距离
*[ambiguous]: 候选投影后几乎为零，成员关系无法判断；另行保留，不算检测阳性
*[SVD]: 奇异值分解；这里用来提取空间的正交行基
*[Rowspace distance]: 两行空间投影矩阵差的归一化 Frobenius 范数；0 为相同，1 为正交，不是错误概率
*[Gradient relative L2]: 梯度差范数除以参考梯度范数，同时反映方向和幅度变化
*[Single-QA ridge]: 只用一种公开问答，闭式校准多层属性分数；不更新模型权重
*[Multi-QA ridge]: 同图多种公开问答共享样本权重，闭式校准多层属性分数；不使用目标问答
*[Single-QA threshold]: 用一种公开问答确定单层判断阈值，保留原评分方向
*[Multi-QA threshold]: 用多种公开问答确定单层判断阈值，保留原评分方向
*[QA flip images]: 同一图像更换问答后至少发生一次预测类别翻转的图像数
*[Held-out QA balanced accuracy]: 在未参与拟合的问答模板上，逐模板计算 Balanced accuracy 后取平均
*[Worst QA balanced accuracy]: 指定问答集合中最低的 Balanced accuracy，不是所有问答的保证
*[Rank at 99.9% energy]: 累计保留 99.9% 梯度奇异值平方所需的方向数，不等于属性信息量
*[Mean top-1 energy]: 每图最大奇异值平方占总量的比例，再跨图平均
*[Class mean gap]: 正类平均分差减负类平均分差，只描述当前评分的类间间隔
*[q_proj]: 注意力查询投影；当前位置如何查询上下文
*[k_proj]: 注意力键投影；更早位置可被后续答案查询
*[v_proj]: 注意力值投影；更早位置的信息可汇入后续答案
*[RMSNorm]: 模型的输入归一化操作
*[RMS]: 均方根；多层属性评分使用公开开发分数的均方根统一各层尺度
*[Recall@100]: 私有 token 中落在前 100 个候选里的比例
*[near-win]: 误差更小的候选，匹配损失也更低的概率；0.5 为随机
*[surrogate]: 图像私有时，攻击者自己生成的随机像素图，用来代替真实图像
*[full coverage]: 候选集包含了样本的全部私有 token
*[FullCoverage]: 候选集包含了样本的全部私有 token
*[Candidate recall]: 逐样本真实 token 在实际候选集中的召回再取平均，不是严格的 Recall@K
*[Question-exclusive recall]: 候选集对问题独有 token 集合 Q∖A 的召回；空集合不参与宏平均并报告有效样本数
*[DAGER-Erasure-LM]: public_residual 筛选加纯文本 LM prior 的展示名，不是原版 DAGER
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
*[Q32]: 第 32 层 q_proj 上的梯度原型探针
*[Relative energy]: 两类原型投影能量差除以其能量和；公开校准阈值的属性分数，保留原分数方向
*[V24]: 第 24 层 v_proj 上的梯度原型探针
*[Dynamic-LM-16]: 静态候选并入 text-only LM 的 Top-16，按梯度排名补齐到 125 个
*[LM-only]: 只用声明长度与公开语言模型解码，不用图像或梯度
*[B1S1]: 单样本、单步本地更新
*[B2S1]: 每步两条问答、本地一步
*[B1S2]: 每步一条问答、本地两步
*[B4S1]: 每步四条问答、本地一步
*[B2S2]: 每步两条问答、本地两步
*[B1S4]: 每步一条问答、本地四步
