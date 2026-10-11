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
*[residual]: 先减去公开信息张成的子空间再打分
*[Recall@100]: 私有 token 中落在前 100 个候选里的比例
*[near-win]: 误差更小的候选，匹配损失也更低的概率；0.5 为随机
*[surrogate]: 图像私有时，攻击者自己生成的随机像素图，用来代替真实图像
*[full coverage]: 候选集包含了样本的全部私有 token
*[GT]: Ground truth：攻击者知道真实图像
*[Joint EM]: 问题和答案都完全正确的样本比例
*[Token Top-1 credit]: 正确 token 的图像排序分数在候选中最高时得分；并列按并列数分摊
*[Token pairwise win]: 正确 token 的图像排序分数超过错误 token 的比例，并列计 0.5
