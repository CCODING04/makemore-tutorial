# 测验 · Part 11（对齐实战：verl 工业级 RL 后训练）

> 依据：本 Part 学习目标与教程正文。建议先完成作业再自测；每题附答案与教程锚点。

## 测验（5 题）

**Q1（概念）** SFT 之后为什么还需要 RL 后训练（RLHF/GRPO）？RLVR 的"可验证奖励"又是什么？

- **答案**：SFT 有两个根本限制——人类示范只覆盖一小部分输入空间（覆盖率问题），且优化目标是"模仿人类"而非"解决问题"（优化目标问题）。RL 通过试错学习弥补：SFT 学格式和风格，RL 学推理和决策。RLVR（可验证奖励）指用规则评估器代替奖励模型——如 GSM8K 从回答中抽取数字、对则 1 分否则 0 分，模型从头到尾不见标准答案标签，"以验证代替标注"；局限是只适用于答案可形式化验证的任务，创意写作、对话质量等主观任务仍需 RM/LLM-as-judge。
- **锚点**：教程 01_handwritten_to_verl.md §理论背景（问题引入 + 概念检验 Q1）

**Q2（对比）** GRPO 与 PPO 的基线来源有何本质区别？"全对组优势全零"为什么是 feature 而不是 bug？全错组的"跳过"又为什么是盲区？

- **答案**：PPO 的基线来自需要额外训练的 Value 网络（显存开销大且可能不稳定）；GRPO 用同一 prompt 的 G 个回答的组内均值做基线，$A_i = (r_i - \text{mean}) / \text{std}$，不需要训练任何额外网络。全对组中所有 $r_i$ 相同导致 std=0、优势全 0，该样本已掌握、无需再学，GRPO 天然把算力集中到有区分度的题上（课程学习的一种形式）——所以是 feature。但全错组同样优势全零且永远学不会：玩具实验中 6 道题里 1 道全错卡死，平均奖励停在 0.83 而非 1.00；DAPO 的 dynamic sampling（过滤全对/全错组后重新采样）正是治这个病。
- **锚点**：教程 01_handwritten_to_verl.md §数学推导（GRPO 的组内优势）+ §代码实现（三个关键观察）

**Q3（数字）** k3 KL 估计器的公式是什么？课程实测中它算出的 KL 值是多少？为什么它恒非负？

- **答案**：k3 估计器为 $\text{KL} = \mathbb{E}[\exp(d) - d - 1]$，其中 $d = \log p_{\text{ref}} - \log p_{\text{new}}$。课程脚本 01 实测 $\text{KL}(\pi_{\text{new}} \| \pi_{\text{ref}}) = 0.0204$（手算验证：两组 token 的贡献分别为 0.0231 与 0.0177，平均得 0.0204）。恒非负的原因是 $\exp(d) - d - 1 \ge 0$ 对所有 d 成立（由 $e^x \ge x+1$），d=0 时取等号。它是低方差估计器，verl 的 KL 惩罚即此形态。
- **锚点**：教程 01_handwritten_to_verl.md §代码实现（③ k3 KL 估计器 + 手算验证）

**Q4（概念）** RL 训练为什么必须分离 rollout 和 training 两个引擎？verl 三角色架构是什么？权重同步由什么组件优化？

- **答案**：rollout（生成）占 RL 训练时间的 60-80%，是 memory-bound 的生成瓶颈；真实 7B 模型生成一个回答要几百 ms，而训练是 compute-bound。两类负载特性不同，故 rollout 用 vLLM/SGLang 高吞吐生成、training 用 FSDP2/Megatron 高效训练。代价是每次更新后要把新权重搬进推理引擎（GB 级拷贝）。verl 三角色为 Actor（训练，FSDP2）、Rollout（生成，vLLM）、Ref（冻结的 SFT 副本，算 KL 惩罚）；权重回同步由 3D-HybridEngine 用重分片+原地转换压到最低。这解释了 GRPO 显存比 PPO 低（无 critic）以及"双卡未必快"（rollout 是瓶颈）。
- **锚点**：教程 02_verl_quickstart.md §理论背景（verl 的三角色架构）+ 01_handwritten_to_verl.md §工程实践（为什么工业版必须"两个引擎"）

**Q5（诊断）** 你自定义的奖励函数"只看最后一个数字对不对"，训练后模型奖励分数很高但题目并没有真正会做——这是什么病？另举两种防范策略。若训练中 loss 卡住不降且发现所有组的奖励全同，又该如何处置？

- **答案**：这是 reward hacking：奖励函数有洞（只看最后数字），模型学会钻洞（如不管问题是什么总输出某个高频答案）而没有真正解决问题。防范策略：奖励函数尽可能严格（检查格式与推理过程）、添加惩罚项（回答过长/格式错误）、多奖励函数组合。若所有组全同（全对或全错）导致优势全零、loss 不降：全对组可跳过（已掌握），全错组需用课程学习（从简单到难）、增大组大小 n 或用 DAPO 的 dynamic sampling 重新采样以制造组内区分度。
- **锚点**：教程 01_handwritten_to_verl.md §工程实践（陷阱 1: Reward Hacking / 陷阱 2: 全对/全错组无梯度）

## 覆盖映射（学习目标 → 题号）

| 学习目标 | 题号 | 说明 |
|---|---|---|
| 理解 RL 后训练在 LLM 链路中的位置和价值 | Q1 | SFT 限制与 RLVR 定位 |
| 手写 RLVR 奖励函数并解释"全对组优势全零" | Q2、Q5 | 组内优势数学性质与两种命运 |
| 画出手写 GRPO 到 verl 三角色架构的映射图 | Q4 | 三角色 + 权重同步 + 双引擎 |
| 配置 verl 的 PPO/GRPO 训练并读懂关键日志 | Q4 | 覆盖配置要点（adv_estimator/n/micro_batch_size）；日志细节（generate_sequences/sync_rollout_weights/val/test_score）未单独出题，正文覆盖充分但篇幅所限并入 Q4 |
| 识别 reward hacking 的风险并设计防范策略 | Q5 | 症状—原因—解法 |

## 闪卡（正/背）

| 正面 | 背面 |
|---|---|
| GRPO 的优势公式？ | $A_i = (r_i - \text{mean}) / \text{std}$，组内标准化；$\sum A_i = 0$ |
| GRPO vs PPO 的基线来源？ | GRPO 用组内均值（免训 Value 网络，Critic-Free）；PPO 用学习出的 Value 网络 |
| RLVR 奖励抽取链的顺序？ | \boxed{} 优先 → `#### 42` 标记 → 最后一个数字；比较用 $\lvert \text{pred} - \text{gt} \rvert < 10^{-4}$ |
| k3 KL 估计器公式与性质？ | $\text{KL} = \mathbb{E}[\exp(d) - d - 1]$，$d = \log p_{\text{ref}} - \log p_{\text{new}}$；恒非负、低方差 |
| "全对组优势全零"为何是 feature？ | 已掌握的样本无梯度，算力自动集中到有区分度的题上（天然课程学习） |
| 玩具 GRPO 实验：奖励从多少涨到多少？BC 上限是多少？ | 0.38 → 0.83（全错组卡死）；BC 有标签直接到 1.00 |
| verl 三角色与各自引擎？ | Actor（训练，FSDP2/Megatron）、Rollout（生成，vLLM/SGLang）、Ref（冻结 SFT 副本，算 KL） |
| RL 训练时间分布？ | rollout 60-80%（生成瓶颈）、reward 5-10%、training 15-30% |
| 把 PPO 换成 GRPO 的关键配置行？ | `algorithm.adv_estimator=grpo` + `actor_rollout_ref.rollout.n=5`（组大小） |
| HybridEngine 解决什么问题？ | 双引擎间权重回同步（GB 级拷贝）的开销——重分片+原地转换消除显存冗余 |
| 组大小 n 的推荐范围与代价？ | 4-16（常用 8-16 权衡）；rollout 成本与显存随 n 线性增加，收益递减 |
| 为什么 verl 必须用官方 Docker？ | verl 与 vllm/torch/transformers 版本锁步耦合，裸 pip 是依赖地狱 |
