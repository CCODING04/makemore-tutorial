# Assignment 17：Agentic RL

> 对应 Part 17 教程（[01 从单轮到 Agent](../../courses/Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md) / [02 奖励设计与工业框架](../../courses/Part17_agentic_rl/tutorial/02_rewards_and_frameworks.md)）。
> 四题纯 CPU 可完成；题 4 为 🌟 弹性题（不实现返回 None，测试自动 SKIP ⏭️）。

## 题目（实现 `agentic_exercises.py`）

### 题 1：轨迹 loss mask（30 分）

给定各 assistant 段切片，构造 0/1 mask；`assistant_token_fraction` 算 assistant 占比
（Agentic 轨迹典型 10-30%）。

**验收标准：**
- [ ] `build_trajectory_mask([(4, 9), (13, 20)], 24)` 返回长度 24 的 list，
      1 的数量 = 各段长度之和（5 + 7 = 12）
- [ ] 段内为 1，段外（含 user 段与 padding）为 0
- [ ] `assistant_token_fraction([(4, 12), (16, 24)], 32) == 0.5`（两段各 8：16/32）

### 题 2：轨迹级 GRPO（30 分）

逐组标准化优势；**全同组 → 全 0**（无区分度组没有梯度）。

**验收标准：**
- [ ] 返回形状与输入一致（`list[list[float]]`）
- [ ] 组内标准化：`[[0.0, 1.0]]` → `[[-1, +1]]`（容差 1e-6）
- [ ] 全同组 `[[2.0, 2.0], [2.0, 2.0]]` → 全 0（eps 兜底，不除零）

### 题 3：工具调用解析（25 分）

从含任意文本的输出中按序解析 `<tool_call> name arg1 arg2 ... </tool_call>`（正则）。

**验收标准：**
- [ ] 按出现顺序返回 `[{"name": str, "args": [int, ...]}, ...]`
- [ ] 无调用时返回 `[]`（不是 None）
- [ ] args 可为空（`<tool_call> list_dir </tool_call>` → `"args": []`）
- [ ] 混合文本中多个调用全部解析出（顺序正确）

### 题 4：🌟 Echo Trap 检测（15 分，弹性题）

多样性得分 = 不同轨迹数 / 总轨迹数；低分 = 探索坍缩（RAGEN 论文的 Echo Trap）。

**验收标准：**
- [ ] 3 条全同轨迹 → 1/3；全不同 → 1.0
- [ ] 未实现时保持 `return None`——测试将 SKIP ⏭️ 而非 ERROR
- [ ] 用 `set(tuple(t))` 去重（轨迹要可哈希）

## 实验题（观测型）

- 跑[脚本 01](../../courses/Part17_agentic_rl/scripts/01_toy_agent_grpo.py)：把 `G`
  从 8 改成 1（=无组基线），观察学习是否退化（组内标准化消失）
- 把 `MAX_TURNS` 从 2 改到 4，观察成功率与"观测 token 占比"的变化
  （轨迹越长 assistant 占比越低、credit assignment 越难）
- 脚本 01 现已内置**真实掩码消融**（开卷/闭卷 × train/holdout 四格对照）。把判分
  改成"答案位置锚定"（只看 `<eos>` 前模型自己生成的数字），预测并验证：泄漏组的
  开卷·train 还能到 96.9% 吗？

## 🎯 面试直通车

- "Agentic RL 和单轮 RLVR 的实现差异？"——多轮轨迹、观测 mask、轨迹级优势广播、
  异步 rollout、上下文管理（答 3 个即合格）
- "观测 token 为什么 mask 出 loss？"——不是模型的话；算 loss 会教模型幻觉工具结果；
  观测含答案时更要防"复读观测"走捷径（Part 17 消融实测：泄漏组闭卷 10.4% vs
  mask 组 43.8%）
- "Echo Trap 是什么？怎么发现/缓解？"——多轮 RL 的熵坍缩到重复模板；监控轨迹多样性；
  StarPO-S（critic + clip-higher + rollout 过滤）
- "奖励怎么设计？"——稀疏结果 / ToolRL 式塑形（格式+参数+结果分解）/ ORM；
  配课程学习（WebRL 式从失败再生任务）

## 🤔 思考题

**Q1：** 轨迹级 GRPO 把整条轨迹的优势广播到所有 assistant token——一条"第一步调对
工具、最后一步抄错答案"的失败轨迹里，第一步的调用会被强化还是削弱？这暴露了什么
问题、有哪些解法？

<details>
<summary>💡 提示</summary>

会被**削弱**（整条轨迹 reward=0 → 负优势 → 所有 assistant token 一起受罚）。
这暴露轨迹级 credit assignment 粒度太粗：好的中间动作被坏的结果连坐。解法：
① ToolRL 式过程奖励分解（格式/参数/结果分开打分）；② GiGPO 在锚定状态上建
step 级分组；③ 过程奖励模型（但 RM 可被 hack）。教程 02 章概念检验 Q1/Q3
与扩展思考 3 是同一条线索。

</details>

**Q2：** 本课脚本的判分取"轨迹中最后出现的数字"，而第二次工具观测本身就等于答案。
这个漏洞会让什么行为拿到不该拿的奖励？真实 RLVR 怎么堵？

<details>
<summary>💡 提示</summary>

模型甚至不需要自己给最终答案——两次调用参数正确即可让"观测冒充答案"拿分；
更糟的是它可能学会不产出答案直接停。堵法：① 格式约束（`\boxed{}`/`####` 锚定
答案必须在 assistant 段）；② 工具协议判分只看 assistant 输出；③ 答案位置锚定
（只解析 `<eos>` 前模型生成的数字）。对应教程 01 章"玩具判分漏洞"专注说明。

</details>

**Q3：** `echo_trap_score` 只看"轨迹去重比例"。一个策略采样出 8 条**互不相同**但
全部失败的轨迹——多样性 1.0，问题出在哪？你会怎么补全 Echo Trap 的检测信号？

<details>
<summary>💡 提示</summary>

多样性高 ≠ 探索健康：8 条不同但全错的轨迹说明"探索的方差存在但方向无效"。
补全：① 同时看 rollout 熵（token 级分布的熵）；② 看"多样性 × 成功率"的联合
曲线（RAGEN 论文正是发现熵坍缩时奖励曲线看似正常）；③ 看模板重复率随训练步的
斜率，而非单点值。

</details>

**Q4：** 为什么 BC 冷启动几乎是 Agentic RL 的必需品？我们实测"示范砍到 2 个组合"
后 RL 六轮组平均奖励纹丝不动——死锁的链条是什么？

<details>
<summary>💡 提示</summary>

链条：随机/覆盖不足的策略 → 未见组合上采不出能拿分的轨迹 → 组内奖励全 0 →
组内 std=0 → GRPO 优势全零 → 无梯度 → 永远采不出（冷启动覆盖度决定 RL 能否
启动）。R1 论文的 cold start SFT 就是解这个；缓解还有软策略（更高温度）、课程
学习（从易到难）。见教程 01 章"三个必讲的观察"①。

</details>

**Q5：** verl/slime 的 multi-turn 实现是"观测进上下文 + loss mask"，而本课消融把
观测内容整个换成 `<mask>`。两者各自演示了什么原则？工业实现为什么不把观测藏起来？

<details>
<summary>💡 提示</summary>

共同原则：**环境给的 token 不承载策略梯度**（loss mask）。本课玩具额外演示的是
孪生问题：观测**内容**泄漏答案时，策略学会"复读观测"而非内化计算（闭卷/换任务
划分现形）。工业不藏观测，因为模型必须读到工具结果才能用工具——藏掉等于废掉
工具；防捷径靠判分协议与奖励设计，不靠遮观测。

</details>

## ✅ 完成自检

```bash
python test_agentic_exercises.py   # 或 pytest test_agentic_exercises.py
```

- [ ] 题 1-3 全绿（✅）
- [ ] 题 4 实现则 ✅、未实现则 ⏭️ SKIP（均不算失败）
- [ ] 能不看笔记回答"面试直通车"四问
