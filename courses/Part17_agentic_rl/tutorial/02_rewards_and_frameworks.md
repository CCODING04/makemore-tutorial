# 02 — 奖励设计与工业框架（Agentic RL 的工程全景）

> 🧭 01 章跑通了最小闭环。本章补齐工程决策：长程任务的**奖励怎么设计**、
> **训练不稳定怎么治**（Echo Trap）、**框架怎么选**、**怎么评估**。

## 学习目标

完成本章后，你将能够：

- ✅ **选型** 奖励类型（稀疏结果 / ToolRL 式塑形 / ORM / 课程学习）并说明各自的
  适用场景与代价
- ✅ **设计** Agentic 奖励的防 hacking 防线（调用去重 / 轮数上限 / 环境沙盒 /
  轨迹抽样审查）
- ✅ **识别** Echo Trap 的症状并说出 StarPO-S 的三个缓解件
- ✅ **配置** verl 的自定义奖励函数（`compute_score`，操作型练习）
- ✅ **给出** 框架与评估基准的选型决策（verl 起步 → slime/verl-agent 进阶；
  τ-bench / GAIA L1 起步）

## 📖 前置知识

- **01 章**：多轮轨迹、观测 mask、轨迹级 GRPO（[传送门](01_from_single_turn_to_agent.md)）
- **Part 11**：GRPO 组内优势与"全同组优势全零"性质（本章 StarPO-S 的 rollout
  过滤直接建立在其上）

## 1. 奖励设计谱系（长程任务的核心难题）

| 类型 | 做法 | 优 | 劣 |
|---|---|---|---|
| 稀疏结果奖励 | 只看最终答案对错（Search-R1） | 简单、不可作弊 | 10+ 轮轨迹的 credit assignment 噪声大 |
| 过程/塑形奖励 | ToolRL：格式奖励 + 每步正确性，细粒度分解 | 信号密、收敛快 | 设计不当会引入偏差 |
| ORM（结果监督 RM） | 训练奖励模型打分 | 可泛化到难形式化任务 | RM 可被 hack、有偏差 |
| 课程学习 | WebRL：从失败任务再生任务，逐步加长 | 与模型能力同步演进 | 系统复杂 |

- 🔑 **ToolRL 的实证**（2504.13958）：把奖励分解为"格式正确 + 参数正确 + 结果正确"
  的细粒度塑形，显著优于单一结果奖励。

### 常见陷阱：reward hacking（Agentic 特有，与 Part 8 07 章污染是近亲）

**症状：** 复读机式工具调用循环、调用"回显 prompt"的工具、打印预期答案作弊、
钻环境 bug——分数上去了，任务没完成。

**原因：** 奖励只度量了任务的"代理指标"，而多轮 + 工具的环境里可钻的空子远多于
单轮——策略学到的是捷径，不是能力（分数与能力脱钩，故与污染是近亲）。

**解法：** 防线四件套——调用去重、轮数上限+溢出惩罚、环境沙盒、轨迹抽样人工审查。

### 调试展示：常见错误与修复

#### 错误 1：reward 全零组占比过高，训练停滞

**症状：**
```
组平均奖励长期贴 0，策略几乎不更新（01 章"没有 BC 就没有 RL"的死锁）
```

**原因：** 合法轨迹太少 → 大量组内奖励全 0 → 组内 std=0 → GRPO 优势全零，
这些组对 loss 无贡献（01 章实测过的反面：BC 覆盖不足时，RL 六轮组平均奖励
纹丝不动）

**解法：**
```python
# 训练中统计全零组占比——占比过高说明"探索不出合法轨迹"，而不是"策略差"
zero_frac = sum(1 for rs in group_rewards if sum(rs) == 0) / len(group_rewards)
# 先 BC 冷启动教会格式；仍偏高则扩大示范覆盖 / 提高组大小 / 降低任务难度
```

#### 错误 2：工具调用解析正则漏配

**症状：**
```
轨迹大量提前终止、奖励恒 0——但人工看模型输出"格式明明是对的"
```

**原因：** 解析正则与协议不匹配（协议是空格分隔、正则却按 JSON 逗号写；或 BC
示范本身不合法 → parse 永远失败——01 章 `parse_call` 注释点名的坑）

**解法：**
```python
# 开训前用一批示范轨迹过 parser，先统计解析成功率（协议一处改动要全链路同步）
ok = sum(parse_call(t) is not None for t in demo_texts) / len(demo_texts)
assert ok == 1.0, "示范都解析不了，RL 一定死锁"
```

#### 错误 3：rollout 长度截断丢最终答案

**症状：**
```
长轨迹批量 0 奖励，且与"模型答错"无关——被截断的轨迹根本没有答案段
```

**原因：** 上下文/步数上限先于 <eos> 触发（01 章脚本的 rollout 有 60 token
上限、闭卷探针 24 步上限），截断轨迹按"无答案 = 0 分"判分——模型被冤枉

**解法：**
```python
# 区分"答错"与"没答完"：截断单独计数，必要时给部分分或不计入分母
truncated = (len(gen) >= MAX_STEPS) and (EOS not in gen)
# 监控截断率；超限走截断/摘要或 partial rollout（见下方 Q2 的三种工程处理）
```

## 2. 稳定性：Echo Trap 与 StarPO-S

多轮 RL 特有的失败模式（RAGEN 论文 2504.20073）：策略熵坍缩到**重复模板**
（同样的工具调用循环往复）——奖励曲线看不出来（模板可能还拿低分），但探索已死。

```
发现：监控 rollout 熵 + 轨迹多样性（不同轨迹比例）
缓解：StarPO-S = critic 辅助 + clip-higher（提高上界探索）+ rollout 过滤（丢弃零方差组）
```

- 💡 我们脚本 01 的组内优势在全同组时归零（Part 11 的性质），本质是同一现象的
  单轮版——"无区分度的组没有梯度"。

## 3. 工业框架选型（2026-08）

| 框架 | star | 特点 | 适合 |
|---|---|---|---|
| **verl** | 23.2k | multi-turn/tool-agent 支持最全（docs 好） | 首选入门与生产 |
| **verl-agent** | 2.3k | GiGPO 官方实现，ALFWorld/TextWorld 玩境 | 小模型 agent RL 研究 |
| slime | 8.3k | 智谱系，Megatron+SGLang，custom generate 灵活 | 大规模生产 |
| rLLM | 5.8k | harness/sandbox 无关的干净 env API | 研究原型 |
| AgentGym-RL | 855 | 多环境开箱 | 教学对比 |
| SkyRL / AReaL | 2.2k / 5.7k | 全异步（长尾轨迹场景） | 大规模 |

> 24GB 实操：verl multi-turn + Qwen2.5-0.5B + 计算器/检索工具（Part 11 环境复用）；
> verl-agent 的 TextWorld 玩境（0.5B/1.5B 友好）。SkyRL/AReaL/AgentGym-RL 按
> 文档定位 ≥8 卡，引用不实操。

## 4. 评估（Agentic 版）

| 基准 | 测什么 | 24GB 可评 |
|---|---|---|
| τ-bench | agent+模拟用户+策略合规（零售/航空） | ✅ 轻量（需 LLM 演用户） |
| GAIA L1 子集 | 真实问题（推理+浏览+工具） | ✅ 文本子集 |
| AgentBench | 8 环境（DB/OS/Web…） | ⚠️ 环境重 |
| WebArena / SWE-bench | 自托管网站 / 真实 issue 修复 | ❌ 小模型≈0%，大模型+重环境 |

## 学完本部分你能...

- ✅ 按任务特征选奖励类型并设计防 hacking 防线
- ✅ 识别 Echo Trap 并说出 StarPO-S 的三个缓解件
- ✅ 给出框架选型决策（verl 起步 → verl-agent/slime 进阶）
- ✅ 为 agent 模型选评估基准（τ-bench/GAIA L1 起步）

**概念检验**

<details>
<summary>Q1: GiGPO 相比轨迹级 GRPO 解决什么？</summary>

A: 轨迹级优势对长轨迹粒度太粗（哪个工具调用是关键的？不可知）。GiGPO 在"锚定状态"
（跨 episode 出现的相同状态）上建 step 级分组，给细粒度 credit 且保持 critic-free
（Part 11 04 章 GRPO 思想的 step 级推广）。

</details>

<details>
<summary>Q2: 工具输出动辄几 KB，多轮后撑爆上下文。三种工程处理？</summary>

A: ① 截断/摘要（Search-R1 只保留相关片段）；②观测入上下文但 mask 出 loss 且
history 压缩；③ partial rollout（上下文满时强制截断轨迹并保留已完成部分的优势）。
verl 的 partial rollout 与 slime 的 context engineering 都在此列。

</details>

<details>
<summary>Q3: 零方差组（组内奖励全同）为什么没有梯度？工程上怎么处理？</summary>

A: GRPO 优势 = (r − mean) / std，组内全同时分子为 0（且 std 也为 0，数值上再被
eps 兜底）→ 整组优势全零 → 对 loss 无贡献。这不是 bug 而是性质："无区分度的组
没有信息量"。工程处理：① StarPO-S 的 rollout 过滤（丢弃零方差组，不浪费更新）；
② 提高组内多样性（更高采样温度 / 更长上下文）；③ 改用更细粒度的奖励分解
（ToolRL 式格式/参数/结果分）让组内出现区分度。01 章脚本里 BC 饱和后
`round` 组平均奖励≈1.0、std≈0 的现象就是它的实例。

</details>

**动手实践**

<details>
<summary>练习 1: 给 verl 写一个自定义奖励函数（操作型）</summary>

**任务：** 在 verl 里用自定义 `compute_score` 替换默认奖励（Part 11 02 章的 Docker
环境可直接复用），实现"ToolRL 式三分奖励"：格式分 + 答案正确分。

**验收标准：**
- [ ] 函数签名 `compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float`
- [ ] 对 `The answer is 42`、`\boxed{42}`、`#### 42` 三种格式都能抽到答案
- [ ] 非法/无答案输出返回 0.0 而不是抛异常（奖励函数崩溃会拖垮整个训练）
- [ ] 在 verl 配置里通过 `custom_reward_function.path` 指向你的文件并跑通一个
      sanity batch（日志里能看到非零奖励）

**步骤提示：**
```python
import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # Step 1: 依次尝试 \\boxed{...} / #### ... / "answer is ..." 三种抽取
    # Step 2: 抽到则与 ground_truth 数值比较（float 化，容忍千分位逗号）
    # Step 3: ToolRL 式分解：格式分(抽到答案) * 0.2 + 正确分 * 0.8
    # Step 4: 任何异常路径都 return 0.0
    ...
```

</details>

<details>
<summary>练习 2: 在本课玩具上实现 ToolRL 式奖励分解（本地可跑）</summary>

**任务：** 把 [scripts/01](../scripts/01_toy_agent_grpo.py) 的 0/1 结果奖励分解为
"格式分 + 参数分 + 结果分"，对照原始稀疏奖励跑 6 轮，比较组平均奖励曲线的前 2 轮。

**验收标准：**
- [ ] `decomposed_reward(real_ids, task) -> (fmt, args, result)` 三元组，各分量 ∈ [0,1]
- [ ] 格式分 = 合法工具调用数 / 2；参数分 = 调用参数与 (a,b)/(p,c) 匹配的比例
- [ ] 总奖励 = 0.2×fmt + 0.3×args + 0.5×result（权重可调，说明你的理由）
- [ ] 打印两种奖励的 round 0-1 组平均奖励对照：分解版应明显更高（信号更密），
      最终成功率不低于稀疏版

**步骤提示：**
```
Step 1: 在 rollout 的判分处，把 re.findall(r"\d+", ...) 的单一判分替换为
        逐段判分（real_ids 里 call 段与 obs 段的边界在构建时已可记录）
Step 2: 格式分：parse_call 成功次数 / MAX_TURNS
Step 3: 参数分：multiply 的 args == (a,b)？add 的 args == (a*b, c)？
Step 4: 结果分：最后一个数字 == answer（保留原判分）
Step 5: 两种奖励各跑一遍 run_experiment，对比 curve[:2] 与最终成功率
```

</details>

## 🧭 扩展思考

没有标准答案——每个问题都值得动手验证后再下结论。

**思考 1：GiGPO 的"锚定状态"搬到本课玩具会是什么样？**
01 章轨迹里，`multiply` 调用后的 `[user | call1 | obs1]` 前缀是天然的锚定状态——
所有任务在"决定第二个调用"时面对的是同构局面。你会怎么在这个状态上建 step 级
分组？轨迹级 GRPO 与 step 级分组各自适合什么轨迹长度/任务结构？
（提示：锚定状态要求跨 episode 可匹配——本玩具靠固定协议长度，真实环境靠什么？）

**思考 2：环境设计反过来决定消融结果。**
01 章的判分漏洞（第二次观测=答案，可冒充最终答案）让泄漏组开卷也能拿分。如果把
判分改成"答案位置锚定"（只看 `<eos>` 前模型自己生成的数字），消融表会怎么变？
泄漏组的开卷·train 还能到 96.9% 吗？这说明了"环境/判分协议"与"算法"之间是什么
关系？（动手：改两行判分代码即可验证你的预测。）

**思考 3：多轮 credit assignment 的下一站。**
10+ 轮轨迹里，第一个工具调用与最终奖励之间隔着几千 token，轨迹级优势把它们一视
同仁地强化/削弱。除了 GiGPO，你还能想到哪些思路（过程奖励模型 / 轮次折扣 /
树搜索重排 / 让模型自己生成子目标）？每种思路各引入什么新问题（奖励模型可被
hack？折扣系数难调？搜索开销爆炸？）。

## 📝 课后作业

👉 [Assignment 17](../../../assignments/assignment_17/)

## 🎓 Part 17 完结：Agentic RL 是 2026 后训练 JD 的第一关键词——你现在拥有
从机制手写（脚本 01）到框架实操（02 章路线）的完整入门。继续：
[面试指南 §7b 方向深挖](../../../docs/llm_interview_guide.md)。

---

[← 上一章](01_from_single_turn_to_agent.md) | [Part 17 README](README.md)
