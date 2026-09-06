# S1 学习报告·P11（alignment_verl）

> 学生画像：数学畏难型。看到公式先慌，靠"手算数字例 + 跑代码对输出"建立信心。
> 审计范围：`courses/Part11_alignment_verl/tutorial/`（README + 01 + 02）、
> `courses/Part11_alignment_verl/scripts/`（2 个）、`assignments/assignment_11/assignment.md` 题面；
> 对照 Part 8 `04_ppo_and_grpo.md` 的 KL 与优势口径。未读任何禁读材料。

## 总分：8.7 / 10

数学讲得比想象中友好（手算验证块是救命的），两个脚本全部跑通且输出与教程宣称逐行一致；
扣分点集中在两处跨章口径不一致没被点名、一处 verl 配置键语义配错。

## 卡点清单（按痛感排序）

| # | 卡点 | 位置 | 痛感 | 说明 |
|---|------|------|------|------|
| 1 | **组内 std 分母两章不一致，教程没点名** | 01 章 L220-223 vs Part 8 04 章 L234-257 | 高 | Part 8 用 torch `r.std(dim=1)`（默认 n−1 样本 std，例子里 adv=±0.87）；Part 11 手算 ±1.0（n 总体 std，GRPO 论文口径）。我在 Part 8 算过 ±0.87，到 Part 11 看到 ±1.0 第一反应是"我又算错了"，慌了 10 分钟。01 章脚注只说"`adv=[0.71,...]` 一类数值出自别的归一化口径"——**Part 8 就是那个"别的口径"，为什么不直说？** 附带：Part 8 是 `std + eps`（1e-4），Part 11 是 `max(std, eps)`（1e-6），作业练习 2 只认后者。 |
| 2 | **k3 推导跳步：`KL = E[exp(d) − d − 1]` 凭什么成立** | 01 章 L230-247、脚本 01 docstring | 中高 | 推导只写了"令 d = log p_ref − log p_new，则 KL = E[exp(d) − d − 1]"。缺了关键一步：采样自 π_new 时 `E[exp(d)] = E[π_ref/π_new] = 1`，所以 `E[k3] = −E[d] = KL(π_new‖π_ref)`。没有这一步，`exp(d)` 从天上掉下来，畏难学生直接跳过推导只背公式。 |
| 3 | **k3 手算例子的输入没在教程正文出现** | 01 章 L260-262 | 中 | 手算写"d₁ = log(0.4) − log(0.5)"，但 0.4/0.5 这些概率只在脚本 01 L226（`logp_ref=[log0.4, log0.6], logp_new=[log0.5, log0.5]`）里有。只读教程的人不知道数从哪来。（手算本身我验算过：0.8+0.2231−1=0.0231、1.2−0.1823−1=0.0177、均值 0.0204，全对。） |
| 4 | **`algorithm.kl_penalty` 被当成系数配置** | 02 章 L279（最佳实践表）、脚本总结 | 中 | verl 里 `algorithm.kl_penalty` 选的是惩罚**类型**（kl/abs/low_var_kl…），系数在 `actor_rollout_ref.actor.kl_loss_coef`（或 kl_ctrl.kl_coef）。02 章表格写"`algorithm.kl_penalty \| 0.01-0.1`"是把类型键填了数值，学生进 Docker 照抄会报错/静默不生效。 |
| 5 | **玩具循环没有 clip，开头却说 GRPO="…→ clip 更新"** | 01 章 L3 vs L272-294 | 低中 | 01 章引言说 Part 8 GRPO 是"采样→打分→组内优势→**clip 更新**"，但脚本 02 的 loss = −(logp·adv).mean() + β·KL 里没有 ratio/clip。其实单步 on-policy ratio≡1 所以能省，但教程一句都没解释，我是自己想通后被脚本 L191"零优势项自动无贡献"间接确认的。 |
| 6 | 脚本 02 注释 shape 笔误 | 脚本 02 L183 | 低 | `d_t = ref_logp - logp  # (P, G)`，实际是 (G, P)（logp 是 (G,P)）。对逐行追形状的读者是一颗小雷。 |
| 7 | Part 8 与 Part 11 的 `k3_kl` 参数顺序相反 | Part 8 `k3_kl(new_logp, ref_logp)` vs Part 11 `k3_kl(logp_ref, logp_new)` | 低 | 内部都算 ref−new，数学没错；但同名函数换个 part 参数就反过来，对照两章代码时容易抄错顺序。（作业练习 3 用 Part 11 顺序，OK。） |

## 分章评分

| 材料 | 分 | 一句话 |
|------|----|--------|
| 01_handwritten_to_verl.md | 9.0 | 数学主线（抽取链→组内优势→k3→装配）符号全、有手算、有数字例，推导都进了代码块/docstring；扣在卡点 1/2/3/5。 |
| 02_verl_quickstart.md | 7.5 | 配置行与 01 章手写件逐条对得上，日志标注"示意非实录"很诚实；扣在 kl_penalty 键语义错（卡点 4），Docker 部分本机无法验证。 |
| README.md | 9.0 | 环境分层表（CPU/1 卡/2 卡能干嘛）对穷学生极友好，版本耦合警告提前打预防针。 |
| scripts/01_reward_and_bridge.py | 9.5 | 0.016s 跑通，6 例单测带 assert；docstring 里"数学推导/数据流/常见陷阱"三段式是畏难学生最好的结构。 |
| scripts/02_grpo_toy_train.py | 9.0 | 1.32s 跑通，torch 版与 math 版 k3 互验 assert（L186-188）是全章最亮的教学设计；扣 shape 笔误。 |
| assignment_11/assignment.md 题面 | 9.0 | 验收标准与教程/脚本完全对齐（含"max(std,eps) 不是 std+eps"这种细节），单组/批量关系在思考题 Q2 讲清了。 |

**脚本运行实录**（scratch：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S1_P11/`，venv 全路径 `makemore-tutorial/.venv/bin/python`，torch 2.6.0+cu124）：

- `01_reward_and_bridge.py`：exit 0，real 0.016s；输出与教程"实测输出"[1][2][3] 逐行一致（含 KL=0.0204）。
- `02_grpo_toy_train.py`：exit 0，real 1.323s；step0/4/9 各行、零梯度组 [1,6,6,6,6,6]、奖励 0.38→0.83、BC 0.50→1.00，与教程逐行一致。
- 02 章 Docker quickstart：**env-blocked**（本机 `docker info` 不可用，daemon 无法访问；GPU 本身在位）。无 time-blocked 项。

## 只改 3 件事

1. **01 章 k3 小节补两行**：手算验证前先写明输入（"logp_ref=[log 0.4, log 0.6]，logp_new=[log 0.5, log 0.5]"），推导里补一句"因为采样自 π_new，E[exp(d)] = E[π_ref/π_new] = 1，所以 E[exp(d)−d−1] = −E[d] = KL"。这两行能把卡点 2、3 一起消掉。
2. **01 章 ±1.0 脚注点名 Part 8**：把"别的归一化口径"改成具体的话——"Part 8 04 章例子里的 ±0.87 用的是 torch `.std()` 的 n−1 样本 std；本教程与 GRPO 论文一致用 1/G 总体 std，故 ±1.0。eps 处理同样不同（max vs +）。两处都对，分母口径要盯住。"并顺手统一 `std + eps` vs `max(std, eps)` 的推荐（作业认后者）。
3. **02 章最佳实践表修 kl_penalty 行**：`algorithm.kl_penalty` 填类型（如 `low_var_kl`），系数另起一行 `actor_rollout_ref.actor.kl_loss_coef`（0.001-0.1 常用），与脚本 02 注释里的写法对齐。

## 最喜欢 3 处

1. **脚本 02 的双实现 k3 互验 assert**（L186-188）：同一个公式 torch 版和纯 math 版各算一遍、逐位比对——这是在教"数学别信单次计算，要自证"，对畏难学生是安全感本身。
2. **"零梯度组的两种命运"**（01 章 L334-337 + 脚本 02 总结）：全对=feature、全错=盲区，而且敢让 GRPO 停在 0.83、BC 到 1.00，用诚实的数字讲清"RLVR 的适用边界"，再顺手连到 DAPO dynamic sampling——一个现象讲出了三代算法的因果。
3. **prompt0 手算验证脚注**（01 章 L220-223）：mean=0.5、std=0.5、A=±1.0 三步全摊开，配"留意分母是什么"的提醒（可惜没点名 Part 8）。这是我全章唯一没慌地读完的公式段。

## 费曼自检：GRPO 手写 → verl 映射讲法

我不看教程，用自己的话讲一遍（先自答，再对照教程核对）：

> 你手写的 GRPO 循环每步干五件事，verl 把每件事变成一个"角色/配置"，数学一个字没换：
> ① `for step: 采 G 个回答` → **rollout 角色**（vLLM/SGLang 的 `generate_sequences`），因为真实模型生成占 60-80% 时间，必须换高吞吐引擎；
> ② `gsm8k_reward()`（\boxed → #### → 最后数字的抽取链）→ **custom reward function**，quickstart 里你唯一必写的代码；
> ③ `group_advantages()`（A=(r−mean)/std）→ `adv_estimator=grpo` 一行配置；
> ④ `ref_policy + k3_kl`（exp(d)−d−1，d=log_ref−log_new）→ **ref 角色**（SFT 冻结副本）+ KL 惩罚配置；
> ⑤ `opt.step()` → **actor 角色**（FSDP2 训练）。
> 两引擎分开的代价：每步更新后要把新权重搬回推理引擎——这就是 **weight sync**，verl 的 HybridEngine 用重分片+原地转换压这个开销。
> 一句话：**算法是配置项，基建才是 verl 的本体。**

核对结果：五条映射与 01 章 L553-567 表、02 章角色表全部对上，讲得通。**自答时卡住的两处**：(a) ④ 里我脱口而出"KL 系数配 algorithm.kl_penalty"——正是卡点 4 的错误来源，说明教程这个口误会真的传给读者；(b) weight sync 方向我犹豫了一下"是 rollout 同步给 actor 还是 actor 给 rollout"（答案是 actor→rollout，教程 02 章 L124 写对了）。结论：**映射主线讲透了，但 kl_penalty 这一处会把学生带沟里，必须修。**

## 附录

- 验算过的数学（全部通过）：prompt0 优势 ±1.0（总体 std）；prompt2 优势 −0.58/1.73（mean=0.25、std=0.4330）；k3 手算 0.0231/0.0177→0.0204；ΣA=0 断言脚本内建；exp(d)−d−1≥0（e^x≥x+1）；02 章"7B 训练状态 ≈112GB"=7B×16B 与 Part 10 的 16Ψ 账本一致。
- 与 Part 8 口径比对结论：**k3 一致**（同为 d=log_ref−log_new、exp(d)−d−1；Part 8 称"无偏"，Part 11 称"低方差"，在采样自 π_new 的方向上 E[exp(d)]=1、无偏成立，两说法不矛盾）；**组内优势不一致**（std 分母 n vs n−1；eps 用法 max vs +），教程未点名，见卡点 1。
- 抽取链边界自测（教程/脚本未覆盖但读代码推演）：`\boxed{1,234}` 逗号在框内→OK；`\boxed{42.}` 尾点→rstrip OK；`"答案 42（满分 100）"` 会取 100 判错——"最后数字"启发式的固有盲区，教程陷阱 1 已提示要防。
- 诚实性检查：02 章日志标注"示意，非本机实录"、性能表标注"课程设计推算"——两处都主动声明了非实录，好评。
- 读过的文件：tutorial/README.md、01、02；scripts/01、02；assignments/assignment_11/assignment.md；对照 Part8/tutorial/04_ppo_and_grpo.md（grep 口径段）。未读禁读材料，未联网。
