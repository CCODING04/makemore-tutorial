# S1 学习报告·P12（微调实战 — LLaMA-Factory：LoRA / QLoRA / DPO）

> 学生画像：数学畏难型。凡是"账对不上、公式没来由、文字和数字打架"的地方我都会卡住。
> 审计材料：tutorial/（README + 01 + 02）、scripts/01_handwritten_sft_lora.py、assignments/assignment_12/assignment.md 题面（含骨架 finetune_exercises.py）。
> 逐抠点：LoRA 低秩分解与 α/r、注入矩阵选择及理由、QLoRA NF4/双量化/分页优化器、DPO-LoRA。

---

## 一、总分：7.8 / 10

一句话：公式和数字例给得足（形状图、参数账、实测输出全部能对上），但 **3 处"账本/口径不自洽"恰好打在畏难点上**——QLoRA 双重量化账方向反了、题 4 的 12B/16B 文字拆解打架、练习 3 的显存公式和性能表矛盾。

| 维度 | 评价 |
|---|---|
| 数学正确性（逐条核对） | 主线公式全对：W' = W + (α/r)·BA、A 高斯/B 零 ⇒ ΔW=0、压缩 256×、题 1=3072、题 4=3.74 均自洽 |
| 数字例密度 | 密。README 256×、01 章 1,536/18×/6,144(3.1%)、脚本真实输出逐项可复现（我全对上了） |
| 推理是否进代码块 | 是。两章的"推导过程"都放在代码块 Step 1/2/3 里，畏难友好 |
| 账本自洽性 | **三处不自洽**（见卡点 1/2/3），是主要扣分点 |
| 声明完整性 | QLoRA 三大件缺"分页优化器"；NF4 只有名字没有机制；DPO-LoRA 只有字段没有公式 |

---

## 二、卡点清单（按严重度）

1. **[高] 02 章 QLoRA 双重量化的账方向混乱**（`02_llamafactory_workflow.md` L55-58）
   原文："双重量化……**额外节省 ~0.37GB**，总计: ~3.5GB **+** 0.37GB ≈ 3.87GB"。
   "节省"的量被**加**进了总账，自相矛盾。正确账：4bit 底座 3.5GB + fp32 量化常数（每 64 参数 4B ≈ 0.5 bits/param → 7B ≈ 0.44GB）→ DQ 后常数 ≈0.06GB，总计 ≈3.56GB（或写"3.94 → DQ 省 0.37 → 3.56"）。且 0.37 的来源没讲：QLoRA 论文是 **0.37 bits/param**，7B 折算 ≈0.32GB，教程直接当 GB 用，口径混了。
2. **[高] 题 4 的"16B 拆解"不自洽**（`assignment.md` 题 4 + 思考题 Q1）
   文字拆解"×12B/参数：**fp32 参数+梯度+AdamW 两个动量**"——4+4+4+4 = **16 字节**，公式却按 **12B** 算。学生若照文字实现 16B：3.5 + 0.32 = 3.82 ≠ 3.74，测试红且不知道为什么。12B 能成立的口径其实是"bf16 参数+梯度 + fp32 两动量"（2+2+8，即 01 章练习 3 的口径）或"fp32 master+两动量"（4+4+4）。思考题 Q1"fp32 的参数副本、梯度、一阶/二阶动量（约 12 字节/参数）"同病。**题 4 公式本身（3.74）是自洽的，坏在文字拆解。**
3. **[高] 01 章练习 3 显存公式与同章性能表矛盾**（`01_handwritten_sft_lora.md` L388-392 vs L219-224）
   `estimate_lora_memory` 的"经验公式"按 **model_params_B 全量**算梯度 2B + 优化器 8B，对 7B 必然算出 ≈84GB+ 激活；但同章性能表白纸黑字"7B LoRA r=8 ≈ 16GB"。LoRA 的梯度/优化器账只该算**可训练的 ~20M 参数**（≈0.24GB）。练习无参考答案，按骨架写完的学生没法自查，这是典型畏难崩溃点。
4. **[中] "分页优化器"全教程 0 提及**（grep `paged|分页` 无命中）。QLoRA 论文三大件 = NF4 + 双重量化 + **Paged Optimizers**（NVIDIA 统一内存把优化器状态页出、防 OOM 峰值），02 章 §3 只讲了前两个。既然 01 章给了 QLoRA 配置表，缺这一件学生不知道它是声明的一部分。
5. **[中] α/r 只讲"是什么"，没讲"为什么除以 r"**。两章都写了"α/r 是缩放因子，控制学习强度"+ 常见 α=2r，但"r 变大时 BA 尺度会变、除以 r 让调 r 不必重调 α/学习率"这一句动机缺失——这恰是 LoRA 论文里 α 存在的理由，也是 yaml 里 `lora_alpha: 2×rank` 惯例的来由。
6. **[中] "注入哪些矩阵、为什么"只有字段没有理由**。手写版注入 MLP 两个 Linear（玩具简化，未说明为何不注 attention）；yaml 对照表给了 `lora_target: all` 和"常填 q_proj,v_proj 或 all"（脚本 docstring），但**为什么原论文选 Wq/Wv、为什么现在 all 更好**（QLoRA 论文附录结论）一个字没讲。
7. **[中] LoRA 低秩假设的动机缺失**。README"数学推导"Step 1-3 全是**参数计数**，没有一句"为什么 ΔW 可以低秩"（微调更新的内在秩低、少数方向就够）。对畏难学生，最需要的直觉反而没写；我只能从题 5 的实验设计（秩 4 目标、r≥4 才 loss→0）反推出来。
8. **[低] 02 章 L31 "LoRA bf16 (~20MB)"**：20M 参数 × 2B = **40MB**，差 2 倍（MB 级说法没错，但账不平）。
9. **[低] DPO-LoRA 只有工程字段没有数学**。§5 只给 `pref_beta: 0.1（= DPO 的 β）、pref_loss: sigmoid`，DPO 损失（β·log πθ/πref 的比值项）一句没有；概念检验 Q3 让读 rewards/margins 曲线，却没给 margins = rewards(chosen) − rewards(rejected) 的定义。跨章引 Part 8 03 可以接受，但"一段式最小公式"该有。
10. **[低] 耗时声明两处不一致**：教程 01 章"~3 秒" vs 脚本 docstring"~40 秒"。**GPU 实测 2.41s，教程是对的**，docstring 过度保守（CPU 上可能确实要几十秒，但应写清"CPU ~40s / GPU ~3s"）。

---

## 三、分章评分

| 章节 | 分 | 一句话 |
|---|---|---|
| README.md | 8/10 | 低秩分解参数账 + 256× 数字例进代码块，α/r 有提及无动机；历史脉络清晰 |
| 01_handwritten_sft_lora.md | 8.5/10 | 全教程最佳：形状追踪图 + 字段对照表；扣在练习 3 公式与性能表矛盾 |
| 02_llamafactory_workflow.md | 7/10 | 命令流水线可照抄、观察点好；QLoRA 数学三处欠账（DQ 账方向、NF4 机制、分页优化器） |
| assignment.md（题面） | 8/10 | 题 3 Frobenius、题 5 秩实验设计出色；题 4 文字拆解 16B/12B 打架 |
| scripts/01 | 9/10 | 注释即教材（逐操作形状、2·BA bug 警示、验证方法论）；docstring 耗时与教程不一致 |

---

## 四、只改 3 件事

1. **修 02 章 DQ 账本**：改为"4bit 底座 3.5GB + 常数 0.44GB → DQ 省 ~0.37 bits/param（≈0.32GB）→ 总 ≈3.56GB"，并在 §3 同段补一句"第三个组件是分页优化器（paged optimizer，防 OOM 峰值）"。
2. **统一题 4 的字节口径**：把"fp32 参数+梯度+AdamW 两个动量"改为"**bf16 参数+梯度 + fp32 一阶/二阶动量（2+2+8=12B）**"，思考题 Q1 同步改；或保留 fp32 四件套文字但把公式与验收数字改成 16B 口径（二选一，别再两头不一致）。
3. **修 01 章练习 3**：公式改为"梯度与优化器状态只按**可训练参数**计"，并给一行数字例：7B、r=8 all → 20M×12B ≈ 0.24GB + 底座 14GB + 激活 ≈ 16GB，与性能表对上。

---

## 五、最喜欢 3 处

1. **01 章 ASCII 形状追踪图**（L101-136）：逐操作标形状 `x@A.T: (B,T,96)@(96,4)→(B,T,4)`，且数字闭环——1,536/层 × 2 Linear × 2 Block = 6,144 = 脚本 `[1]` 真实输出，压缩比 18× 也给出来了。对畏难学生这是"看得见的矩阵乘法"，全程不需要想象。
2. **脚本 merge_lora 的 bug 注释与验证方法**（scripts L217-232）：明写"只加不减旁路是经典 bug：输出变 Wx + 2·BAx（实测 max|Δlogits|≈2.9）"，并用**逐元素 logits 差**而非"采样文本看起来一样"来验证合并——既给了坑又教了"什么才算证据"。
3. **"以上为脚本真实输出"的诚实口径**：教程概览的每个数字（6,144/200,664 3.1%、loss 3.572→0.076、回声 2/3 并注明"正常欠拟合"、assignment Q3 的 max|Δlogits|≤2.4e-06 与断言阈值 1e-4）我实测全部吻合（我跑出 2.38e-06）。数字敢写"可复现"且真的可复现，建立信任。

---

## 六、费曼自检：LoRA「直觉 → 公式 → 数字」

- **直觉**（我自答后核对）：微调只需要在少数几个"方向"上修改权重，所以 ΔW 是低秩的——把 d×k 的大更新换成两个瘦矩阵相乘。核对结果：教程只讲了"用低秩矩阵近似权重更新"（定义），**没讲为什么可以低秩**（卡点 7）；直觉这一层我从题 5 的实验设计（秩 4 目标 + r≥4 才打到近零 loss）自己补上了。
- **公式**：前向 `h = Wx + (α/r)·BAx`；合并 `W' = W + (α/r)·BA`（精确加法 ⇒ 推理零开销）；初始化 A ~ N(0,1)/√r、B = 0 ⇒ 起点ΔW=BA=0（Frobenius 范数为 0，"起点无损"）。核对：01 章 Step 1-3 代码块齐全，B=0 的原因（若 A 也为 0 则梯度恒零、死鞍点）在 assignment 思考题 Q2 里补全了，赞。
- **数字**：d=k=4096, r=8 → 65,536 vs 16,777,216 = **256× 压缩**（README，我验算 ✔）；脚本层例 4×96 + 288×4 = 1,536、18×（✔）；全模型 6,144/200,664 = 3.1%（实测 ✔）。
- **自答-核对记录（5 问）**：
  | 自答 | 教程核对 |
  |---|---|
  | α/r 除以 r 是为了让改 r 时 ΔW 尺度稳定 | 只说"学习强度"，动机缺（卡点 5） |
  | LoRA 该注入哪些层？attention 的 Wq/Wv 起家，现在 all 更强 | 只给字段映射，无 why（卡点 6） |
  | QLoRA 的 4bit 只量化冻结底座，A/B 保持高精度 | 02 章 Q1 答案与我一致 ✔ |
  | QLoRA 还有第三件：分页优化器 | 教程未提（卡点 4） |
  | margins = rewards(chosen) − rewards(rejected)，reward 来自 β·log(πθ/πref) | 02 章 Q3 只给读法无公式（卡点 9） |

---

## 七、附录

### A. 脚本实测（scratch：/home/admin02/Code/WorkSpace/makemore-tutorial-review/scratch/S1_P12）

- 命令：`.venv 全路径 python courses/Part12_finetune_llamafactory/scripts/01_handwritten_sft_lora.py`
- 环境：cuda（RTX 4090），**real 2.41s** → 教程"~3 秒"成立（docstring"~40 秒"为 CPU 保守值，卡点 10）
- 输出逐项核对：`[1] 6,144/200,664 (3.1%)` ✔；`[2] loss 3.572→0.076` ✔；`[3] 回声 2/3`（w3 w7 → w19 错）✔；`[4] max|Δlogits| = 5.36e-07 / 1.07e-06 / 2.38e-06，全部 < 1e-4` ✔，与 assignment Q3"≤2.4e-06"吻合
- 作业测试：`test_finetune_exercises.py` 未实现时 ❌/⏭️ 提示清晰，SKIP 机制正常（0.8s）
- 手算核对：题 1 `4×(288+96)×2 = 3,072` ✔；题 4 `7e9×4/8 + 20e6×12 = 3.74GB` ✔（公式自洽，文字拆解不自洽见卡点 2）；题 5 前向公式与脚本 LoRALinear 一致 ✔

### B. 环境阻塞（env-blocked，未跑）

02 章工具链（llamafactory-cli train/webui/export/chat）需 clone LLaMA-Factory + 装 torch 全家桶 + 下载 0.5B/7B 权重 + 1-2h 训练，超出本次 20 分钟窗口，全部记 env-blocked。02 章 0.5B 行"本机实测"无法本次复核。

### C. 逐抠点结论速查

| 任务要求抠的点 | 结论 |
|---|---|
| α/r 缩放讲了吗 | 讲了"是什么"（8/4=2.0 数字例 ✔），没讲"为什么除 r" |
| 注入哪些矩阵为什么 | 玩具注 MLP（未说明简化理由）；q,v/all 只有字段无 why |
| QLoRA NF4/双量化声明 | NF4 一句话、DQ 账本方向混乱（卡点 1） |
| 分页优化器提了吗 | **没提**（grep 无命中，卡点 4） |
| DPO-LoRA | 仅命令 + pref_beta/pref_loss 字段，无公式（卡点 9） |
| 符号/数字例齐全？ | 符号齐、主线数字例密且可复现；0.37GB 来源、40MB/20MB 两处欠账 |
| 推导进代码块了吗 | 是，两章推导均为代码块 Step 格式，畏难友好 |

### D. 读过的文件

- /home/admin02/Code/WorkSpace/makemore-tutorial/courses/Part12_finetune_llamafactory/tutorial/README.md
- /home/admin02/Code/WorkSpace/makemore-tutorial/courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md
- /home/admin02/Code/WorkSpace/makemore-tutorial/courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md
- /home/admin02/Code/WorkSpace/makemore-tutorial/courses/Part12_finetune_llamafactory/scripts/01_handwritten_sft_lora.py
- /home/admin02/Code/WorkSpace/makemore-tutorial/assignments/assignment_12/assignment.md（题面）
- /home/admin02/Code/WorkSpace/makemore-tutorial/assignments/assignment_12/finetune_exercises.py（骨架，题 4/题 5 核对用）
- 未读：assignment_reference/、REVIEW/、.claude/skills/、.zcode/、STUDENT_FEEDBACK 等（按要求禁读）；未联网。
