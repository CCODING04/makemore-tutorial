# Part 6 (Transformer/GPT) 分章审计计划

> 审计人：T1 主教（预审）｜日期：2026-09-04
> 审计对象：REPO/courses/Part6_transformer/（tutorial 5 篇 1527 行 + scripts 7 个 + gpt.py + assignments/assignment_6 376 行）
> 对照材料：docs/course_roadmap_v3.md 节点 6（L238-262，阶段二现代架构，面试簇 A ★★★★★ 重镇）
> 目标读者画像：面试备战人群；本 Part 承载 TOP8 #1「多头因果自注意力白板默写」，是全课程面试密度最高的 Part
> 预审已做实测：仓库 `.venv/bin/python` 实跑脚本 01/03，教程引用输出**逐字吻合**（含词表 65、`encode('hi there')=[46,47,1,58,46,43,56,43]`、train/val=1,003,854/111,540）。系统 python3 无 torch，正式审计一律用 `.venv/bin/python`

---

## 一、C1 标题-内容对应表

### README.md（98 行）
| 标题/承诺 | 实际小节 | 判定 |
|---|---|---|
| 章节导航 4 章 + 脚本映射（01→脚本01/02；02→03/04/05；03→05/06/07；04→走读） | 与 7 个脚本实际分工一致 | ✅ |
| 学完清单 9 条（tokenizer→RLHF） | 各章均有落点 | ✅ |
| 演进 loss 双列对照表 + "看趋势别背数字"免责 | 与各章正文数字一致（2.50/2.39/2.45/2.50†/2.23/2.23/2.80 vs 1.48） | ✅ |
| 学习路线图终点："Transformer 之后（阅读 Karpathy 的 micrograd / minGPT 等）" | **无 Part 7（minimind 组件升级线）指引**；而 Part7 01 章已 4 处回引 Part6 | ❌ 缺口 C1-1 |
| roadmap 学习验证"训后 ppl≈9-11（指南硬数字）" | 全 Part 无 ppl/困惑度字样（初始 loss ln65≈4.17 有，训后 ppl 未换算未提及） | ❌ 缺口 C1-2 |
| "每章末尾有 2-3 道思考题" | 01-04 章各 3 题（details 折叠） | ✅ |

### 01_data_and_tokenizer.md《数据与 Tokenizer：从 ChatGPT 到字符级语言模型》（381 行）
| 承诺 | 实际小节 | 判定 |
|---|---|---|
| "从 ChatGPT" | 课程动机节 + GPT 三字母拆解 | ✅ |
| README 导航 8 项（动机/数据/tokenizer/划分/Dataloader/Bigram/交叉熵/AdamW） | 逐一有落点，输出为实测真实日志 | ✅ |
| BPE/sentencepiece 对比（词表大小 vs 序列长度权衡表） | 有专节 + tiktoken 实例 | ✅ |
| 章末作业映射"题 1（Tokenizer）+ 题 2（Dataloader）" | 与 assignment 对应 | ✅（但见 C1-3：题 3 Bigram 被挂到 02 章） |

### 02_attention_from_scratch.md《数学技巧、Self-Attention 与 6 条笔记》（446 行）
| 承诺 | 实际小节 | 判定 |
|---|---|---|
| 数学技巧 | Part A：v1 for 循环 / v2 tril 归一化 / v3 masked_fill+softmax，allclose 三方等价 | ✅ |
| Self-Attention 单头 | Part B：位置编码→Head 全码→逐行拆解→数据依赖亲和力实测→scaled 方差论证 | ✅ |
| 6 条 attention 笔记 | 逐一独立小节展开（通信/无空间/batch 隔离/三角遮罩/self vs cross/缩放） | ✅ |
| Multi-Head | 分组卷积类比 + cat/proj 形状 ASCII 图 + head_size=n_embd//n_head | ✅ |
| 章末作业映射"题 3（Bigram）+ 题 4（单头 Self-Attention）" | **题 3 Bigram 是 01 章内容**，章-作业映射错位（01 章只挂题 1/2，无人挂题 3） | ❌ 缺口 C1-3 |
| assignment 题 4(a) 要求模块级 `scaled_dot_product_affinity(q, k)` | 教程 Head 为内联实现，正文无此函数拆分与签名 | ❌ 缺口 C1-4 |

### 03_transformer_block.md《FeedForward、残差连接、LayerNorm、Scale Up》（375 行）
| 承诺 | 实际小节 | 判定 |
|---|---|---|
| FeedForward | 4× 内层（论文 512→2048）+ per-token 语义 + Phase2 反常 loss 解释（† 脚注闭环） | ✅ |
| 残差连接 | "梯度超高速公路" + 回链 Part 4 加法梯度规则 + Phase3 最明显一跃 | ✅ |
| LayerNorm | 行/列对比 + 与 BatchNorm 三差异表（running buffer/两态）+ std=1.118 无偏方差数值细节 | ✅ |
| pre-norm | 有 Block 代码 + "别写成 post-norm"顺序警告 + ln_f 归位 | ✅（但 pre-norm 为何更稳仅一句带过，见第四节 A4） |
| Dropout | 三个放置位置（softmax 后/两处残差前） | ✅ |
| Scale Up | 参数统计 0.112M + CPU/GPU 超参对照表 + 150 步真实日志 + "步数不足非架构错"定性 | ✅ |
| 章末作业映射"题 5（🌟 完整 Block）" | 对应 | ✅ |

### 04_beyond_transformer.md《Encoder/Decoder、nanoGPT、回到 ChatGPT》（227 行）
| 承诺 | 实际小节 | 判定 |
|---|---|---|
| Encoder vs Decoder vs 完整架构 | 翻译数据流图 + "删除遮罩行即 encoder" + <START>/<END> 特殊 token | ✅ |
| nanoGPT 走读 | model.py/train.py 分工 + 三细节（4D batched MHA/GeLU 双重动机/参数分组 decay）+ 权重绑定提及 | ✅ |
| 回到 ChatGPT/GPT-3 | 预训练=文档补全器 + 175B/300B 硬数字对比表 + SFT→RM→RLHF 三步 | ✅ |
| roadmap"04 章含 RLHF 预告" | 预告无出口：未链接本课程 Part 8（SFT/RLHF 已建成），展望仅泛提"线性注意力、MoE、Mamba" | ❌ 缺口 C1-5 |

**C1 缺口合计：5 个**（C1-1 README 路线图无 Part7 出口；C1-2 ppl≈9-11 硬数字未落地；C1-3 题 3 挂错章；C1-4 题 4(a) 函数拆分教程无落点；C1-5 04 章 RLHF 预告无 Part8 出口）

---

## 二、roadmap 节点 6 承诺逐条核对

| roadmap 承诺（L240-262） | 落点 | 判定 |
|---|---|---|
| 教程 4 章 + 6 条 attention 笔记 | 实际 4 章+README，笔记在 02 章专节 | ✅ |
| 脚本 01-07 阶梯（数据→bigram→trick→单头→多头FFN→LN→scale-up） | 与脚本命名一一对应 | ✅ |
| 作业 376 行、题 5 带 🌟 属性测试（shape 与不变量） | 属实（三角遮罩上三角=0、softmax 行和、方差 std≈1、残差恒等） | ✅ |
| 能解释：√d 方差论证 / mask 在哪一步 / decoder-only vs enc-dec / pre-norm 为什么稳 / 各头学什么 | 前 3 项弹药充足；"pre-norm 为什么稳"与"各头在学什么"展开薄弱 | 🟡 |
| 手写 TOP8 #1 到默写级（causal mask 位置是判卷点） | 路径通畅（见第四节 A7），但默写版依赖全局变量 | 🟡 |
| 学习验证：脚本 03/04 加权均值可视化 | 脚本 03/04 均为**文本打印**无图（G4 关联）；"可视化"实为打印矩阵 | 🟡 |
| 学习验证：脚本 07 文本可读、ppl≈9-11 | CPU 缩小型 val≈2.80（ppl≈16.4），ppl 数字全 Part 未出现 | 🟡（ppl 即 C1-2） |

---

## 三、章级学生单元划分与预计难度

| 单元 | 内容 | 难度 | 时长 |
|---|---|---|---|
| U0 | README 导航 + loss 演进总表 | ★ | 10 min |
| U1 | 01 章前半：ChatGPT 动机 + 数据 + 字符级 tokenizer + BPE 对比 | ★★ | 25-35 min |
| U2 | 01 章后半：train/val + DataLoader/chunk 多样本 + Bigram + 交叉熵 reshape + generate + AdamW | ★★☆ | 30-40 min |
| U3 | 02 章 Part A：数学技巧三版本（批矩阵乘 + softmax 归一化） | ★★★ | 25-35 min |
| U4 | 02 章 Part B：位置编码广播 + 单头 Head（**全 Part 灵魂，TOP8 #1 素材**） | ★★★★ | 40-60 min |
| U5 | 02 章 6 条笔记 + Multi-Head（cat/proj 形状链） | ★★★ | 30-40 min |
| U6 | 03 章 FeedForward + 残差（梯度高速公路） | ★★★ | 25-35 min |
| U7 | 03 章 LayerNorm vs BatchNorm + pre/post-norm | ★★★ | 25-35 min |
| U8 | 03 章 Dropout + Scale Up + 生成 | ★★ | 15-25 min |
| U9 | 04 章全景走读（无新代码） | ★★ | 30-40 min |
| U10 | Assignment 6（题 1-4 基础 + 题 5 🌟 完整 Block） | ★★★☆ | 2.5-4 h |

- 合计 ≈6.5-10 h，与 roadmap 预算（≈6-10h）吻合 ✅
- 难度峰值在 U4（单头 self-attention）与 U10 题 5（完整 Block 组装）；正式审计对这两单元优先做"零基础可跟随"走查

---

## 四、本 Part 特有审计要点（正式审计逐项执行）

**A1 tokenizer**：词表 65 与 encode 输出已实测自洽 ✅。待审：① 教程 `chars = sorted(list(set(text)))` 是动态词表，按 G11 需检查是否有"换数据小样本翻车"的 ⚠️ 提示（当前无，作业固定 input.txt 风险低但应声明）；② 索引 0=`\n` 与空格区分已强调 ✅；③ 与 Part7 BPE 的"词表 vs 序列长度"权衡已有铺垫，可低成本挂 Part7 出口（并入 C1-1 修法）。

**A2 self-attention / 为什么除 √d_k（TOP8 #1 判卷点）**：方差论证现存两处（02 章 L233-235、笔记 6）+ assignment Q1 提示，逻辑正确（unit gaussian → Var(q·k)≈head_size → softmax 尖锐化 → 缩放回 1 保扩散）。待审：① 论证全为文字+行内代码，**无 LaTeX 推导**（Var(q·k)=Σᵢ Var(qᵢkᵢ)=d 的三行展开缺席）——既是 G2 问题也是面试深度问题，建议正式审计要求补 3 行推导 + 数值验证（assignment 已承诺"测试验证 std≈√head_size→1"，教程侧应有对应演示）；② `* k.shape[-1] ** -0.5` 写法与"除以 √head_size"的等价换算有解释 ✅。

**A3 multi-head 拼接与投影形状链**：单头 (B,T,hs)→cat→(B,T,n_head·hs)→proj→(B,T,n_embd) 链条完整、ASCII 图清晰 ✅；proj"暂时只是形状变换、03 章才发挥残差通路作用"的伏笔处理诚实 ✅。待审：① `n_embd % n_head != 0` 的整除边界无提示；② 04 章 nanoGPT 4D 写法（c_attn 一次投影 + view/transpose）只有片段、无完整可运行版本——面试若考"向量化多头"学生无现成代码（见 A7）。

**A4 残差 + LayerNorm 位置（pre-LN/post-LN）**：Block 代码正确（x + sa(ln1(x)) / x + ffwd(ln2(x))）+ 顺序警告 + ln_f 归位 + "论文 post-norm→现代 pre-norm"声明 ✅。待审：① **为什么 pre-norm 更稳**只有一句带过（"现代标准"），缺"梯度不经 LN 直接沿残差通路回传"的机制解释——roadmap 明确要衔接节点 3 坐标系并作为高频面试题，正式审计应要求补 2-3 句机制论证；② Dropout 三处位置（含"为什么放残差前"）可再对齐 GPT-2 原版顺序。

**A5 feed-forward**：4× 规律 + per-token 独立语义 + 与 attention"通信 vs 计算"分工清晰 ✅。待审：GeLU 在 04 章 nanoGPT 节已讲双重动机，但 03 章 FFN 用 ReLU 时未留"现代用 GeLU/SwiGLU，见 Part7"钩子（跨 Part 出口缺口的一部分）。

**A6 decoder-only 因果 mask**：tril buffer + masked_fill(-inf) + softmax 位置 + encoder"删一行"对照 ✅；assignment 属性测试直接检查严格上三角为 0（判卷点有测试兜底）✅。待审：① `self.tril[:T, :T]` **切片为何必要**（generate 裁剪后 T 可小于 block_size）教程与 assignment 均无解释——审计要点；② mask 发生在 softmax 前的"哪一步"问答已覆盖（assignment Q3 + 笔记 4）✅。

**A7 采样循环 + 白板默写路径评估**：generate（裁剪 idx→取末位→softmax→multinomial→cat）逐行有讲 ✅。**默写路径判定：基本通畅**——02 章 Head 21 行 + 03 章 Block 14 行 + 6 条笔记 + 作业题 4/5 双重属性测试构成完整闭环，causal mask 位置（softmax 前）是显式判卷点 ✅。两处路径缝隙：① 教程 Head/MultiHead 依赖全局 `n_embd/block_size/device`，与 assignment 题 4 的参数化签名（`exercise_4_head(head_size, n_embd, block_size)`）不一致——白板场景学生需自行参数化，建议教程补一个"面试默写版"（全参数化、含 multi-head 向量化、约 30 行）代码框，同时消除 C1-4；② KV Cache/每步全序列重算 O(T) 低效的钩子缺失——Part7 03 章讲 KV Cache，Part6 端应有"我们的 generate 每步重算整个序列，Part7 会修"一句（并入 C1-1 修法）。

---

## 五、scripts 运行档位建议（正式审计执行用）

| 脚本 | 档位 | 预计耗时（.venv，CPU 单线程） | 数据依赖 | 待核对输出 |
|---|---|---|---|---|
| 01_explore_data | 直接跑 | <5s | data/input.txt | ✅ 预验通过（教程逐字吻合） |
| 02_bigram_baseline | 直接跑 | ≈1-2 min（1500 步） | 同上 | 教程日志 2.50/2.52/2.50 与生成样例 |
| 03_attention_trick | 直接跑 | <5s | 无（randn） | ✅ 预验通过；教程引用的 tensor 输出格式（带省略号摘录）与实际打印格式比对 |
| 04_self_attention | 直接跑 | ≈2-4 min（3000 步） | input.txt | val≈2.39 日志 + 亲和力矩阵数字（L213-218 为具体数值，须逐位核对） |
| 05_multihead_feedforward | 直接跑 | ≈2-3 min（400+400+1100 步三 Phase） | input.txt | 2.4545/2.5006/2.2324 三 Phase 数字 |
| 06_layernorm_transformer | 直接跑 | ≈1-2 min（1200 步） | input.txt | 2.2340 终值 + LN/BN 演示数字 |
| 07_scaleup_generate | `CPU_MODE=not cuda` 自动切换 | 缩小版 <30s（150 步）；完整版 GPU A100≈15min/4090≈8min——**审计只跑缩小版** | input.txt | ≈2.8044 + 0.112M 参数 + 乱码样例定性 |

- 数据文件存在（1,115,394 字节，与教程"约 1.1MB/百万字符"一致）✅；脚本用 `__file__` 相对定位数据，不依赖 cwd ✅
- 教程所有输出声称"实跑真实结果"且带 ≈ 免责（README"看趋势别背数字"）；01/03 已证可信，02/04/05/06/07 的日志数字为正式审计实跑核对重点（尤其 04 章亲和力矩阵与 05 章三 Phase 的具体到 4 位小数的数字）
- 环境注意：系统 python3 无 torch，必须用仓库 `.venv/bin/python`（已验证可跑）

---

## 六、格式规范预检（G1/G2/G4/G10）违反位置清单

### G1（编号点必须换行）：0 处
抽查全 5 篇：GPT 三字母、q/k/v、6 条笔记、Dropout 三条、nanoGPT 三细节等并列要点均为列表/独立行 ✅。

### G2（数学一律 LaTeX）：确定 0 处 + 候选 3 处
全 Part LaTeX 使用近零，核心数学均以行内代码/文字承载。候选（需正式审计按"简单数量关系豁免尺度"统一判定）：
- 02 章 L233-235：unit gaussian 方差论证段（"方差大约是 head_size"）——数学关系未用 `$...$`（且与 A2 补推导联动）
- 02 章 L356-364（笔记 6）：同上重复论证段
- 03 章 L141：`sqrt(5/4) ≈ 1.118` 无偏方差说明
豁免判定参考：`-ln(1/65)=ln65≈4.17` 位于脚本实测输出代码块内（01 章 L269-272），属脚本输出，不算违规。

### G4（能画则画）：确定 2 处（缺图）+ 候选 3 处
全 Part 无 images/ 目录、7 个脚本无任何 matplotlib/savefig。可画对象盘点：
- **确定违规**：① loss 演进曲线——README 表格自述"本教程会反复看到这张表"，且 CPU 列 8 个实测点齐备，是最应该画未画的图；② softmax 缩放"扩散 vs 尖锐"对比——02 章 L226-231 有两组实测概率数据，画双柱状/折线图成本极低
- 候选（ASCII 图是否豁免由正式审计定）：③ attention 亲和力权重热图（02 章"有的被重视有的被冷落"文字论证）；④ LayerNorm 行/列归一化示意（现 ASCII）；⑤ multi-head 拼接形状链（现 ASCII）
- 关联：roadmap 学习验证写"脚本 03/04 加权均值可视化"，实际脚本只打印文本——若按 G4 补图，建议图由脚本生成（脚本可加 `--plot` 开关，属课程侧改动，正式审计提出即可）

### G10（正文代码可拼凑运行）：确定 1 处 + 候选 2 处
- **确定**：01 章 L203-209 `get_batch` 使用 `device`（`x.to(device)`），全教程正文从未出现 `device = 'cuda' if ...` 定义行（02/03 章继续沿用）——拼凑即 NameError
- 候选：① 02 章 Head 类使用全局 `block_size`（register_buffer），02 章正文未以代码形式定义（01 章仅文字块提及）；② 03 章完整模型使用 `n_layer/n_head/vocab_size/dropout`，正文超参仅存在于对照表格非代码。若教程惯例允许"全局超参见脚本"则降级为风格问题，需统一尺度
- 输出真实性：教程输出全部自称实跑且 01/03 已实测吻合 ✅；其余脚本日志列入第五节核对清单（无发现"凭记忆写输出"迹象）

**格式违规合计：确定 3 处（G4×2 + G10×1），候选 5 处（G2×3 + G10×2），另 G4 弱候选 3 处待豁免判定**

---

## 七、跨 Part 一致性：与 Part 7 分工声明

| 方向 | 现状 | 判定 |
|---|---|---|
| Part7 → Part6 回引 | `Part7_minimind/tutorial/01_bpe_tokenizer.md` 4 处链接（"必须掌握 Part 6 01 章"、"从 Part 6 结束的地方出发"、压缩率对照表、往返一致性回引） | ✅ 充分 |
| Part6 → Part7 出口 | **零引用**：5 篇 md 中无 Part7/minimind/RMSNorm/RoPE/GQA/KV Cache/SwiGLU 任何字样；README 路线图终点停在"读 Karpathy micrograd/minGPT"；04 章展望泛提"线性注意力、MoE、Mamba" | ❌ **双向引用单侧断裂** |
| 内容分工本身 | 清晰无越界：Part6 只讲标准组件（LayerNorm/绝对位置编码/MHA/ReLU-FFN），现代升级组件全部留给 Part7，无重复讲授 | ✅ |
| roadmap 设计意图 | 节点 6→7 定位为"现代组件升级线"（LayerNorm→RMSNorm、绝对位置→RoPE、MHA→GQA、ReLU→SwiGLU、generate→KV Cache），升级映射是 v3 的核心叙事 | 分工声明缺 Part6 侧落点 |

**修法建议（供正式审计采用）**：在 04 章"总结与展望"加一张"本课组件 → Part7 现代升级"四行对照小表（LayerNorm→RMSNorm、position embedding→RoPE、Multi-Head→GQA+KV Cache、ReLU FFN→SwiGLU）并链接 Part7 README；README 路线图终点改为指向 Part 7；01 章 tokenizer 对比段加一句"Part 7 将亲手实现 BPE"。三处均为低成本高收益修补（同时闭环 C1-1）。

另：04 章 RLHF 三步讲完无去向链接，应挂 Part 8（闭环 C1-5）。

---

## 八、正式审计执行清单（建议顺序）

1. 实跑脚本 02/04/05/06/07（缩小版），逐位核对教程引用日志（第五节清单），重点 04 章亲和力矩阵、05 章三 Phase 4 位小数
2. U4/U10 深查：单头 Head 与题 5 完整 Block 的"零基础可跟随"走查；核对 assignment `scaled_dot_product_affinity` 签名与教程差异（C1-4）
3. G2/G10 候选项统一尺度判定；G4 两处缺图确认并给出图规格（数据源、图型、存放路径 `courses/Part6_transformer/images/`）
4. 核对跨 Part 修补三处的最小 diff 方案（第七节）
5. 汇总 P0/P1/P2 分级，产出 fix_P06.md
