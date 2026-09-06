# Part 9（CUDA 内核编程）分章审计计划 — T1 主教预审

- 预审日期：2026-09-04
- 对象：`REPO/courses/Part9_cuda_kernels/`（tutorial 6 个 md 共 1722 行 + scripts 9 个源文件 + Makefile）+ `assignments/assignment_9/assignment.md` + `docs/course_roadmap_v3.md` 节点 9
- 性质：全课程唯一硬依赖 GPU 的 Part（v3.1 新增 05 章 Flash Attention 毕业内核，543 行为全 Part 之最），按两学生 × 两单元分工
- 预审方式：只读通读 + 环境只读探测，未运行任何脚本、未改任何课程文件

---

## 〇、本机 CUDA 档位判定（主教实测，2026-09-04）

| 项 | 实测值 | 结论 |
|---|---|---|
| GPU | RTX 4090 24GB（sm_89 / CC 8.9），驱动 550.120（CUDA 12.4 capable） | ✅ 全档可跑 |
| nvcc | 系统 PATH 为 11.8 (V11.8.89)；`/usr/local/cuda-11.8` 与 `/usr/local/cuda-12.4` 并存，`/usr/local/cuda` 软链指向 11.8 | ✅ 与 README"多版本共存"声明逐字吻合 |
| gcc | 默认 gcc 12.3.0，gcc-11 / gcc-12 均在 PATH | ✅ Makefile 的 `-ccbin gcc-11` 分支可生效 |
| python | 系统 python3 无 torch；**课程环境为 `REPO/.venv`：torch 2.6.0+cu124 / triton 3.2.0 / cuda available=True** | ✅ 与 README/05 章声称的版本完全一致 |
| 既有产物 | `scripts/bin/` 01-06 六个二进制已编译；`~/.cache/torch_extensions/py312_cu124/` 有 polynomial_activation_part9 | 说明 8/31-9/2 期间实跑过，实测数字可信源就在本机 |

**审计可执行档位：全档**。`.cu` 脚本可编译运行（`make` 走 12.4 自动选择分支，无需 gcc-11 回退）；07/08/09 可用 `.venv/bin/python` 直跑。注意 4090 当前有其他任务占用（显存 5.3GB/利用率 3%），05 章"共享 vs 独占"计时口径审计时必须复现两种条件之一并声明（G14）。脚本 09 宣称 2-4 分钟（autotune 大头），审计排期时按 5 分钟预算。

## 一、C1 标题-内容对应表（6 文件逐个）

| # | 文件（行数） | 标题 | 实际内容 | 对应脚本 | 章末作业映射（实况） | C1 判定 |
|---|---|---|---|---|---|---|
| 0 | README.md (176) | Part 9: CUDA 内核编程——打开深度学习的引擎盖 | 章节导航表、前置知识（C 子集/Part 6/7/8）、多版本 CUDA 共存指南、编译坑点、两块实测参考（matmul 阶梯 + FA）、作业入口 | 全部 9 个 | "题 1-4 纯 CPU；题 5 Triton 实战" ✅ 与 assignment 实况一致 | ✅ 对应；实测表与 02/05 章数字互相咬合 |
| 1 | 01_gpu_and_first_kernel.md (240) | GPU 架构与第一个 CUDA 内核 | CPU vs GPU 哲学、grid/block/thread/warp 层级图、"够用的 C"、nvcc 编译流水线、vector add 五步曲、线程层级实操 | 01, 02 | 题 1（索引数学）+ 题 2（行/列主序）✅ 映射正确 | ✅ 对应 |
| 2 | 02_matmul_optimization.md (256) | matmul 优化阶梯：从 naive 到 cuBLAS | roofline/算术强度、合并访存（9 倍）、L1-L5 阶梯（每级"省什么"）、通向 cuBLAS 差距表、面试映射表 | 03, 04 | 题 3（tiling 账本）+ 题 4（GFLOPS 报告）✅ | ✅ 对应；**naive 4697.8（脚本03）vs L2 coalesced 4844.9（脚本04）两组近邻数字的关系未点破**（见 §三-2） |
| 3 | 03_profiling_and_cuda_apis.md (274) | Profiling、Atomics、Streams 与 CUDA 库 | nsys/ncu 实测（SOL 表）、NVTX、树形归约（77 倍）、streams"反而更慢"教训、cuBLAS 列主序恒等式、CUDA_CHECK | 05, 06 | 题 3/4"延伸"（措辞与题号不冲突）✅ | ✅ 对应；**ncu 表中 Memory 92.01% 与 Compute 92.01% 数值全同，疑似誊写错误**（见 §三-3） |
| 4 | 04_triton_and_extensions.md (233) | Triton 与 PyTorch 扩展：通向 llm.c | CUDA vs Triton 哲学对照、vecadd/softmax 内核、扩展三件套（dispatch/restrict/pybind）、autograd.Function、毕业去向三路线 | 07, 08 | 题 5（Triton softmax）✅ | ✅ 对应 |
| 5 | 05_flash_attention.md (543) | Flash Attention：亲手写出毕业内核 | naive 显存账本、FA1→FA3→Flex→Sage 演进表、online softmax 三步推导（2.1-2.3）、内核逐行（3.1-3.6）、SDPA 四后端实测、四陷阱、生态、两练习 | 09 | **全章无"📝 课后作业/Assignment"节**（01-04 章均有；只有"练习与思考"两练习） | ❌ 缺口 1：章末作业指引缺失（README 承诺"每章末尾有思考题"满足，但作业入口链断） |

**C1 缺口合计：3 确认 + 1 待核**
1. 05 章无 Assignment 9 入口链接（其余各章均有；05 章练习 1/2 可视作章内作业，但格式不齐）。
2. roadmap 节点 9"学习内容安排"第 1 条只列"教程 4 章：01→04"，**未含 05 章**（教程实为 5 章；roadmap L607 自己都说"Part 9 新增手写 Triton FA 内核章"）——课程地图与教程章节清单失同步。
3. roadmap L600 技能表验收线"Triton 内核 ≥ SDPA 最优后端 **105%**"vs 教程/README/脚本 09 的承诺"**≥50% 合格 / >85% 优秀**"（主表实测最慢场景 94.6%，达不到 105% 线）——三方口径冲突，审计须裁定以教程验收线为准并修 roadmap。
4. （待核）03 章 ncu SOL 表 Memory=Compute=92.01 数值全同：naive 的 L1/TEX 94.15 独高、DRAM 3.36 合理，但两主指标精确相等不符合真机输出常态，复跑 `ncu --section SpeedOfLight` 即可裁定。

## 二、学生单元划分（两学生 × 两单元；05 章为重中之重）

| 单元 | 文件 | 行数 | 难度画像 | 审计重点 |
|---|---|---|---|---|
| **单元 A** | README + 01 + 02 | 176+240+256 ≈ 672 | README ★★（口径源）、01 ★★（入门）、02 ★★★★（roofline + 阶梯数字群，本 Part 数字密度第二高） | 线程层级讲解正确性（warp=32 锁步、block≤1024 须同 SM）；索引公式与脚本 01/02 互证；阶梯六级数字与脚本 04/06 实测对表；coalescing 9 倍、理想强度 85.33 vs 屋顶线 ~82 的推导；"够用的 C"技术事实；README 多版本 CUDA 指南与本机实况逐条核对（主教已预验，全部吻合） |
| **单元 B** | 03 + 04 + 05 | 274+233+543 ≈ 1050 | 03 ★★★★（profiling 实测 + 归约）、04 ★★★（Triton/扩展）、05 ★★★★★（全课程最高难：online softmax 推导 + 内核逐行 + 实测验收，543 行） | nsys/ncu 表格与真机复现；归约 77 倍算术（1.396/0.018）；cuBLAS 列主序恒等式正确性（`C^T=B^T@A^T`、参数序 N,M,K）；Triton 版本兼容声明；05 章 online softmax 推导逐步核对（§三-4 专列）；SDPA 四后端数字与验收线；陷阱 1-4 技术事实 |

- 学生 1 → 单元 A；学生 2 → 单元 B。两学生都须通读 README（全局数字口径源）；单元 B 须回看 02 章 L3（SMEM/FA 预告段）作为 05 章接口。
- 05 章建议学生 2 按四遍读：①推导（§二）②内核（§三）③实测（§四）④陷阱/生态（§五/六），每遍独立产出问题清单。
- assignment.md 与 scripts 由主教在汇诊阶段统一裁定，两学生均可引用、不单独分派。

## 三、本 Part 特有审计要点（主教预标 + 学生核查清单）

1. **线程层级（grid/block/warp）讲解**（01 章 L86-112）：
   - 事实核：warp=32 锁步（简化说法，现代架构线程级调度下"基本成立"，教程表述可接受但审计注意是否有过度声明）；block≤1024 且全 block 驻留同一 SM；`blockIdx*blockDim+threadIdx` 公式与"第几班×每班人数+班内学号"类比自洽；warp divergence 描述与 `if (i % 2)` 例子。
   - 与脚本 `02_thread_hierarchy.cu` 互证：16 线程乱序打印、1D/2D 等价验证是否真在脚本里实现（G10：正文声称"脚本验证了这一点"须脚本里真有）。
2. **matmul 优化阶梯数字可复现性**（02 章 ↔ `03_naive_matmul.cu` / `04_matmul_tiled.cu` / `06_cublas_sgemm.cu`，README L124-139）：
   - 六级数字群：L1 553.6 / L2 4844.9 / L3 5905.4 / L4 6967.5 / L5 8795.2 / cuBLAS 22163.1，naive 4697.8——审计须真机复跑对表（±浮动可接受，量级/排序不可翻）。
   - **级间倍数自洽**：L1/L2=8.75x（教程称"~9 倍"✅）；cuBLAS/L5=2.52x（教程称"2.5 倍"✅）；assignment 思考题 Q2"cuBLAS ~22 TFLOPS vs 峰值 ~82"与 roadmap 验证数字一致 ✅。
   - **待点破处**：脚本 03 的 naive（x→col，已合并）4697.8 与脚本 04 的 L2 coalesced 4844.9 是两个几乎同构实现，教程只说"脚本 03 已经是合并的"，未解释两者 3% 差异来源（BLOCK_SIZE 32 vs 04 的 tile 参数？计时协议差异？）——登记为低危口径缺口。
   - **512³ 规模自首**：教程 Q1/ncu 段/assignment Q3 三处都自认"512³ 全进 L2 会低估 tiling 收益"，三处口径一致 ✅（这是 G6/G14 的正面样本）；审计确认 4096³ 重测指引只是思考题、无未兑现的正文承诺。
3. **atomics/streams**（03 章 ↔ `05_atomics_streams.cu`）：
   - 树形归约：两层结构（SMEM 树形 8 步 + block 间 atomicAdd）与脚本实现互证；"100 万次 atomic 1.396ms vs 3907 次 0.018ms = 77 倍"算术核对；`sum_naive_wrong` 无同步的竞争条件叙述与"跑出过 3.88"实测轶事。
   - streams"重叠反而更慢"（1.026 vs 0.942ms）：教程已诚实归因（块太小 + 现代拷贝引擎本就重叠）并声明脚本简化（4 chunk 共用 `d_x` 不校验数值）——审计确认声明醒目度足够（G12/G14 正面样本，但检查"演示简化说明"引用块位置是否显眼）。
4. **Flash Attention 分块 softmax 推导**（05 章 §二，重点中的重点）：
   - 推导三步是否"讲透"：2.1 整行障碍（T=4096 一行 fp32=16KB 的数字核对：4096×4B=16KB ✅）→ 2.2 两块合并 α 推导（`l_new=l₁·α+l₂, α=exp(m₁-m_new)`，分子 `o_new=o₁·α+o₂`；检查"分子分母同乘 exp(-m_new) 结果不变"的表述是否严格）→ 2.3 exp2 换底（`qk_scale=sm_scale×log2e`，"m_new 也是 log2 域"的自洽性）。
   - **判定标准（主教预判：已讲透）**：m/l/acc 三状态语义、首次迭代 `m_i=-inf ⇒ α=0` 清空历史、`α≤1` 且多数块 α=1 零损耗——三处均有；学生补充核对：α 推导只写了分母侧 l，分子侧 o 的 α 修正只给结论未展开（L125"输出的分子部分同理乘 α"）——教学取舍可接受，但审计确认 3.2 节代码⑤处有无补齐。
   - 内核逐行（3.2 五行核心）与脚本 `_fa_fwd_inner` 实现逐行对照（G10/G15：变量名、`-1.0e6` 常量、`p.to(v.dtype)` 位置）。
   - causal 三阶段（3.3）：`diag_lo` 对齐 BLOCK_N 的坑位声明（教程明说官方 tutorial 隐式避开、我们显式对齐）——与脚本 09 段 2 打印输出互证；"causal 比 full 快 ~1.6×（0.279/0.436）"算术核对（=1.562，教程 Q3 解释三笔开销）。
5. **Triton 版本兼容**：教程声明"torch 自带 triton，Linux 无需单独装"，实测口径绑定 triton 3.2.0/torch 2.6.0——审计确认 04/05 章有无"其他版本可能不同"的声明；`@triton.jit` 嵌套定义坑在 04 章与 assignment 题 5 两处重复且一致 ✅；`tl.math.exp2` 在新版 Triton 的命名空间变化（`tl.exp2`）是否需要兼容注——学生查 `.venv` 内 triton 3.2 实况后裁定。
6. **无 GPU 降级路径真实性**：README/01 章声称"概念全可读、Colab/Kaggle 可跑、作业题 1-4 纯 CPU 可完成"——审计三件套：①assignment `test_cuda_exercises.py` 题 5 无 GPU 时是否真优雅跳过（⏭️ rc=0）；②题 1-4 是否零 torch GPU 依赖（题 2 用 `torch.matmul` 对照——CPU torch 即可？还是 import torch 就崩？读测试文件核实）；③`cuda_exercises.py` 骨架题 5 `triton_softmax` 未实现返回 None 的约定是否成立。roadmap L72"除 Part 9 题 5 / Part 14 实验题外均可纯 CPU 完成"与此互证。
7. **PyTorch 扩展坑点**（04 章 L148-154 ↔ Makefile ↔ 本机实况）：CUDA 11.x→gcc-11、12.x→无参数、CUDA_HOME 缓存须覆盖 `torch.utils.cpp_extension.CUDA_HOME`、ninja 依赖、`~/.cache/torch_extensions` 清缓存——六条全为本机真实验证过的（主教已确认 gcc-11 在位、扩展缓存目录存在），审计核对脚本 08 是否真按此实现。

## 四、scripts 运行档位（9 源文件 + Makefile）

| 脚本 | 依赖 | 预计耗时（4090 实测口径） | 版本敏感点 |
|---|---|---|---|
| 01_vector_add.cu | nvcc | 编译秒级 + 运行 <1s | 无 |
| 02_thread_hierarchy.cu | nvcc | <1s | 无 |
| 03_naive_matmul.cu | nvcc | <1s | 无 |
| 04_matmul_tiled.cu | nvcc | <1s（六级全跑） | `-arch=native` 需与本机 sm_89 匹配（CUDA 11.8 起支持 native；11.8 对 sm_89 恰好可用） |
| 05_atomics_streams.cu | nvcc | <1s | 无 |
| 06_cublas_sgemm.cu | nvcc + `-lcublas` + rpath | <1s | **多版本运行时库**：Makefile rpath 焊入；审计可用 `ldd bin/06_cublas_sgemm` 验证解析版本 |
| 07_triton_kernels.py | .venv (torch+triton) | <1min | triton 3.2 与教程口径一致 |
| 08_pytorch_extension.py | .venv + ninja + nvcc(12.4 自动选) | 首次 JIT 编译 1-2min，产物已缓存 | gcc 上限分支 + CUDA_HOME 缓存覆盖（04 章 L148-154 六坑全对应） |
| 09_flash_attention_triton.py | .venv + GPU（教程称需独占） | **2-4 min（autotune 16 组合编译大头）**；审计按 5min 预算、G16 先拷 scratch | SDPA 后端锁定 API（torch 2.6 `sdpa_kernel`）、`create_block_mask` 签名、共享 vs 独占计时漂移（README 自述两次实测 95-127% vs 105-163% 且跨 4090/4090 D） |

- `make run` 全套 .cu：约 1-2 分钟。`make NVCC=` 强制版本分支审计时不必需（自动选 12.4）。
- nvcc 版本敏感点汇总：11.8 是支持 sm_89 的最低版本（本机默认 PATH 恰为 11.8，编 .cu 可行）；但 torch 扩展（脚本 08）走 torch 侧 CUDA 12.4 工具链，与系统 PATH 的 11.8 无关——审计不要被 `nvcc -V` 显示 11.8 误导。

## 五、格式规范预检（违反位置清单；学生复核后定稿）

**G1 编号点换行（1 处硬 + 2 处待核）**
- 03 章 L250：Q2 答案"② 权重加载/量化转换与推理流水线重叠；③ 多路互不依赖的 batch 并行"两个编号挤同一行（① 在上一行行尾）。
- 待核：05 章 L337-340（"①只做前向…②autotune…③小形状"折行分布）、05 章 L471-473（Q3 答案"①②③"）——每行含一个以上编号即违规。

**G2 数学一律 LaTeX（5 处，本 Part 最大格式问题群）**
- 02 章 L87-91：算术强度推导（"2MNK FLOPs…≈1 FLOP:1 读"）用代码块承载。
- 03 章 L187-198：列主序恒等式推导（`C^T=(A@B)^T=B^T@A^T` 及 cublasSgemm 参数块）用代码块承载——注意参数调用块是代码可保留，恒等式推导须转 LaTeX。
- 05 章 L100-102：数值稳定 softmax 公式；L112-115：m₁/l₁/o₁ 定义；L119-123：l_new α 重缩放推导——毕业章核心推导全在代码块里，与 G2"禁止用代码块承载数学推导"直接冲突。**修复时注意下标 Unicode（m₁/l₁/o₁）与 `$...$` 的转换**；`$$` 单行闭合、公式内无中文。
- 轻微候选：assignment 题 3/4 公式（`M*N*2*K`、GFLOPS 公式）写在普通文本/行内代码——按 G2 口径评估是否需 `$...$`。

**G4 能画则画（全 Part 缺图，1 项结构性缺口）**
- tutorial/ 无 images/ 目录、6 个 md 无一张 PNG。最佳画图点：02 章 roofline 包络图与阶梯 GFLOPS 柱状图、05 章 causal 三阶段网格图（现是 ASCII）与 T 扫描耗时曲线。审计按"成本收益"分级：roofline 柱状图列为建议项，ASCII 图（01 章 L90-104、05 章 L154-169/201-210）本身质量高可保留。

**G10 可拼凑运行（无硬伤，2 处待核）**
- 01 章"跑 scripts/01_vector_add.cu"等链接路径均实存 ✅；04 章正文裁剪版代码（L104-124）省略号截断处是否仍"按序可拼"由学生核。
- 05 章练习 1 提示块引用 Part7 脚本 05 路径实存性（`../../Part7_minimind/scripts/05_full_model.py`）。

**G13 外部来源数字双列制（整体优秀，1 处待核）**
- 05 章做得最好：FA 论文演进表（2×/约 3×/90% FA2）全部带 arXiv 链接 + 教程实测同表双列 + "4090 用不了 FA3"的适用性判定 ✅；FlexAttention "90%（A100）"与本机 94%-108% 双列 ✅。
- 待核：05 章 L317"bf16 尾数 8 位 ~0.4% 起步"、L435"40 系 INT4 吞吐极高（论文实测 INT8 只有 INT4 一半）"——前者为通用常识、后者有论文背书但未逐字对论文表号；02 章 L194 Tensor Core 描述为概念层无数字，均低危。

**G14 实跑口径（4 处轻违规：缺日期/环境完整度）**
- 02 章 L55："实测（4090，512×512×512，fp32）"——无日期、无 driver/toolkit 版本。
- 04 章 L87："实测（4090）"与 L138 实测块——口径最简，无日期无版本。
- README L124："实测参考（RTX 4090，fp32，512³ matmul）"——无日期（但 FA 段 L140-141 有完整口径：torch 2.6.0+cu124/triton 3.2.0/2026-09-02，同文件内标准不一）。
- 01 章 L166："实测输出（4090，N=100 万）"——无日期。
- 正面样本：03 章 nsys 段"（4090，driver 550.120，CUDA 12.4）"、05 章 §四标题全口径、README L149 两次运行的"独占 vs 共享"公平性声明。**审计建议以 05 章 §四口径为全 Part 统一标准回填。**

**合计：格式违规 6 处硬（G1×1 + G2×5）+ 8 处待核/轻（G1×2、G10×2、G13×2、G14×4 中已计 4 轻）**；G3/G5/G6/G7/G8/G9/G11/G12/G15/G16/G17 未发现候选（G6 正面：全部实测数字来自本机真跑且脚本无硬编码文案迹象；G8 注意 09 脚本 2-4 分钟已有分段打印，确认 flush）。

## 六、跨 Part 一致性

1. **与 Part 14（推理部署）分工**：05 章 L538-539 明示"Part 10 分布式与 Part 14 vLLM，同一个内核出现在不同系统位置"——审计时在 Part 14 侧反查：vLLM 章讲 PagedAttention/连续批处理时应引用 Part 9 的内存墙/内核语言而非重讲原理；02 章 L204 面试表把"LLM 推理 memory-bound"指向 Part 7/8（Part 14 不在表内，因 v3.1 时序）——检查 Part 14 是否有反向引用 Part 9。
2. **与 Part 8（后训练/推理服务）分工**：02 章 L204-207 面试映射表把 memory-bound/tiling/bf16 三概念锚到 Part 7 KV Cache、Part 8 autocast；04 章"路线 C"让学生回 Part 7/8 用 nsys 复看——审计确认 Part 8 06 章（推理与服务，讲过 RTN/GPTQ/AWQ）与 Part 9 无量化内容重叠（Part 9 只在 05 章 SageAttention 提 INT8/INT4 量化路线，属 attention 专属，无冲突 ✅）。
3. **与 Part 6/7 接口**：README 前置知识"Part 6 知道 q@k^T 是算子即可"、05 章练习 1 要求接回 Part7 脚本 05 的 MiniMindAttention（GQA 4 KV 头 repeat 成 8）——学生抽查 Part7 脚本 05 的实际头数配置与练习提示一致性。
4. **roadmap 侧**：§一缺口 2/3（节点 9 章节清单缺 05 章、验收线 105% 冲突）上报 roadmap 维护方；roadmap L72"Part 9 题 5 除外均可纯 CPU"与 assignment 实况一致性见 §三-6。
5. **导航链**：README 首尾链接 Part 8/Part 10 均实存（Part10_distributed 目录在位）；05 章末尾"上一章 04 / 返回 README"、04 章末尾只有"上一章 Part8"无"下一章"（因 05 章当时未写完？）——**待核：04 章末尾缺"下一章：05"链接，而 05 章实存**，登记为导航链小缺口（补入 §五统计外的 C1 关联项）。

## 七、审计执行注意事项

- 全程用 `.venv/bin/python`（系统 python3 无 torch）；复跑数字前确认 GPU 空闲度并声明共享/独占（G14，05 章已示范两次漂移的诚实写法）。
- `.cu` 重编译进 scratch 或用现有 `bin/`（勿 `make clean` 污染已验证产物；G16 先拷再跑）。
- 09 脚本跑一次 5 分钟预算，复跑只在对表冲突时做；autotune 选择随状态波动（教程已声明），审计以"验收线是否达标"为准不抠个位数。
- 数字对表基准：README L124-154 两块实测表 = 02 章 L126-135 = 05 章 §4.3 三方必须互洽，任何一处改动需三方联动（G15 精神）。
