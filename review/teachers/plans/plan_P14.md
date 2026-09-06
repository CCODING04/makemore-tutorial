# plan_P14 — Part 14 推理部署（vLLM）整体预审计划（T1 主教）

- 日期：2026-09-04 ｜ 主教：T1 ｜ 模式：只读预审（未改文件、未跑脚本）
- 范围：`courses/Part14_inference_vllm/`（tutorial 3 个 md 共 781 行 + scripts/ 1 个 122 行）+ `assignments/assignment_14/assignment.md` + `docs/course_roadmap_v3.md` 节点 14
- 环境事实：本机未装 vLLM（默认 python 无 vllm）；HF 缓存已有 `Qwen--Qwen2.5-0.5B-Instruct`（另有 0.5B base、7B-Instruct、Qwen3-Embedding），**无 GPTQ-Int4 权重** → 脚本 01 可离线复跑，02 章量化节需下载

---

## 1. C1 对应表（roadmap 节点 14 ↔ 仓库实体）

| roadmap 节点 14 条目 | 仓库实体 | 状态 |
|---|---|---|
| 教程 2 章：01 朴素基线与 TTFT/TPOT/吞吐测量 | `tutorial/01_naive_baseline.md`（385 行） | ✅ |
| 教程 2 章：02 vLLM 实战（离线→服务→benchmark→量化/投机→对比表） | `tutorial/02_vllm_serving.md`（396 行） | ✅ |
| 脚本 01_naive_generate_baseline.py | `scripts/01_naive_generate_baseline.py`（122 行） | ✅ |
| 作业 14："3 编码题（指标/KV/浪费率）+ 4090 实验题" | `assignments/assignment_14/`：**4 编码题**（指标/KV/浪费率/投机解码）+ 题 5 stretch 调度模拟器 + 实验题 + 5 思考题 | ⚠️ C1-1 roadmap 落后于实际（题 4、题 5 未列） |
| 论文 vLLM/PagedAttention（Kwon 2023）与节点 8 分页模拟互证 | README 关键论文 arXiv 2309.06180 + 02 章 §6 总账表互证；锚点已核：P8 06 章 L101 "41%→5%"、脚本 09 含分页+投机模拟 | ✅ |
| 延伸 llama.cpp（126.3k★）/SGLang | README 相关资源两项目在列 | ✅（star 数未标时点 → G13-3） |
| 学习目标：TTFT/TPOT/goodput 定义与测量 | 01 章 §1-2 + 01 Q1（空载 TTFT/goodput）+ 02 Q1 | ✅ |
| 学习目标：KV 容量账（1.07GB 锚点） | assignment 题 2（`kv_cache_gb(32,32,128,2048,1)≈1.07`，自算复核 1.0 GiB ✅） | ✅ |
| 学习目标：静态批浪费率 | assignment 题 3（0.675 自算复核 ✅） | ✅ |
| 学习目标：prefix caching 命中条件 | 02 Q2 + 实验题（共享 vs 随机 prompt） | ✅ |
| 学习目标："量化位宽与 **GGUF 命名规则**" | 教程 02 量化节仅 GPTQ/AWQ/FP8 服务，**全文无 GGUF 命名内容**（llama.cpp 仅一行链接） | ❌ C1-2 目标悬空 |
| 学习目标：跑通 vLLM 服务+benchmark、三行对比表 | 02 章 §1-3 + 填空表 | ✅（本机未装 → 概念档审计，见 §2.3） |
| 学习验证：pytest assignment_14 全绿 | test_serving_exercises.py 存在（本人未跑） | ✅（交学生档执行） |
| 学习验证：三行对比表一页纸 | 01 §3 填空表 + 02 §3 | ✅（"三行" vs 表实为 4 行，见 C1-4） |
| 学习验证：面试直通车 4 问 | assignment 🎯 节实为 **5 条** | ⚠️ C1-3 计数不一致 |

**C1 缺口汇总 = 4**：C1-1 roadmap"3 编码题"vs 实际 4+1；C1-2 GGUF 命名规则零覆盖；C1-3 "4 问" vs 实际 5 条；C1-4 "三行对比表"术语 vs 01 §3 表含第 4 行（KV 显存），教程/作业/roadmap 三处均沿用"三行"叫法未解释 KV 行地位。前 3 条建议改 roadmap/补教程一句即可，无结构性缺失。

**交叉链接抽验（全通过）**：`../../Part8_post_training/tutorial/06_inference_and_serving.md`、Part9 README/01/02（01 章含异步+synchronize、02 章 matmul 含内存墙，与 01 章前置引用精确对应）、Part7/10/13/15 README、`../../../assignments/assignment_14/`、`../scripts/01_naive_generate_baseline.py` 均存在。02 章"脚本 09：41%→5%"指向 `P8 scripts/09_quantize_and_serve.py` ✅（该脚本 docstring 写 60-80%→<4%，06 章正文写 41%→5%，两处口径不同但 06 章 L101 已解释"60-80% 来自真实长尾负载"，P14 转述与 06 章正文一致，不判错）。

## 2. 学生单元划分（3 学生并行）

- **S1 测量口径档**：01 章全文 + `scripts/01_naive_generate_baseline.py`（§2.1 执行复跑核对）
- **S2 vLLM 实战档（概念级）**：02 章 §1-3（安装两案/离线/服务/benchmark）+ README 环境节（§2.3 清单）
- **S3 进阶+作业档**：02 章 §4-6（量化/n-gram/总账）+ `assignment.md` 全部 + README 其余（G1/G13/C1 对 roadmap）+ 格式通检互查

## 3. 特有审计要点

### 3.1 TTFT/TPOT/吞吐测量口径（重点，可复现）
- 脚本口径设计良好：吞吐分母 = 64 次 32-token generate 计时段合计（`t_gen=sum(all_t)`），TTFT 单步探测与 tokenize 不入分母、其 token 不入分子（分子分母同口径）；wall 单列不打进吞吐。教程 01 §2 与脚本注释口径一致 ✅。
- TPOT 定义 = (总时间−TTFT)/(n−1)，实现用独立探测的 TTFT（近似已显式声明）✅；p50/p90 取索引（64 个取 [32]/[57]）合理。
- **学生执行项（S1，GPU 可跑、权重在缓存）**：复跑脚本核对教程引用的 6 个数字：158 tok/s / 1071 tok/s / TTFT 7.5ms / TPOT 6.2ms / 计时段 12.95s / wall 13.44s；并核 "吞吐×6.8"（1071/158=6.78 ✅ 自洽）。
- 审查点：正文"最佳实践：至少测 3 次取平均" vs 脚本单次运行、教程数字未声明平均口径 → 见 G14-1。
- 01 Q1"空载 TTFT"概念（串行基线 vs 高负载尾部）是好辨析，保留。

### 3.2 PagedAttention 原理讲解
- 02 章"数学推导"给了 75%→0% 的显存账（数值自算 ✅：ceil(512/16)=32、余 0），但**放在代码块里**（G2-4）；启动日志三行（# GPU blocks / max seq len / KV usage）落地到实操 ✅；与 P8 06 章 41%→5%、论文 <4% 三层数字关系讲清 ✅。
- 02 §6 总账表（手写→工业 5 行对照）是全 Part 亮点，数字锚点（41%→5%、α≈0.60、P7 KV dict）均已核存在。

### 3.3 vLLM 两案安装（本机未装 → 概念档审计为重）
- 教程诚实声明"本课开发环境未安装 vLLM，未做同机实测"，vLLM 行全部标"⚠️ 预期"——诚实标注文化好，保留为范本。
- 概念档审计清单（S2）：
  1. **方案 B 自毁性问题（P1 级）**：README 方案 B 称"复用课程 venv，pin vllm==0.6.6（恰好 pin torch==2.5.1+cu121）"——但课程 venv 是 torch 2.6.0+cu124，装 0.6.6 会把 torch **降级重装**，"复用"名不副实；代价栏应写明"会重写课程 venv 的 torch"。
  2. **02 章错误 1 解法过时且与 README 矛盾（P1 级）**：`--extra-index-url .../cu118`、"Python 3.8+/CUDA 11.8+" vs README 方案 A "CUDA 12.x wheel"——vLLM 官方 wheel 自带 CUDA 依赖、cu118 路线已废弃、Python 下限 3.9+。两处必统一。
  3. **benchmark_serving.py 路径悬空（P1 级）**：`python benchmarks/benchmark_serving.py` 不随 pip 安装提供，需 `git clone vllm-project/vllm` 后在其仓库根运行；教程未给获取方式，学生必卡在第一步。
  4. `speculative_config={"method":"ngram",...}` 字典写法是新版 API；方案 B pin 的 0.6.6 时代用 `speculative_model="[ngram]"`——若走方案 B 该代码不兼容（P2，概念档注明"以你安装版本的 docs 为准"即可）。
  5. `--enable-prefix-caching`"新版默认开"声明与 GPTQ/AWQ/FP8 直加载为版本敏感行为，概念档标注"以实测日志为准"。

### 3.4 量化（GPTQ/AWQ）声明
- 02 §4 用 `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4`：HF 缓存**无此权重**（约 0.4GB 下载），教程未提示；fp16 ~1GB → int4 ~0.4GB 体积账数量级正确 ✅。
- 02 Q3 精度数字（4bit 降 1-3%/FP8 <1%/小模型更明显）为经验值无出处 → G13-2；预期性能行（~4000+）已标预期 ✅。
- 4090(sm89) 支持 FP8 的隐含声明成立；"KV 不变"（权重量化不动 KV）口径正确 ✅。

### 3.5 n-gram 投机解码
- 机制解释（prompt lookup、免 draft、`prompt_lookup_num_tokens`）正确且与 P8 06 章 α≈0.60→2.31/2.47 锚点呼应 ✅；"24GB 卡友好"定位准确。API 版本兼容见 §3.3-4。

### 3.6 三行对比表数字口径
- 01 §3 表 4 行（吞吐/TTFT/TPOT/KV 显存）+ 02 §3 参考形态 + 02 性能表：naive 侧 158/7.5/6.2 全标"实测"且有脚本支撑（待 S1 复跑终核）；vLLM 侧标"预期未实测" ✅。缺口：① "三行"叫法 vs 4 行表（C1-4）；② 02 性能表**显存列** "~2GB/~4GB" 标 ✅ 实测，但脚本不打印任何显存数字——实测标注无来源（见 G14-2，建议降级为"约值/示意"或补 nvidia-smi 口径）。

## 4. scripts 档位
- GPU 档（默认）：64×32 约 20s，符合"<1 分钟"免 G8 档；CPU 降档自动（8 prompt × 8 token）✅，脚本头打印 device/torch/transformers 版本（G14 自证好）。但 **print 无 flush=True**，CPU 档 0.5B 64 步×8 请求可能逼近/超过 1 分钟（G8 边缘，P2 建议）；教程未给 CPU 档数字方向预期（G14-3）。
- 权重：`Qwen2.5-0.5B-Instruct` 已在本地 HF 缓存 → 脚本 01 离线可跑；教程"≈1GB 下载"提示对缓存命中的学生可注明可跳过。

## 5. 格式规范预检违反清单（合计 16 条 + 2 观察项）

| # | 条款 | 位置 | 问题 | 级别 |
|---|---|---|---|---|
| 1 | G1 | assignment Q5 参考答案 | ①②③ 三要点挤在同一段落，未逐条成行 | P2 |
| 2 | G2 | README "数学推导" 块 | 代码块承载 TTFT/TPOT/吞吐公式推导 | P1 |
| 3 | G2 | 01 章 "数学推导" 块 | 同上（且与 README 逐字重复，违 G15 单一事实源精神） | P1 |
| 4 | G2 | 01 章 §1 "可操作定义" 块 | 公式（E2E≈TTFT+TPOT×(n−1) 等）放代码块且含中文 | P1 |
| 5 | G2 | 02 章 "PagedAttention 显存优化" 块 | 75%→0% 计算推导放代码块 | P1 |
| 6 | G4 | 全 Part | 无 images/ 目录、零图；吞吐对比柱状图、静态 vs 连续批 makespan、历史演进时间线均"能画未画" | P2 |
| 7 | G10 | 02 章 §3 | benchmarks/benchmark_serving.py 获取方式缺失，路径悬空 | P1 |
| 8 | G10 | 02 章错误 1 | cu118/Python3.8 建议过时，且与 README 方案 A "CUDA 12.x" 自相矛盾 | P1 |
| 9 | G10 | 02 章 §5 | speculative_config 字典 API 与方案 B（0.6.6）不兼容风险 | P2 |
| 10 | G13 | README "为什么推理部署" | "推理成本占 70%+" 无出处 | P2 |
| 11 | G13 | 02 章 Q3 | 量化精度降幅（1-3%/<1%）无出处无双列 | P2 |
| 12 | G13 | README/roadmap | star 数（90.5k/32.9k/126.3k）未标"截至时点" | P2 |
| 13 | G13 | 02 章性能表 | vLLM 两行预期数已标"预期"（合规一半）但来源仅"官方论文/社区量级"，无具体锚点 | P2 |
| 14 | G14 | 01 章 | 基线数字未声明运行次数/是否平均；正文要求"3 次取平均"与单次脚本自相矛盾；seed 未声明（greedy 影响小，仍应写明） | P2 |
| 15 | G14 | 02 章性能表 | 显存列 "~2GB/~4GB" 标 ✅ 实测，但脚本不产出显存数字——实测标注无来源 | **P1** |
| 16 | G14 | 01 章/README | CPU 降档的数字方向预期未标注（"基线思想可读"未说明 CPU 档数字量级差多少） | P2 |
| 观 1 | G2 边缘 | 01 章"形状追踪" ASCII 框 | 框内含公式行，整改 G2 时顺带处理 | — |
| 观 2 | G8 边缘 | 脚本 01 | CPU 档可能 >1 分钟，建议 print 加 flush=True | — |

**统计：格式违规 16 条 = P1 7 条（#2 #3 #4 #5 #7 #8 #15）+ P2 9 条；P0 = 0。**

## 6. 总评与派工

总评：Part 14 是全课程诚实标注文化的范本（实测/预期口径分离、吞吐分母口径自证），结构与跨 Part 互证锚点全部扎实；主要整改面是"公式搬出代码块"（4 处 G2）、vLLM 安装章节三处 P1（方案 B 自毁性、cu118 矛盾、benchmark 路径悬空）与一条无来源的"实测"显存数字，另补 C1 的 GGUF 悬空目标与 roadmap 题数同步。无 P0，适合直接进入 B 阶段三学生并行。
