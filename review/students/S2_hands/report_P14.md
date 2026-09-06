# S2 学习报告·P14（推理部署实战 — vLLM）

> 审计视角：学生 S2（实操薄弱型）。全程只读教程/scripts/作业三件，先做后看。
> 环境：RTX 4090 ×2，torch 2.6.0+cu124，transformers 4.57.6，Python 3.12.12，**本机未装 vLLM**（且代理不可达、无法补装）。

## 总分：8.8 / 10

| 维度 | 得分 | 一句话 |
|---|---|---|
| README | 9 / 10 | 安装两案诚实且给了验证命令，CPU 降级表清晰 |
| 01 章 | 9.5 / 10 | 数字实测复现、口径自曝修正，质量很高 |
| 02 章 | 7 / 10 | 预期数字标注诚实，但 benchmark 路径悬空 + 修复命令不可信 |
| 脚本 01 | 9.5 / 10 | 一次跑通（换离线环境变量后）、输出即填空表 |
| 作业 | 9.5 / 10 | 5/5 PASS 全一次过，docstring 提示充分，SKIP 机制优雅 |

## 卡点清单（按学生实际遭遇顺序）

| # | 卡点 | 类型 | 影响 | 解决 |
|---|---|---|---|---|
| 1 | 首跑脚本即崩：环境代理 `192.168.0.105:7890` 不可达（No route to host），模型明明已在 HF 缓存，transformers 4.57.6 仍对 huggingface.co 发 HEAD/model_info 校验 | 环境 + 教程缺口 | 中（新手会卡在"模型都下载了为什么还联网"） | 1 次提示（`HF_HUB_OFFLINE=1`）后解决；教程/脚本均无离线环境提示 |
| 2 | `benchmark_serving.py` 引用悬空：02 章以 `python benchmarks/benchmark_serving.py` 相对路径出现 3 处，课程仓库无 benchmarks/ 目录，也未说明需先 clone vLLM 源码并 cd 到其根目录；新版 vLLM 实为 `vllm bench serve` 子命令 | 文档悬空引用 | 高（照抄必报 No such file，且 pip 装的 vLLM 根本没有该相对路径） | 未解决（教程侧待修） |
| 3 | 02 章"错误 1：vLLM 安装失败"的修复命令 `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118` 不可信：vLLM 从不在 PyTorch cu118 index 发布 wheel；"Python 需要 3.8+"也已过时（新版要求更高） | 文档错误 | 中（真正的安装失败者按此操作会二次失败） | 未解决（教程侧待修） |
| 4 | vLLM 全链条 env-blocked：本机无 vLLM（方案 A 需 ~5GB 下载，代理死装不了）→ 02 章 CLI 实操、作业实验题、三行对比表回填全部无法执行 | 环境 | 02 章实操与观测型实验题 0 执行 | env-blocked，仅静态评估 |
| 5 | 小项：脚本 docstring 称"GPU 约 20s"，实测 wall 14.0s + 模型加载，总量级略高于 20s，可接受 | 文档精度 | 低 | 无需修 |

## 分章评分

### README（9/10）
- ✅ 两案安装策略真实可拼凑：方案 A `uv venv .venv-vllm && pip install vllm` + `python -c "import vllm; print(vllm.__version__)"` 验证闭环完整，版本"以 pip 实际解析为准"的声明诚实。
- ✅ "你有什么 / 能做什么"降级表（CPU only → Colab；1×4090 → 全部内容）是学生真正需要的信息。
- ⚠️ 方案 B 预告了 flash-attn 解析冲突（issue #11283）但没给冲突发生时的具体绕法（如 `--enforce-eager` / `VLLM_ATTENTION_BACKEND`）。

### 01 章（9.5/10）
- ✅ **数字实测复现**（本机 4090）：

| 指标 | 教程声称 | S2 实测 | 偏差 |
|---|---|---|---|
| 吞吐（逐请求） | 158 tok/s（计时段 12.95s） | 152 tok/s（计时段 13.49s） | ~4% |
| TTFT p50/p90 | 7.5 / 7.6 ms | 7.9 / 8.0 ms | ~5% |
| TPOT p50/p90 | 6.2 / 6.3 ms | 6.5 / 6.5 ms | ~5% |
| 静态批 batch=8 | 1071 tok/s（×6.8） | 1032 tok/s（×6.8） | ~4% |

  方向与量级完全一致，教程"数字随机器状态略有波动属正常"的免责声明兑现。
- ✅ 声称"计时点前后共 7 处 `torch.cuda.synchronize()`"——逐行核对脚本恰好 7 处（L64/68/73/77/84/100/104），无夸大。
- ✅ 吞吐口径（分母不含 TTFT 探测）与脚本 `t_gen = sum(all_t)` 实现严格一致；"早期版本低估 5-10%"的自曝修正是最好的测量教学。
- ⚠️ 唯一缺口：无离线/代理受限环境的运行提示（卡点 1）。

### 02 章（7/10）
- ✅ vLLM 侧数字全部标"⚠️ 预期，未本机实测"，"没实测的一律标出来"的约定贯彻到位。
- ✅ 离线推理 5 行代码、`vllm serve` 命令、参数概念映射（`--max-model-len` = KV 池上限等）静态读来可拼凑。
- ❌ 卡点 2（benchmark 路径悬空，3 处）与卡点 3（错误 1 修复命令不可信、Python 3.8+ 过时）都出在本章——02 章是全 Part 最薄弱的一环。
- ⚠️ `speculative_config={"method":"ngram","prompt_lookup_num_tokens":4}` 参数名与新版 vLLM API 一致（env-blocked 无法实测验证）。

### 脚本 01（9.5/10）
- ✅ 换 `HF_HUB_OFFLINE=1` 后一次跑通（exit 0），教程号称为真：左 padding、7 处同步、CPU 降级分支（`PROMPTS[:8], max_new=8`）、输出即"待 vLLM 填空表"。
- ✅ 与教程代码块逐段对应，无"教程写了脚本没有"的漂移。

### 作业（9.5/10）
- ✅ 5/5 PASS 全一次过（含 stretch 题 5），详见下节。
- ✅ assignment.md 声明"实验题观测型、不占分"——vLLM 依赖有预案，S2 这种装不了 vLLM 的学生仍可拿满 100 分，设计周到。
- ⚠️ 思考题参考答案质量高（Q4 "PagedAttention 不改系数、改 seq 的计量方式"是亮点），但参考答案直接嵌在作业文件里，学生做到思考题时容易顺手看到（非硬伤）。

## 一致性核对表

| # | 检查项 | 结果 |
|---|---|---|
| 1 | README 章节导航 ↔ 实际 md 文件（01/02） | ✅ |
| 2 | README 脚本列 ↔ scripts/01_naive_generate_baseline.py 存在；02 标"CLI 实操"无脚本（诚实） | ✅ |
| 3 | 教程 01 数字 ↔ 脚本本机实测（152/7.9/6.5/1032） | ✅ 偏差 2-5%，方向一致 |
| 4 | "7 处 synchronize" 声称 ↔ 脚本实际 7 处 | ✅ |
| 5 | 吞吐口径说明 ↔ 脚本实现（探测不计分母） | ✅ |
| 6 | 教程内相对链接（../scripts/、../../../assignments/assignment_14/） | ✅ 均可达 |
| 7 | 作业验收标准数值 ↔ 测试断言（0.675 / 66 / 2.3056 / 5.0 等） | ✅ 全对上 |
| 8 | 测试双模式（python / pytest）可独立运行 | ✅ pytest 9.1.1 实测通过 |
| 9 | 题 5 未实现 → 优雅 SKIP 机制存在；实现后 PASS | ✅ |
| 10 | `benchmark_serving.py` 路径（3 处） | ❌ 悬空，仓库无此文件且未说明来源 |
| 11 | 02 章错误 1 修复命令（cu118 extra-index-url）与 Python "3.8+" | ❌ 不可信/过时 |
| 12 | vLLM 未本机实测数字的"⚠️ 预期"标注约定 | ✅ 诚实 |
| 13 | 环境版本声明（torch 2.6.0+cu124 / transformers 4.57.6 / 4090） ↔ S2 本机 | ✅ 逐项一致 |

## 作业元数据 + pytest 输出

- 工作目录：`/home/admin02/Code/WorkSpace/makemore-tutorial-review/students/S2_hands/work/assignment_14/`（仅编辑 `serving_exercises.py`）
- 运行命令：`cd <工作目录> && /home/admin02/Code/WorkSpace/makemore-tutorial/.venv/bin/python -m pytest test_serving_exercises.py -v`
- vLLM 依赖题：作业题 1-5 无 vLLM 依赖；**实验题（观测型，不占分）env-blocked**——本机未装 vLLM 且无法联网安装。

| 题 | 内容 | 一次过? | 提示次数 | 耗时 | 结果 |
|---|---|---|---|---|---|
| 1 | serving 指标（E2E / 吞吐） | ✅ | 0 | ~1 min | PASS |
| 2 | KV 容量账（1.07GB / GQA / max_batch） | ✅ | 0 | ~1 min | PASS |
| 3 | 静态批浪费率（67.5%） | ✅ | 0 | ~1 min | PASS |
| 4 | 投机解码（α 极限 / 加速比） | ✅ | 0 | ~1 min | PASS |
| 5 🌟 | 连续批处理调度模拟器 | ✅ | 0 | ~2 min（落笔前在纸上推演 slot_steps 口径约 2 分钟） | PASS |

```
============================= test session starts ==============================
platform linux -- Python 3.12.12, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python
cachedir: .pytest_cache
rootdir: .../students/S2_hands/work/assignment_14
plugins: anyio-4.14.2
collected 5 items

test_serving_exercises.py::test_ex1_metrics PASSED                       [ 20%]
test_serving_exercises.py::test_ex2_kv PASSED                            [ 40%]
test_serving_exercises.py::test_ex3_waste PASSED                         [ 60%]
test_serving_exercises.py::test_ex4_spec_decode PASSED                   [ 80%]
test_serving_exercises.py::test_ex5_stretch_sim PASSED                   [100%]

============================== 5 passed in 0.02s ===============================
```

## 只改 3 件事

1. **修 `benchmark_serving.py` 悬空引用（02 章 L107/349/381）**：注明"该脚本在 vLLM 源码仓库内，需 `git clone https://github.com/vllm-project/vllm` 后在其根目录执行"，并补充新版等价命令 `vllm bench serve ...`（pip 安装即自带，学生无需 clone）。
2. **修 02 章"错误 1"的修复方案**：删掉 `--extra-index-url https://download.pytorch.org/whl/cu118`（vLLM 不在该 index 发布）；Python 要求从"3.8+"改为"3.9+/新版 3.10+"，并指向 vLLM 官方安装页作为唯一权威来源。
3. **教程 01 章/脚本 docstring 加一行离线环境提示**：模型已在 HF 缓存时，代理不可达会导致 transformers 联网校验崩溃，运行前 `export HF_HUB_OFFLINE=1` 即可——本次审计的真实第一卡点。

## 最喜欢 3 处

1. **01 章的"吞吐口径修正"自曝段**（L126-129）：公开承认早期版本把 TTFT 探测时间计入分母导致吞吐系统性低估 5-10%，并解释错在哪、怎么改——测量教学里最难得的"把错误摆上台面"。
2. **脚本输出即"待 vLLM 填空的对比表"**：教程 → 脚本 → 02 章实操 → 作业实验题形成闭环，S2 跑完脚本时自然知道下一步要填什么，学习目标零歧义。
3. **"没实测的一律标出来"的诚实约定**：02 章 vLLM 两行数字全部挂"⚠️ 预期，未本机实测"，与 01 章"(预期)"标注同一口径——这在教程类文档里罕见地克制且可信。
