# Part 8 整改报告（T2 教师）

- 日期：2026-09-05；REPO 只读未动；全部修改写入 REVIEW 镜像
- 输入：plan_P08.md + S1/S2/S3 三报告 + T0 必修清单 12 项
- 台账：`ledger/ledger_P08.md`（fixed 22 条：P0×6、P1×10、P2×6；disputed 4 条）

## 一、镜像改动清单

### scripts（7 个改动 + 1 重命名）
| 文件 | 改动 |
|---|---|
| `09_reasoning_models.py → 13_reasoning_models.py` | 重命名（双 09 冲突）；docstring 自引用同步；"非 NM"→"非 NN" |
| `08_eval_and_chat.py` | 新增 `infer_config_from_state_dict()`：按 ckpt 权重实际形状（tok_emb/pos_emb/blocks/heads）反推配置加载，config 谎报时打 ⚠️；失败时打印同档重跑指引；vocab 与当前数据不符时跳过 |
| `06_ppo_training.py` | 保存 ckpt 的 config 改为按被保存模型实际形状生成（根因修复）；SMALL=1 开关；训练 print flush；docstring 档位说明 |
| `07_grpo_training.py` | 同 06（保存侧 + SMALL + flush） |
| `02_pretrain.py` | SMALL=1 开关；训练 print flush；docstring 档位说明 |
| `03_sft.py` | "masked 通常 > unmasked"断言补适用条件（欠训练+合成数据可相反） |
| `09_quantize_and_serve.py` | "本课 2M 小模型"→"0.4M"；"本课 40M 模型"→"GPU 档（~89M）" |
| `10_lora_from_scratch.py` | 未改（验证基线） |

### tutorial（10 个 md 全部有改动）
README（参数量口径+导航 13）、01（参数量表重算+untied 注）、02（masked 断言）、03（作业指引、DPO 5 式 LaTeX、β 边界段、KTO 差异声明）、04（作业指引、clip 双联图+数字例、GAE/TD/GRPO/k3 LaTeX、k3 无偏推导+数字例、GRPO std N−1 注）、05（作业段+新收尾+运行顺序档位声明+SMALL 用法）、06（病句+模拟/论文口径）、07（166 处引号统一、roadmap_v2→v3、下一章链接）、08（LoRA 补一行、下一步改指 09 章、"本章"）、09（残留句删、13 重命名同步、实测口径、诚实声明、上一章链接）。

### assignments/assignment_8
- `test_post_training_exercises.py`：`_skip()` pytest 下改抛 `pytest.skip`；`_skip_exceptions()` 显式列出两种跳过异常（pytest 的 Skipped 继承 BaseException，except Exception 捕不到）；main except 链适配
- `assignment.md`："本课 2M 模型"→"~0.4M 模型（脚本 09 的量化实验模型）"

### 其它产出
- `outline_review/outline_suggestions.md` 追加 5 条 roadmap 冲突项（投机解码数字、8 章/脚本 01-10 清单、ignore_index 口径、pytest 全绿已修复待复核、PagedAttention 论文值口径）——未改 docs

## 二、验证结果（全部通过）

1. **脚本实跑**（镜像脚本，scratch=t2_P8/run_scripts，CUDA_VISIBLE_DEVICES="" 强制 CPU）：
   - `10_lora_from_scratch.py` rc=0（秒级）
   - `13_reasoning_models.py` rc=0（SFT loss 0.68；n=1/4/8=58/70/80%；"非 NN" 修复在输出中生效）
   - `09_quantize_and_serve.py` rc=0（int8 Δ−0.37 等四实验；α≈0.56→理论 2.16）
   - `08_eval_and_chat.py` rc=0：**修复前** GRPO/PPO ckpt（config 谎报 64 档、权重 512 档）加载必崩；**修复后** 5 个 ckpt 全部 [OK] 按权重形状加载
   - `02_pretrain.py` rc=0（SMALL 档行为同原 CPU 档，flush 生效）；`07_grpo_training.py SMALL=1` rc=0，新保存 ckpt 的 config 与权重逐字段一致（n_head 表达式对真实 sd 验证=8）
   - 全部改动脚本 py_compile 通过
2. **作业测试**（参考答案 + 修复后 test，四象限）：
   - pytest + 参考答案 = **8 passed**；pytest + 未实现 = **8 skipped（0 failed）**
   - 独立运行 + 参考答案 = 8/8 通过；独立 + 未实现 = 8 跳过 0 失败
3. **check_latex.py**：10 个 md + assignment.md 全部"✅ 未发现问题"（含新增 13 处 `$$` 公式）。

## 三、根因新发现（三报告未定位到的深层原因）

08 脚本崩溃不是"GPU ckpt vs CPU config 的预期不一致"，而是 **06/07 保存侧 bug**：保存时 config 写脚本顶部全局变量，而 policy 实际从 ckpt 加载（档位可不同），产出 config 与权重不符的 ckpt（实测 `ckpt_grpo.pt`：config n_embed=64/2 blocks/4 heads，权重 (65,512)/12 blocks/8 heads）。本次双向修复：08 侧形状推断兜底（能加载任何档位），06/07 保存侧根治（config 恒与权重一致）。

## 四、通用问题候选（≤3，供全课程规范）

1. **章末作业指引与作业题序缺同步校验**：题序调整后教程末尾未回改（03/04 章错位、05 章缺指引）——建议课程级"作业题号↔章末指引"一致性脚本化检查。
2. **ckpt 档位无自描述校验**：保存侧 config 取全局变量而非被保存对象实际形状，加载侧无校验——建议全课程脚本统一"保存时从 state_dict 推断 config、加载前 shape 校验"模式（08 已示范）。
3. **参数量/实测数字多处手写无单一出处**：同一事实四处三个数（2M/0.4M/40M/89M）——建议每 Part 的 README 规模表为唯一口径源，正文引用不重复写数字。

## 五、好写法候选（≤3）

1. **06 章"一个原理打全部：decode 是 memory-bound"**（三学生共同点赞）——把量化/批处理/投机解码/KV 统一成一句锚点的写法，可作其它工程章节模板。
2. **作业 docstring"公式+提示+验证标准"三段式 + SKIP 机制**（S2 近乎满分体验）——修复后 pytest/独立双路径全绿，可作全课程作业模板。
3. **04 章 GRPO "3+5=" 全中间数手算例 + 本次补的 k3 数字例**——"敢把中间数全写出来"的公式讲解范式（S1：跟着手算一遍就懂），建议推广到 DPO/GAE 等其余推导节。
