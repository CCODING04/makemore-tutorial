# 审计状态表（单一事实源）

> 任何 agent 开工第一件事：读本表。T0 每单元收发即更新。

## 试点轮（已结束）

| 轮 | 章节 | 状态 | 备注 |
|---|---|---|---|
| R1 | P13-C01 手写去重 | closed | 用户通过（R1b 格式 + R1c LaTeX 修复）；格式规范 v1.2 转正 |
| R2 | P18-C01 朴素RAG到混合检索 | closed | 用户通过（进入全量即视为认可）；R3 取消 |
| R3 | — | cancelled | 用户决策：直接全量（2026-09-05） |

## 全量阶段（2026-09-05 启动，**已全部完成**）

状态机：`pending → t1_pre → students → fixing → t0_gate → closed`（异常：`blocked`）

| 批 | Part | 章节数 | 状态 | 镜像改动 | 台账 | 备注 |
|---|---|---|---|---|---|---|
| 1 | P01 bigrams | 4 | closed | 教程4+脚本3+作业2 | ledger_P01 (27项) | disputed 4 已仲裁（2入大纲清单） |
| 1 | P02 mlp | 4 | closed | 教程4+脚本7+作业3 | ledger_P02 (29项) | disputed 4 已仲裁 |
| 2 | P03 batchnorm | 4 | closed | 教程4+脚本6+作业3+图4 | ledger_P03 (32项) | disputed 3 已记录 |
| 2 | P04 backprop | 4 | closed | 教程4+脚本6+作业3+图3 | ledger_P04 (22项) | BN公式修复实证 4.6e-5→9.3e-10 |
| 2 | P05 wavenet | 4 | closed | 教程4+脚本7+作业2+图3 | ledger_P05 (28项) | 参数量170K→76579 实测重校 |
| 2 | P06 transformer | 5 | closed | 教程5+脚本3+作业1+图2 | ledger_P06 (19项) | 教程数字系CPU口径，已加声明 |
| 3 | P07 minimind | 7 | closed | 教程7+脚本7+作业1+图1 | ledger_P07 (31项) | 配置口径双轨制落地；脚本09修复 |
| 3 | P08 post_training | 10 | closed | 教程10+脚本8+作业2 | ledger_P08 (22项) | ckpt config 根因修复；pytest 四象限绿 |
| 4 | P09 cuda_kernels | 6 | closed | 教程6+作业test+图1 | ledger_P09 (16项) | CUDA全档实测；105%裁决归大纲清单 |
| 4 | P10 distributed | 5 | closed | 教程5+脚本6+作业2+图3 | ledger_P10 (18项) | FSDP CPU 假承诺修复；zero1/2 裁决归大纲 |
| 5 | P11 alignment_verl | 3 | closed | 教程3+脚本2+作业3+图3 | ledger_P11 (16项) | kl_penalty 错位修复；双 std 口径对照 |
| 5 | P12 finetune | 3 | closed | 教程3+脚本1+作业1+图3 | ledger_P12 (16项) | 双量化账本重写；分页优化器补齐 |
| 5 | P13 data_engineering | 3 | closed | 教程3+脚本1+README+图2 | ledger_P13 累计(40项) | fit 假正号修复；C01 已于 R1 关闭 |
| 5 | P14 inference_vllm | 3 | closed | 教程3+脚本1+作业1+图1 | ledger_P14 (17项) | 显存真值回填 1.85GiB；naive 基线 ±7% 复现 |
| 6 | P15 vision_language | 3 | closed | 教程3+脚本2+作业1+图3 | ledger_P15 (20项) | 题3数字/参数账29620 修复 |
| 6 | P16 image_video | 4 | closed | 教程4+脚本1+图1 | ledger_P16 (29项) | DDPM 推导补步；作业404系误报已仲裁 |
| 6 | P17 agentic_rl | 3 | closed | 教程3+脚本1+作业2+图1 | ledger_P17 (22项) | SHOW_NEGATIVE 开关；verl multi-turn 兑现 |
| 7 | P18 rag | 3 | closed | 教程3+脚本2+图1 | ledger_P18 累计(40项) | 1.9% 定谳无缺陷；C01 于 R2 关闭 |
| 7 | P19 agents | 3 | closed | 教程3+脚本3+作业1+图1 | ledger_P19 (21项) | 白名单绕过安全课补齐 |

## 每批收尾动作（T0）

- [ ] 批内横向一致性抽查（字符映射/术语/交叉链接/导航行）
- [ ] 全仓 `pytest assignments/ -q` 回归（对照改动前基线）
- [ ] T3 批级复核 agent 抽查（diff + 重跑 + 评分抽查）
- [ ] 通用问题/好写法回写 §10（G 规则接续编号）
- [ ] 本表更新 + git 无关确认（REPO 永不写入）
