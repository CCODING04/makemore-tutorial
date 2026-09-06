# manifest：逐文件改动清单

| 镜像路径（REVIEW 下） | 改动类型 | 理由（台账编号） | 验证 | 合回状态 |
|---|---|---|---|---|
| courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.md | 优化（409→509 行） | P13-C01 台账 F01-F09/F14-F19 | 论文核实+输出回归 | R1 待评审 |
| courses/Part13_data_engineering/scripts/01_minhash_dedup.py | 修复（139→152 行，输出无回归） | F10-F13/F19/F20 | diff 一致+assert 验证 | R1 待评审 |
| assignments/assignment_13/assignment.md | 口径对齐 | F21 | 与章对照 | R1 待评审 |
| courses/Part13_data_engineering/images/lsh_s_curve.png | 新增（matplotlib 生成） | F24 格式规范 G4 | scratch/verify_R1/make_scurve.py 可复现 | R1b 待评审 |
| （教程/作业三文件） | 格式重写（G1-G5） | F22-F25 | 图表数值互核 | R1b 待评审 |
| check_latex.py（REVIEW 根目录） | 新增工具 | LaTeX 结构检查（跨行$$/公式内中文/列表内display） | 教程+作业扫描通过 | R1c 待评审 |
| courses/Part18_rag/tutorial/01_naive_to_hybrid.md | 优化（589→约640行，G1-G5 应用） | P18-C01 台账 F01-F13 | 三方数字复现+check_latex 0问题 | R2 待评审 |
| courses/Part18_rag/images/recall_ablation.png | 新增（matplotlib） | F08 | scratch/teacher_R2/make_recall_fig.py 可复现 | R2 待评审 |
| assignments/assignment_18/assignment.md | 格式修订 | F14 | check_latex 0 问题 | R2 待评审 |
| courses/Part6_transformer/tutorial/README.md | 优化（路线图 Part7 出口+演进表对齐实跑+ppl 注+挂图） | ledger_P06 C00-T1(C1-1)/C00-T1(G4)/C02-S3-01 | check_latex 0 问题；链接校验 | 批2 待评审 |
| courses/Part6_transformer/tutorial/01_data_and_tokenizer.md | 优化（device 定义 G10+实跑口径声明+日志/样例对齐+题3 挂章+BPE 出口） | C01-T(G10)/C02-S1-04/C01-T1(C1-3)/C00-T1(C1-1)/C01-T1(A1①) | CPU 实跑逐位对照（cpu_01/02.log） | 批2 待评审 |
| courses/Part6_transformer/tutorial/02_attention_from_scratch.md | 优化（√d_k LaTeX 推导+scaled_dot_product_affinity 段+各头观察实验+挂图+NaN/整除/tril 备忘） | C02-S1-01/C02-T1(C1-4)/C02-S3-03/C02-S1-06 等 | 修改版脚本 04 实跑逐位引用（04_mirror_cpu.log） | 批2 待评审 |
| courses/Part6_transformer/tutorial/03_transformer_block.md | 优化（残差措辞+LN 有偏/无偏+pre-norm 稳定性+ppl 小节+07 双档位 SMALL=1+日志对齐） | C03-S1-02/C03-S1-03/C03-S3-02/C02-S3-01/C02-T-01 | exp_small/exp_prenorm 实测；07 SMALL=1 与原版逐位一致 | 批2 待评审 |
| courses/Part6_transformer/tutorial/04_beyond_transformer.md | 优化（5 处不同清单+Part7 组件对照表+Part8 出口） | C04-S3-05/C04-T1(C1-5)/C00-T1(C1-1) | 链接校验；条目可回溯正文 | 批2 待评审 |
| courses/Part6_transformer/scripts/04_self_attention.py | 增强（√d_k 数值验证打印，独立 Generator 零 RNG 扰动） | C02-S1-01 | 训练日志 diff 逐位一致 | 批2 待评审 |
| courses/Part6_transformer/scripts/05_multihead_feedforward.py | 注释修正（残差梯度措辞，代码零改动） | C03-S1-02 | py_compile 通过 | 批2 待评审 |
| courses/Part6_transformer/scripts/07_scaleup_generate.py | 修复（SMALL=1 强制缩小型+print flush+GPU 提示行；默认档行为不变） | C02-T-01 | SMALL=1 与原版逐位一致；GPU 默认档探针 | 批2 待评审 |
| courses/Part6_transformer/images/loss_evolution.png | 新增（matplotlib，README 表实测数据） | C00-T1(G4) | make_plots.py 可复现 | 批2 待评审 |
| courses/Part6_transformer/images/softmax_scaling.png | 新增（matplotlib，脚本 04 实测概率） | C00-T1(G4) | make_plots.py 可复现 | 批2 待评审 |
| assignments/assignment_6/assignment.md | 措辞修正（题5 残差"均分"→"原样复制"） | C03-S1-02 | check_latex 0 问题；pytest 7 passed | 批2 待评审 |
| courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md | 优化（推导补步+陷阱合并重写+性能表分口径+记号约定+挂图） | ledger_P16 P1-1/P1-2/P1-3/P2-1..P2-9 | 脚本基线逐位复现+check_latex 0 问题 | 批 待评审 |
| courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md | 优化（8×/48× 双口径+idx 去黑盒+版本钉+下载命令） | P2-3/P2-4/P2-16/P1-4 | check_latex 0 问题；练习参考数值学生实测复核 | 批 待评审 |
| courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md | 优化（骨干三家的因果分层口径+Q2 重写+练习措辞+引述口径） | P1-5/P1-6/P2-11/P2-13/P2-14/P2-15 | check_latex 0 问题 | 批 待评审 |
| courses/Part16_image_video_generation/tutorial/README.md | 优化（权重下载三步+学习目标措辞+star 时点+版本钉） | P1-4/P2-10/P2-16/P2-17/P0-1 | 链接 ls 核对 | 批 待评审 |
| courses/Part16_image_video_generation/scripts/02_alignment_mechanisms.py | 增强（scale=0 解耦性 assert+尾注 Latte 口径限定） | P2-12/P1-5 | scratch 实跑断言通过、数值行零回归 | 批 待评审 |
| courses/Part16_image_video_generation/images/ddpm_forward_reverse.png | 新增（matplotlib 真实数据过程图） | P2-18 格式规范 G4 | scratch/T2_P16/make_fig.py 可复现 | 批 待评审 |

## 全量镜像文件清单（战役结束自动生成，以此为准）

以下为 REVIEW 相对 REPO 存在差异/新增的全部文件（不含 scratch/students/teachers 报告与 ledger/plan 等审计工件，仅课程本体）：

```
只在 -review/courses/Part10_distributed 中存在：images
文件 /courses/Part10_distributed/scripts/01_distributed_basics.py 和 -review/courses/Part10_distributed/scripts/01_distributed_basics.py 不同
文件 /courses/Part10_distributed/scripts/02_ddp_gpt.py 和 -review/courses/Part10_distributed/scripts/02_ddp_gpt.py 不同
文件 /courses/Part10_distributed/scripts/03_zero_memory.py 和 -review/courses/Part10_distributed/scripts/03_zero_memory.py 不同
文件 /courses/Part10_distributed/scripts/04_fsdp_gpt.py 和 -review/courses/Part10_distributed/scripts/04_fsdp_gpt.py 不同
文件 /courses/Part10_distributed/scripts/06_pipeline_parallel.py 和 -review/courses/Part10_distributed/scripts/06_pipeline_parallel.py 不同
只在 /courses/Part10_distributed/scripts 中存在：__pycache__
文件 /courses/Part10_distributed/tutorial/01_why_and_collectives.md 和 -review/courses/Part10_distributed/tutorial/01_why_and_collectives.md 不同
文件 /courses/Part10_distributed/tutorial/02_ddp.md 和 -review/courses/Part10_distributed/tutorial/02_ddp.md 不同
文件 /courses/Part10_distributed/tutorial/03_memory_zero_fsdp.md 和 -review/courses/Part10_distributed/tutorial/03_memory_zero_fsdp.md 不同
文件 /courses/Part10_distributed/tutorial/04_tp_pp_and_beyond.md 和 -review/courses/Part10_distributed/tutorial/04_tp_pp_and_beyond.md 不同
文件 /courses/Part10_distributed/tutorial/README.md 和 -review/courses/Part10_distributed/tutorial/README.md 不同
只在 -review/courses/Part11_alignment_verl 中存在：images
文件 /courses/Part11_alignment_verl/scripts/01_reward_and_bridge.py 和 -review/courses/Part11_alignment_verl/scripts/01_reward_and_bridge.py 不同
文件 /courses/Part11_alignment_verl/scripts/02_grpo_toy_train.py 和 -review/courses/Part11_alignment_verl/scripts/02_grpo_toy_train.py 不同
文件 /courses/Part11_alignment_verl/tutorial/01_handwritten_to_verl.md 和 -review/courses/Part11_alignment_verl/tutorial/01_handwritten_to_verl.md 不同
文件 /courses/Part11_alignment_verl/tutorial/02_verl_quickstart.md 和 -review/courses/Part11_alignment_verl/tutorial/02_verl_quickstart.md 不同
文件 /courses/Part11_alignment_verl/tutorial/README.md 和 -review/courses/Part11_alignment_verl/tutorial/README.md 不同
只在 -review/courses/Part12_finetune_llamafactory 中存在：images
文件 /courses/Part12_finetune_llamafactory/scripts/01_handwritten_sft_lora.py 和 -review/courses/Part12_finetune_llamafactory/scripts/01_handwritten_sft_lora.py 不同
只在 /courses/Part12_finetune_llamafactory/scripts 中存在：__pycache__
文件 /courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md 和 -review/courses/Part12_finetune_llamafactory/tutorial/01_handwritten_sft_lora.md 不同
文件 /courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md 和 -review/courses/Part12_finetune_llamafactory/tutorial/02_llamafactory_workflow.md 不同
文件 /courses/Part12_finetune_llamafactory/tutorial/README.md 和 -review/courses/Part12_finetune_llamafactory/tutorial/README.md 不同
只在 -review/courses/Part13_data_engineering 中存在：images
文件 /courses/Part13_data_engineering/scripts/00_scaling_laws.py 和 -review/courses/Part13_data_engineering/scripts/00_scaling_laws.py 不同
文件 /courses/Part13_data_engineering/scripts/01_minhash_dedup.py 和 -review/courses/Part13_data_engineering/scripts/01_minhash_dedup.py 不同
只在 /courses/Part13_data_engineering/scripts 中存在：output_scaling_epoch.png
只在 /courses/Part13_data_engineering/scripts 中存在：output_scaling_fit.png
只在 /courses/Part13_data_engineering/scripts/__pycache__ 中存在：00_scaling_laws.cpython-312.pyc
文件 /courses/Part13_data_engineering/scripts/__pycache__/01_minhash_dedup.cpython-312.pyc 和 -review/courses/Part13_data_engineering/scripts/__pycache__/01_minhash_dedup.cpython-312.pyc 不同
文件 /courses/Part13_data_engineering/tutorial/00_scaling_laws.md 和 -review/courses/Part13_data_engineering/tutorial/00_scaling_laws.md 不同
文件 /courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.md 和 -review/courses/Part13_data_engineering/tutorial/01_dedup_from_scratch.md 不同
文件 /courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md 和 -review/courses/Part13_data_engineering/tutorial/02_data_juicer_pipeline.md 不同
文件 /courses/Part13_data_engineering/tutorial/README.md 和 -review/courses/Part13_data_engineering/tutorial/README.md 不同
只在 -review/courses/Part14_inference_vllm 中存在：images
文件 /courses/Part14_inference_vllm/scripts/01_naive_generate_baseline.py 和 -review/courses/Part14_inference_vllm/scripts/01_naive_generate_baseline.py 不同
只在 /courses/Part14_inference_vllm/scripts 中存在：__pycache__
文件 /courses/Part14_inference_vllm/tutorial/01_naive_baseline.md 和 -review/courses/Part14_inference_vllm/tutorial/01_naive_baseline.md 不同
文件 /courses/Part14_inference_vllm/tutorial/02_vllm_serving.md 和 -review/courses/Part14_inference_vllm/tutorial/02_vllm_serving.md 不同
文件 /courses/Part14_inference_vllm/tutorial/README.md 和 -review/courses/Part14_inference_vllm/tutorial/README.md 不同
只在 -review/courses/Part15_vision_language 中存在：images
文件 /courses/Part15_vision_language/scripts/01_vit_projector_pipeline.py 和 -review/courses/Part15_vision_language/scripts/01_vit_projector_pipeline.py 不同
文件 /courses/Part15_vision_language/scripts/02_clip_siglip_alignment.py 和 -review/courses/Part15_vision_language/scripts/02_clip_siglip_alignment.py 不同
只在 /courses/Part15_vision_language/scripts 中存在：__pycache__
文件 /courses/Part15_vision_language/tutorial/01_handwritten_projection_vlm.md 和 -review/courses/Part15_vision_language/tutorial/01_handwritten_projection_vlm.md 不同
文件 /courses/Part15_vision_language/tutorial/02_alignment_losses_and_schemes.md 和 -review/courses/Part15_vision_language/tutorial/02_alignment_losses_and_schemes.md 不同
文件 /courses/Part15_vision_language/tutorial/README.md 和 -review/courses/Part15_vision_language/tutorial/README.md 不同
只在 -review/courses/Part16_image_video_generation 中存在：images
文件 /courses/Part16_image_video_generation/scripts/02_alignment_mechanisms.py 和 -review/courses/Part16_image_video_generation/scripts/02_alignment_mechanisms.py 不同
文件 /courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md 和 -review/courses/Part16_image_video_generation/tutorial/01_ddpm_from_scratch.md 不同
文件 /courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md 和 -review/courses/Part16_image_video_generation/tutorial/02_t2i_i2i_pipelines.md 不同
文件 /courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md 和 -review/courses/Part16_image_video_generation/tutorial/03_alignment_and_video.md 不同
文件 /courses/Part16_image_video_generation/tutorial/README.md 和 -review/courses/Part16_image_video_generation/tutorial/README.md 不同
只在 -review/courses/Part17_agentic_rl 中存在：images
文件 /courses/Part17_agentic_rl/scripts/01_toy_agent_grpo.py 和 -review/courses/Part17_agentic_rl/scripts/01_toy_agent_grpo.py 不同
文件 /courses/Part17_agentic_rl/scripts/__pycache__/01_toy_agent_grpo.cpython-312.pyc 和 -review/courses/Part17_agentic_rl/scripts/__pycache__/01_toy_agent_grpo.cpython-312.pyc 不同
文件 /courses/Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md 和 -review/courses/Part17_agentic_rl/tutorial/01_from_single_turn_to_agent.md 不同
文件 /courses/Part17_agentic_rl/tutorial/02_rewards_and_frameworks.md 和 -review/courses/Part17_agentic_rl/tutorial/02_rewards_and_frameworks.md 不同
文件 /courses/Part17_agentic_rl/tutorial/README.md 和 -review/courses/Part17_agentic_rl/tutorial/README.md 不同
只在 -review/courses/Part18_rag 中存在：images
只在 /courses/Part18_rag/scripts 中存在：01_minimal_rag.py
文件 /courses/Part18_rag/scripts/02_contextual_retrieval.py 和 -review/courses/Part18_rag/scripts/02_contextual_retrieval.py 不同
文件 /courses/Part18_rag/scripts/03_rag_eval.py 和 -review/courses/Part18_rag/scripts/03_rag_eval.py 不同
只在 /courses/Part18_rag/scripts/__pycache__ 中存在：01_minimal_rag.cpython-312.pyc
文件 /courses/Part18_rag/scripts/__pycache__/02_contextual_retrieval.cpython-312.pyc 和 -review/courses/Part18_rag/scripts/__pycache__/02_contextual_retrieval.cpython-312.pyc 不同
文件 /courses/Part18_rag/scripts/__pycache__/03_rag_eval.cpython-312.pyc 和 -review/courses/Part18_rag/scripts/__pycache__/03_rag_eval.cpython-312.pyc 不同
文件 /courses/Part18_rag/tutorial/01_naive_to_hybrid.md 和 -review/courses/Part18_rag/tutorial/01_naive_to_hybrid.md 不同
文件 /courses/Part18_rag/tutorial/02_advanced_rag.md 和 -review/courses/Part18_rag/tutorial/02_advanced_rag.md 不同
文件 /courses/Part18_rag/tutorial/README.md 和 -review/courses/Part18_rag/tutorial/README.md 不同
只在 -review/courses/Part19_agents 中存在：images
文件 /courses/Part19_agents/scripts/01_agent_loop.py 和 -review/courses/Part19_agents/scripts/01_agent_loop.py 不同
文件 /courses/Part19_agents/scripts/02_mini_mcp.py 和 -review/courses/Part19_agents/scripts/02_mini_mcp.py 不同
文件 /courses/Part19_agents/scripts/03_tau_mini.py 和 -review/courses/Part19_agents/scripts/03_tau_mini.py 不同
只在 /courses/Part19_agents/scripts 中存在：__pycache__
文件 /courses/Part19_agents/tutorial/01_agent_loop.md 和 -review/courses/Part19_agents/tutorial/01_agent_loop.md 不同
文件 /courses/Part19_agents/tutorial/02_protocols_and_frameworks.md 和 -review/courses/Part19_agents/tutorial/02_protocols_and_frameworks.md 不同
文件 /courses/Part19_agents/tutorial/README.md 和 -review/courses/Part19_agents/tutorial/README.md 不同
只在 /courses/Part1_bigrams 中存在：images
只在 /courses/Part1_bigrams 中存在：makemore_part1_bigrams.ipynb
只在 /courses/Part1_bigrams/scripts 中存在：01_explore_data.py
文件 /courses/Part1_bigrams/scripts/02_bigram_counting.py 和 -review/courses/Part1_bigrams/scripts/02_bigram_counting.py 不同
只在 /courses/Part1_bigrams/scripts 中存在：03_visualize_matrix.py
只在 /courses/Part1_bigrams/scripts 中存在：04_probability_sampling.py
只在 /courses/Part1_bigrams/scripts 中存在：05_nll_loss.py
文件 /courses/Part1_bigrams/scripts/06_neural_network.py 和 -review/courses/Part1_bigrams/scripts/06_neural_network.py 不同
文件 /courses/Part1_bigrams/scripts/07_gradient_descent.py 和 -review/courses/Part1_bigrams/scripts/07_gradient_descent.py 不同
只在 /courses/Part1_bigrams/scripts 中存在：bigram_matrix.png
文件 /courses/Part1_bigrams/tutorial/01_introduction.md 和 -review/courses/Part1_bigrams/tutorial/01_introduction.md 不同
文件 /courses/Part1_bigrams/tutorial/02_bigram_model.md 和 -review/courses/Part1_bigrams/tutorial/02_bigram_model.md 不同
文件 /courses/Part1_bigrams/tutorial/03_neural_network.md 和 -review/courses/Part1_bigrams/tutorial/03_neural_network.md 不同
文件 /courses/Part1_bigrams/tutorial/README.md 和 -review/courses/Part1_bigrams/tutorial/README.md 不同
只在 /courses/Part2_mlp 中存在：makemore_part2_mlp.ipynb
文件 /courses/Part2_mlp/scripts/01_explore_data.py 和 -review/courses/Part2_mlp/scripts/01_explore_data.py 不同
文件 /courses/Part2_mlp/scripts/02_dataset_with_context.py 和 -review/courses/Part2_mlp/scripts/02_dataset_with_context.py 不同
文件 /courses/Part2_mlp/scripts/03_embedding.py 和 -review/courses/Part2_mlp/scripts/03_embedding.py 不同
文件 /courses/Part2_mlp/scripts/04_mlp_forward.py 和 -review/courses/Part2_mlp/scripts/04_mlp_forward.py 不同
文件 /courses/Part2_mlp/scripts/05_minibatch_training.py 和 -review/courses/Part2_mlp/scripts/05_minibatch_training.py 不同
文件 /courses/Part2_mlp/scripts/06_visualize_embedding.py 和 -review/courses/Part2_mlp/scripts/06_visualize_embedding.py 不同
文件 /courses/Part2_mlp/scripts/07_sampling.py 和 -review/courses/Part2_mlp/scripts/07_sampling.py 不同
文件 /courses/Part2_mlp/tutorial/01_introduction.md 和 -review/courses/Part2_mlp/tutorial/01_introduction.md 不同
文件 /courses/Part2_mlp/tutorial/02_mlp_architecture.md 和 -review/courses/Part2_mlp/tutorial/02_mlp_architecture.md 不同
文件 /courses/Part2_mlp/tutorial/03_training_and_eval.md 和 -review/courses/Part2_mlp/tutorial/03_training_and_eval.md 不同
文件 /courses/Part2_mlp/tutorial/README.md 和 -review/courses/Part2_mlp/tutorial/README.md 不同
只在 -review/courses/Part3_batchnorm/images 中存在：01_tanh_saturation_single.png
只在 -review/courses/Part3_batchnorm/images 中存在：02_bn_running_stats.png
只在 -review/courses/Part3_batchnorm/images 中存在：02_bn_standardization.png
只在 -review/courses/Part3_batchnorm/images 中存在：03_grad_data_ratio.png
文件 /courses/Part3_batchnorm/scripts/01_diagnose_initial_loss.py 和 -review/courses/Part3_batchnorm/scripts/01_diagnose_initial_loss.py 不同
文件 /courses/Part3_batchnorm/scripts/02_diagnose_tanh_saturation.py 和 -review/courses/Part3_batchnorm/scripts/02_diagnose_tanh_saturation.py 不同
文件 /courses/Part3_batchnorm/scripts/03_kaiming_init.py 和 -review/courses/Part3_batchnorm/scripts/03_kaiming_init.py 不同
文件 /courses/Part3_batchnorm/scripts/04_batchnorm_implementation.py 和 -review/courses/Part3_batchnorm/scripts/04_batchnorm_implementation.py 不同
文件 /courses/Part3_batchnorm/scripts/05_deep_network.py 和 -review/courses/Part3_batchnorm/scripts/05_deep_network.py 不同
文件 /courses/Part3_batchnorm/scripts/06_diagnostic_tools.py 和 -review/courses/Part3_batchnorm/scripts/06_diagnostic_tools.py 不同
只在 /courses/Part3_batchnorm/scripts 中存在：diagnostic_tools.png
只在 /courses/Part3_batchnorm/scripts 中存在：kaiming_init.png
只在 /courses/Part3_batchnorm/scripts 中存在：tanh_saturation.png
文件 /courses/Part3_batchnorm/tutorial/01_diagnosis.md 和 -review/courses/Part3_batchnorm/tutorial/01_diagnosis.md 不同
文件 /courses/Part3_batchnorm/tutorial/02_batchnorm.md 和 -review/courses/Part3_batchnorm/tutorial/02_batchnorm.md 不同
文件 /courses/Part3_batchnorm/tutorial/03_deep_network.md 和 -review/courses/Part3_batchnorm/tutorial/03_deep_network.md 不同
文件 /courses/Part3_batchnorm/tutorial/README.md 和 -review/courses/Part3_batchnorm/tutorial/README.md 不同
只在 -review/courses/Part4_backprop/images 中存在：bn_backward_comparison.png
只在 -review/courses/Part4_backprop/images 中存在：dlogits_heatmap.png
只在 -review/courses/Part4_backprop/images 中存在：loss_curve_manual_training.png
只在 /courses/Part4_backprop 中存在：makemore_part4_backprop.ipynb
文件 /courses/Part4_backprop/scripts/04_cross_entropy_backward.py 和 -review/courses/Part4_backprop/scripts/04_cross_entropy_backward.py 不同
文件 /courses/Part4_backprop/scripts/05_batchnorm_backward.py 和 -review/courses/Part4_backprop/scripts/05_batchnorm_backward.py 不同
文件 /courses/Part4_backprop/scripts/06_manual_training.py 和 -review/courses/Part4_backprop/scripts/06_manual_training.py 不同
只在 /courses/Part4_backprop/scripts 中存在：dlogits_heatmap.png
文件 /courses/Part4_backprop/tutorial/01_why_backprop.md 和 -review/courses/Part4_backprop/tutorial/01_why_backprop.md 不同
文件 /courses/Part4_backprop/tutorial/02_forward_and_backward.md 和 -review/courses/Part4_backprop/tutorial/02_forward_and_backward.md 不同
文件 /courses/Part4_backprop/tutorial/03_simplified_and_training.md 和 -review/courses/Part4_backprop/tutorial/03_simplified_and_training.md 不同
文件 /courses/Part4_backprop/tutorial/README.md 和 -review/courses/Part4_backprop/tutorial/README.md 不同
只在 -review/courses/Part5_wavenet/images 中存在：wavenet_loss_comparison.png
只在 -review/courses/Part5_wavenet/images 中存在：wavenet_tree_fusion.png
只在 /courses/Part5_wavenet 中存在：makemore_part5_cnn1.ipynb
文件 /courses/Part5_wavenet/scripts/01_pytorchify_layers.py 和 -review/courses/Part5_wavenet/scripts/01_pytorchify_layers.py 不同
文件 /courses/Part5_wavenet/scripts/02_fix_lr_plot.py 和 -review/courses/Part5_wavenet/scripts/02_fix_lr_plot.py 不同
文件 /courses/Part5_wavenet/scripts/03_increase_context.py 和 -review/courses/Part5_wavenet/scripts/03_increase_context.py 不同
文件 /courses/Part5_wavenet/scripts/04_flatten_consecutive.py 和 -review/courses/Part5_wavenet/scripts/04_flatten_consecutive.py 不同
文件 /courses/Part5_wavenet/scripts/05_wavenet_architecture.py 和 -review/courses/Part5_wavenet/scripts/05_wavenet_architecture.py 不同
文件 /courses/Part5_wavenet/scripts/06_batchnorm_3d_fix.py 和 -review/courses/Part5_wavenet/scripts/06_batchnorm_3d_fix.py 不同
文件 /courses/Part5_wavenet/scripts/07_scaled_wavenet.py 和 -review/courses/Part5_wavenet/scripts/07_scaled_wavenet.py 不同
文件 /courses/Part5_wavenet/tutorial/01_pytorchify.md 和 -review/courses/Part5_wavenet/tutorial/01_pytorchify.md 不同
文件 /courses/Part5_wavenet/tutorial/02_wavenet_architecture.md 和 -review/courses/Part5_wavenet/tutorial/02_wavenet_architecture.md 不同
文件 /courses/Part5_wavenet/tutorial/03_training_and_bugs.md 和 -review/courses/Part5_wavenet/tutorial/03_training_and_bugs.md 不同
文件 /courses/Part5_wavenet/tutorial/README.md 和 -review/courses/Part5_wavenet/tutorial/README.md 不同
只在 /courses/Part6_transformer 中存在：gpt.py
只在 -review/courses/Part6_transformer 中存在：images
只在 /courses/Part6_transformer/scripts 中存在：01_explore_data.py
只在 /courses/Part6_transformer/scripts 中存在：02_bigram_baseline.py
只在 /courses/Part6_transformer/scripts 中存在：03_attention_trick.py
文件 /courses/Part6_transformer/scripts/04_self_attention.py 和 -review/courses/Part6_transformer/scripts/04_self_attention.py 不同
文件 /courses/Part6_transformer/scripts/05_multihead_feedforward.py 和 -review/courses/Part6_transformer/scripts/05_multihead_feedforward.py 不同
只在 /courses/Part6_transformer/scripts 中存在：06_layernorm_transformer.py
文件 /courses/Part6_transformer/scripts/07_scaleup_generate.py 和 -review/courses/Part6_transformer/scripts/07_scaleup_generate.py 不同
只在 -review/courses/Part6_transformer/scripts 中存在：__pycache__
文件 /courses/Part6_transformer/tutorial/01_data_and_tokenizer.md 和 -review/courses/Part6_transformer/tutorial/01_data_and_tokenizer.md 不同
文件 /courses/Part6_transformer/tutorial/02_attention_from_scratch.md 和 -review/courses/Part6_transformer/tutorial/02_attention_from_scratch.md 不同
文件 /courses/Part6_transformer/tutorial/03_transformer_block.md 和 -review/courses/Part6_transformer/tutorial/03_transformer_block.md 不同
文件 /courses/Part6_transformer/tutorial/04_beyond_transformer.md 和 -review/courses/Part6_transformer/tutorial/04_beyond_transformer.md 不同
文件 /courses/Part6_transformer/tutorial/README.md 和 -review/courses/Part6_transformer/tutorial/README.md 不同
只在 -review/courses/Part7_minimind 中存在：images
文件 /courses/Part7_minimind/scripts/03_gqa_kv_cache.py 和 -review/courses/Part7_minimind/scripts/03_gqa_kv_cache.py 不同
文件 /courses/Part7_minimind/scripts/04_swiglu_ffn_moe.py 和 -review/courses/Part7_minimind/scripts/04_swiglu_ffn_moe.py 不同
文件 /courses/Part7_minimind/scripts/06_pretrain_pipeline.py 和 -review/courses/Part7_minimind/scripts/06_pretrain_pipeline.py 不同
文件 /courses/Part7_minimind/scripts/07_sft_training.py 和 -review/courses/Part7_minimind/scripts/07_sft_training.py 不同
文件 /courses/Part7_minimind/scripts/08_dpo_alignment.py 和 -review/courses/Part7_minimind/scripts/08_dpo_alignment.py 不同
文件 /courses/Part7_minimind/scripts/09_eval_demo.py 和 -review/courses/Part7_minimind/scripts/09_eval_demo.py 不同
文件 /courses/Part7_minimind/scripts/12_mla_nsa_accounting.py 和 -review/courses/Part7_minimind/scripts/12_mla_nsa_accounting.py 不同
只在 /courses/Part7_minimind/scripts 中存在：output_long_context.png
只在 /courses/Part7_minimind/scripts 中存在：__pycache__
只在 /courses/Part7_minimind 中存在：temp
文件 /courses/Part7_minimind/tutorial/01_bpe_tokenizer.md 和 -review/courses/Part7_minimind/tutorial/01_bpe_tokenizer.md 不同
文件 /courses/Part7_minimind/tutorial/02_modern_components.md 和 -review/courses/Part7_minimind/tutorial/02_modern_components.md 不同
文件 /courses/Part7_minimind/tutorial/03_gqa_and_ffn.md 和 -review/courses/Part7_minimind/tutorial/03_gqa_and_ffn.md 不同
文件 /courses/Part7_minimind/tutorial/04_training_pipeline.md 和 -review/courses/Part7_minimind/tutorial/04_training_pipeline.md 不同
文件 /courses/Part7_minimind/tutorial/05_reproduce_minimind.md 和 -review/courses/Part7_minimind/tutorial/05_reproduce_minimind.md 不同
文件 /courses/Part7_minimind/tutorial/06_attention_mla_nsa.md 和 -review/courses/Part7_minimind/tutorial/06_attention_mla_nsa.md 不同
文件 /courses/Part7_minimind/tutorial/README.md 和 -review/courses/Part7_minimind/tutorial/README.md 不同
文件 /courses/Part8_post_training/scripts/02_pretrain.py 和 -review/courses/Part8_post_training/scripts/02_pretrain.py 不同
文件 /courses/Part8_post_training/scripts/03_sft.py 和 -review/courses/Part8_post_training/scripts/03_sft.py 不同
文件 /courses/Part8_post_training/scripts/06_ppo_training.py 和 -review/courses/Part8_post_training/scripts/06_ppo_training.py 不同
文件 /courses/Part8_post_training/scripts/07_grpo_training.py 和 -review/courses/Part8_post_training/scripts/07_grpo_training.py 不同
文件 /courses/Part8_post_training/scripts/08_eval_and_chat.py 和 -review/courses/Part8_post_training/scripts/08_eval_and_chat.py 不同
文件 /courses/Part8_post_training/scripts/09_quantize_and_serve.py 和 -review/courses/Part8_post_training/scripts/09_quantize_and_serve.py 不同
只在 /courses/Part8_post_training/scripts 中存在：09_reasoning_models.py
只在 -review/courses/Part8_post_training/scripts 中存在：13_reasoning_models.py
只在 /courses/Part8_post_training/scripts/__pycache__ 中存在：01_gpt_model.cpython-312.pyc
只在 -review/courses/Part8_post_training/scripts/__pycache__ 中存在：02_pretrain.cpython-312.pyc
文件 /courses/Part8_post_training/scripts/__pycache__/03_sft.cpython-312.pyc 和 -review/courses/Part8_post_training/scripts/__pycache__/03_sft.cpython-312.pyc 不同
文件 /courses/Part8_post_training/scripts/__pycache__/06_ppo_training.cpython-312.pyc 和 -review/courses/Part8_post_training/scripts/__pycache__/06_ppo_training.cpython-312.pyc 不同
文件 /courses/Part8_post_training/scripts/__pycache__/07_grpo_training.cpython-312.pyc 和 -review/courses/Part8_post_training/scripts/__pycache__/07_grpo_training.cpython-312.pyc 不同
只在 -review/courses/Part8_post_training/scripts/__pycache__ 中存在：08_eval_and_chat.cpython-312.pyc
只在 /courses/Part8_post_training/scripts/__pycache__ 中存在：09_quantize_and_serve.cpython-312.pyc
只在 /courses/Part8_post_training/scripts/__pycache__ 中存在：10_lora_from_scratch.cpython-312.pyc
只在 /courses/Part8_post_training/scripts/__pycache__ 中存在：11_hallucination_safety.cpython-312.pyc
只在 /courses/Part8_post_training/scripts/__pycache__ 中存在：12_lm_eval_hands_on.cpython-312.pyc
只在 -review/courses/Part8_post_training/scripts/__pycache__ 中存在：13_reasoning_models.cpython-312.pyc
只在 /courses/Part8_post_training 中存在：temp
文件 /courses/Part8_post_training/tutorial/01_gpt_and_pretrain.md 和 -review/courses/Part8_post_training/tutorial/01_gpt_and_pretrain.md 不同
文件 /courses/Part8_post_training/tutorial/02_sft_and_chat.md 和 -review/courses/Part8_post_training/tutorial/02_sft_and_chat.md 不同
文件 /courses/Part8_post_training/tutorial/03_reward_and_dpo.md 和 -review/courses/Part8_post_training/tutorial/03_reward_and_dpo.md 不同
文件 /courses/Part8_post_training/tutorial/04_ppo_and_grpo.md 和 -review/courses/Part8_post_training/tutorial/04_ppo_and_grpo.md 不同
文件 /courses/Part8_post_training/tutorial/05_eval_and_deploy.md 和 -review/courses/Part8_post_training/tutorial/05_eval_and_deploy.md 不同
文件 /courses/Part8_post_training/tutorial/06_inference_and_serving.md 和 -review/courses/Part8_post_training/tutorial/06_inference_and_serving.md 不同
文件 /courses/Part8_post_training/tutorial/07_evaluation.md 和 -review/courses/Part8_post_training/tutorial/07_evaluation.md 不同
文件 /courses/Part8_post_training/tutorial/08_lora_and_classification.md 和 -review/courses/Part8_post_training/tutorial/08_lora_and_classification.md 不同
文件 /courses/Part8_post_training/tutorial/09_reasoning_models.md 和 -review/courses/Part8_post_training/tutorial/09_reasoning_models.md 不同
文件 /courses/Part8_post_training/tutorial/README.md 和 -review/courses/Part8_post_training/tutorial/README.md 不同
只在 -review/courses/Part9_cuda_kernels 中存在：images
只在 /courses/Part9_cuda_kernels 中存在：scripts
文件 /courses/Part9_cuda_kernels/tutorial/01_gpu_and_first_kernel.md 和 -review/courses/Part9_cuda_kernels/tutorial/01_gpu_and_first_kernel.md 不同
文件 /courses/Part9_cuda_kernels/tutorial/02_matmul_optimization.md 和 -review/courses/Part9_cuda_kernels/tutorial/02_matmul_optimization.md 不同
文件 /courses/Part9_cuda_kernels/tutorial/03_profiling_and_cuda_apis.md 和 -review/courses/Part9_cuda_kernels/tutorial/03_profiling_and_cuda_apis.md 不同
文件 /courses/Part9_cuda_kernels/tutorial/04_triton_and_extensions.md 和 -review/courses/Part9_cuda_kernels/tutorial/04_triton_and_extensions.md 不同
文件 /courses/Part9_cuda_kernels/tutorial/05_flash_attention.md 和 -review/courses/Part9_cuda_kernels/tutorial/05_flash_attention.md 不同
文件 /courses/Part9_cuda_kernels/tutorial/README.md 和 -review/courses/Part9_cuda_kernels/tutorial/README.md 不同
只在 /assignments/assignment_1 中存在：bigram_exercises.py
文件 /assignments/assignment_1/README.md 和 -review/assignments/assignment_1/README.md 不同
文件 /assignments/assignment_1/test_bigram_exercises.py 和 -review/assignments/assignment_1/test_bigram_exercises.py 不同
文件 /assignments/assignment_10/assignment.md 和 -review/assignments/assignment_10/assignment.md 不同
只在 /assignments/assignment_10 中存在：__pycache__
只在 /assignments/assignment_10 中存在：.pytest_cache
文件 /assignments/assignment_10/test_distributed_exercises.py 和 -review/assignments/assignment_10/test_distributed_exercises.py 不同
文件 /assignments/assignment_11/alignment_exercises.py 和 -review/assignments/assignment_11/alignment_exercises.py 不同
文件 /assignments/assignment_11/assignment.md 和 -review/assignments/assignment_11/assignment.md 不同
只在 /assignments/assignment_11 中存在：__pycache__
文件 /assignments/assignment_11/test_alignment_exercises.py 和 -review/assignments/assignment_11/test_alignment_exercises.py 不同
文件 /assignments/assignment_12/assignment.md 和 -review/assignments/assignment_12/assignment.md 不同
只在 /assignments/assignment_12 中存在：finetune_exercises.py
只在 /assignments/assignment_12 中存在：__pycache__
只在 /assignments/assignment_12 中存在：.pytest_cache
只在 /assignments/assignment_12 中存在：test_finetune_exercises.py
文件 /assignments/assignment_13/assignment.md 和 -review/assignments/assignment_13/assignment.md 不同
只在 /assignments/assignment_13 中存在：data_exercises.py
只在 /assignments/assignment_13 中存在：__pycache__
只在 /assignments/assignment_13 中存在：.pytest_cache
只在 /assignments/assignment_13 中存在：test_data_exercises.py
文件 /assignments/assignment_14/assignment.md 和 -review/assignments/assignment_14/assignment.md 不同
只在 /assignments/assignment_14 中存在：__pycache__
只在 /assignments/assignment_14 中存在：.pytest_cache
只在 /assignments/assignment_14 中存在：serving_exercises.py
只在 /assignments/assignment_14 中存在：test_serving_exercises.py
文件 /assignments/assignment_15/assignment.md 和 -review/assignments/assignment_15/assignment.md 不同
只在 /assignments/assignment_15 中存在：__pycache__
只在 /assignments/assignment_15 中存在：.pytest_cache
只在 /assignments 中存在：assignment_16
文件 /assignments/assignment_17/agentic_exercises.py 和 -review/assignments/assignment_17/agentic_exercises.py 不同
文件 /assignments/assignment_17/assignment.md 和 -review/assignments/assignment_17/assignment.md 不同
文件 /assignments/assignment_17/__pycache__/agentic_exercises.cpython-312.pyc 和 -review/assignments/assignment_17/__pycache__/agentic_exercises.cpython-312.pyc 不同
只在 /assignments/assignment_17/__pycache__ 中存在：test_agentic_exercises.cpython-312.pyc
只在 /assignments/assignment_17/__pycache__ 中存在：test_agentic_exercises.cpython-312-pytest-9.1.1.pyc
只在 /assignments/assignment_17 中存在：.pytest_cache
只在 /assignments/assignment_17 中存在：test_agentic_exercises.py
文件 /assignments/assignment_18/assignment.md 和 -review/assignments/assignment_18/assignment.md 不同
只在 /assignments/assignment_18 中存在：__pycache__
只在 /assignments/assignment_18 中存在：rag_exercises.py
只在 /assignments/assignment_18 中存在：test_rag_exercises.py
文件 /assignments/assignment_19/assignment.md 和 -review/assignments/assignment_19/assignment.md 不同
文件 /assignments/assignment_2/mlp_exercises.py 和 -review/assignments/assignment_2/mlp_exercises.py 不同
文件 /assignments/assignment_2/README.md 和 -review/assignments/assignment_2/README.md 不同
文件 /assignments/assignment_2/test_mlp_exercises.py 和 -review/assignments/assignment_2/test_mlp_exercises.py 不同
文件 /assignments/assignment_3/batchnorm_exercises.py 和 -review/assignments/assignment_3/batchnorm_exercises.py 不同
文件 /assignments/assignment_3/README.md 和 -review/assignments/assignment_3/README.md 不同
文件 /assignments/assignment_3/test_batchnorm_exercises.py 和 -review/assignments/assignment_3/test_batchnorm_exercises.py 不同
文件 /assignments/assignment_4/backprop_exercises.py 和 -review/assignments/assignment_4/backprop_exercises.py 不同
只在 /assignments/assignment_4 中存在：__pycache__
文件 /assignments/assignment_4/README.md 和 -review/assignments/assignment_4/README.md 不同
文件 /assignments/assignment_4/test_backprop_exercises.py 和 -review/assignments/assignment_4/test_backprop_exercises.py 不同
只在 /assignments/assignment_5 中存在：__pycache__
文件 /assignments/assignment_5/README.md 和 -review/assignments/assignment_5/README.md 不同
文件 /assignments/assignment_5/wavenet_exercises.py 和 -review/assignments/assignment_5/wavenet_exercises.py 不同
文件 /assignments/assignment_6/assignment.md 和 -review/assignments/assignment_6/assignment.md 不同
只在 -review/assignments/assignment_6 中存在：__pycache__
只在 -review/assignments/assignment_6 中存在：.pytest_cache
文件 /assignments/assignment_7/assignment.md 和 -review/assignments/assignment_7/assignment.md 不同
只在 /assignments/assignment_7 中存在：minimind_exercises.py
只在 /assignments/assignment_7 中存在：test_minimind_exercises.py
文件 /assignments/assignment_8/assignment.md 和 -review/assignments/assignment_8/assignment.md 不同
只在 /assignments/assignment_8 中存在：__pycache__
文件 /assignments/assignment_8/test_post_training_exercises.py 和 -review/assignments/assignment_8/test_post_training_exercises.py 不同
只在 /assignments/assignment_9 中存在：__pycache__
文件 /assignments/assignment_9/test_cuda_exercises.py 和 -review/assignments/assignment_9/test_cuda_exercises.py 不同
```

## 形态优化 v2 追加清单（2026-09-05）

```
218
71
```
新增产物：widgets/（9 个单文件交互件与动画）、interview/（92 卡三格式）、tools/（extractor 与两个 CI 检查器）、ci/audit_gate.yml、site/（mkdocs 配置）、tools/build_site.py
