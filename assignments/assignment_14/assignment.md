# Assignment 14：推理部署（vLLM）

> 对应 Part 14 教程（[01 朴素基线](../../courses/Part14_inference_vllm/tutorial/01_naive_baseline.md) / [02 vLLM 实战](../../courses/Part14_inference_vllm/tutorial/02_vllm_serving.md)）。
> 三题纸笔可完成；实验题跑 01/02 两个脚本。

## 题目（实现 `serving_exercises.py`）

1. **serving 指标**（30 分）：E2E = TTFT + TPOT×(n−1)；吞吐 = 总 token / 墙钟
2. **KV 容量账**（35 分）：KV 公式（LLaMA-7B fp16 seq2048 = 1.07GB；GQA 1/4）+
   给定预算反推最大并发 batch
3. **批处理浪费**（35 分）：静态批处理按 max pad 的浪费率（100/10/10/10 → 67.5%！）
   vs 连续批处理理想 0——量化 Orca 论文的动机

## 实验题（4090，观测型）

- 跑脚本 01 出 naive 基线（吞吐/TTFT/TPOT），装 vLLM（两案选一）跑 02 章，
  填完三行对比表——面试即用的实证
- 打开 `--enable-prefix-caching`，构造"共享系统提示词的 64 请求"vs"随机 64 请求"，
  对比 TTFT 差异，写出 prefix caching 的适用条件

## 🎯 面试直通车

- "你怎么证明 vLLM 快？"——同模型/prompt/指标的三行对比表 + 归因（连续批处理/分页/缓存）
- "TPOT 变差了为什么还用它？"——serving 优化吞吐成本，单请求延迟用 SLO 路由（goodput）
- "GQA 省的是什么？"——KV cache 随 kv_heads 线性缩（题 2 第一手数字：1.07GB→0.27GB）
- "静态批处理的浪费怎么算？"——题 3：pad 到 max 的份额；连续批处理消掉它
