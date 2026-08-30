# 02 — vLLM 实战：离线推理 → 服务 → benchmark → 量化/投机解码

> 🧭 填空时间。装 vLLM（README 两案），同一模型（Qwen2.5-0.5B-Instruct）、同一批 prompt，
> 把 01 章的对比表填完。每一步都标注"对应 Part 8 06 章手写的哪一块"。

## 📖 前置知识

- **01 章**：指标定义与 naive 基线（181 tok/s / 6.3ms / 5.2ms，待超越）
- **Part 8 06 章**：PagedAttention/连续批处理/投机解码的手写模拟

## 1. 离线推理（5 行）

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")     # 启动日志找 'GPU KV cache size'
outs = llm.generate(["The quick brown fox", "Deep learning models are"],
                    SamplingParams(temperature=0.7, max_tokens=64))
print(outs[0].outputs[0].text)
```

启动日志三行必看：`# GPU blocks`（KV 池的块数——**PagedAttention 的页表容量**）、
`max seq len`、`GPU KV cache usage`（运行时占用率）。对照手写：Part 8 06 章模拟里
"整块预留浪费 41%"的问题，这里因为分页只剩 ~5%（论文口径 <4%）。

## 2. OpenAI 兼容服务

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 2048
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

概念映射：`--max-model-len` = KV 池上限；连续批处理自动开启（无需配置）；
`--gpu-memory-utilization` = KV 池占比调参；`--enable-prefix-caching`（新版默认开）=
共享前缀免重算（Part 8 06 章"prefix sharing"）。

## 3. Benchmark（官方脚本，01 章基线的同款指标）

```bash
# 服务端起好后：
python benchmarks/benchmark_serving.py --backend vllm \
  --model Qwen/Qwen2.5-0.5B-Instruct --dataset-name random --num-prompts 64 \
  --request-rate inf      # 吞吐、TTFT/TPOT 的 p50/p99 一次出齐
```

**填空表参考形态**（4090 实测量级，你的数字会不同——这正是要自己跑的原因）：
吞吐从 naive 的 ~181 tok/s 到 **数千 tok/s**（0.5B 小模型上 10×+；7B 上同样量级收益），
TPOT 反而可能略升（batch 大了 decode 稍慢）但吞吐大涨——**serving 的本质是吞吐换延迟**。

## 4. 量化服务（Part 8 06 章"手写量化"的工业对应）

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4   # GPTQ/AWQ/FP8 权重直接加载
# 对比：模型文件体积（fp16 ~1GB → int4 ~0.4GB）、KV 不变、吞吐与精度的实测变化
```

## 5. n-gram 投机解码（无需 draft 模型——24GB 卡友好）

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct",
          speculative_config={"method": "ngram", "prompt_lookup_num_tokens": 4})
# ngram 投机 = prompt lookup：从 prompt 已有文本里检索匹配片段当草稿 token，
# 所以不需要 draft 模型；prompt_lookup_num_tokens = 每步草稿长度
# 对比开/关投机解码的吞吐；Part 8 09 的"接受率 α"概念 = vLLM 日志里的 acceptance rate
```

## 6. 总账：本课程"手写 → 工具"的完整对照

| 手写（Part 8/9） | vLLM/工业 | 你能做的验证 |
|---|---|---|
| KV cache 字典（P7） | PagedAttention 块表 | 日志 KV usage + 显存对比 |
| 分页模拟（P8 09：41%→5%） | 真实 <4% | 同 batch 下 KV 显存 |
| 手写量化（P8 09） | GPTQ/AWQ 权重 | 文件体积 + ppl/acc |
| 手写投机解码（P8 09，α≈0.60） | n-gram/EAGLE | acceptance rate + 吞吐 |
| naive/静态批基线（P14 01） | 连续批处理 | 三行对比表 |

> 🔑 面试结论模板："我在 4090 上用 Qwen2.5-0.5B 做过 naive→vLLM 的对比，吞吐 181→N tok/s，
> 差异归因于连续批处理和 PagedAttention——我的手写模拟复现了同样的方向性。"——
> 这段话的每个数字你都能现场推导。

## 学完本部分你能...

- ✅ 独立完成 vLLM 安装（两案）、离线推理、OpenAI 服务部署
- ✅ 用官方 benchmark 出 TTFT/TPOT 分位数并填完对比表
- ✅ 部署量化模型与 n-gram 投机解码，解释各自的适用条件
- ✅ 把课程的手写模块逐一对应到工业实现，形成"懂原理 + 会工具"的完整叙事

**课后练习**

<details>
<summary>Q1: vLLM 的 TPOT 有时比 naive 略高，为什么还用它？</summary>
A: 大 batch 下 decode 步变慢（每步算更多请求），但吞吐（tok/s 合计）大增——
serving 优化的是"每瓦特/每卡的 token 成本"。单请求延迟敏感场景用小 batch/SLO 路由，
吞吐场景用大 batch——goodput 的含义（Part 8 06 章）。
</details>

<details>
<summary>Q2: prefix caching 什么时候收益最大？什么时候没收益？</summary>
A: 共享前缀长且重复率高（系统提示词、few-shot 模板、多轮对话历史）时收益巨大
（TTFT 降几倍）；prompt 完全随机时纯开销（查表成本）。看业务 prompt 分布决定开关。
</details>

## 📝 课后作业

👉 [Assignment 14](../../../assignments/assignment_14/)

## 🎓 课程毕业

Part 1-14 完整覆盖：从手写 bigram 到工业 RL 与 serving。
回 [课程总览](../../../README.md) 规划你的 Part 15（多模态）与面试（docs/llm_interview_guide.md）。
