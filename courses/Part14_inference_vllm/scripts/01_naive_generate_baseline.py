#!/usr/bin/env python3
"""
Part 14 - 脚本 01: 朴素生成基线（vLLM 对比实验的"手写侧"）
目标：建立 serving 对比的基准线。用 HuggingFace naive 逐请求生成循环测
      TTFT / TPOT / 吞吐，产出的数字将和 Part 14 脚本 02（vLLM）在同一模型、
      同一批 prompt 上对比——这就是"手写 vs 工具"的实验设计。
对应教程：tutorial/01_handwritten_to_vllm.md

运行（需要 HF 下载 Qwen2.5-0.5B-Instruct ≈1GB，safetensors；GPU 建议约 2 分钟）：
    python 01_naive_generate_baseline.py
输出：每请求 TTFT/TPOT/吞吐 + 一张"待 vLLM 填空"的对比表。
"""

import os
import sys
import time
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # 与 02 章 vLLM、Part 11 verl quickstart 同款
                                       # （safetensors 权重，新版 transformers 可安全加载）
PROMPTS = [
    "The quick brown fox",
    "Deep learning models are",
    "The central bank announced",
    "Once upon a time in a distant",
    "Data quality matters because",
    "The engine was optimized for",
    "Researchers discovered that",
    "In the final minutes of the",
] * 8                              # 64 个请求（与 vLLM benchmark 同规模语义）


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("═══ 朴素生成基线（HF 逐请求循环）═══")
    print(f"  device={device}, model={MODEL}, requests={len(PROMPTS)}\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'   # decoder-only 批处理必须左 padding（transformers 会警告你）
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    if device == 'cpu':            # CPU 上 0.5B 很慢：缩规模保可跑（数字仍具方向性）
        prompts_use, max_new = PROMPTS[:8], 8
    else:
        prompts_use, max_new = PROMPTS, 32

    # ── 逐请求生成（serving 反模式：一次一个请求，其余全在排队）──
    ttfts, tpots, all_t = [], [], []
    gen_tokens = 0
    t_total0 = time.perf_counter()
    for p in prompts_use:
        ids = tok(p, return_tensors='pt').to(device)
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        first = time.perf_counter()
        # generate 是一次调用，TTFT 用"首 token 时间"近似：单步探测
        t1 = time.perf_counter()
        _ = model.generate(**ids, max_new_tokens=1, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        ttft = time.perf_counter() - t1
        total = first - t0
        ttfts.append(ttft)
        tpots.append((total - ttft) / max(max_new - 1, 1))
        all_t.append(total)
        gen_tokens += max_new
    t_total = time.perf_counter() - t_total0

    print(f"[1] 逐请求循环（serving 反模式基线）:")
    print(f"    TTFT  p50/p90 : {sorted(ttfts)[len(ttfts)//2]*1000:.1f} / "
          f"{sorted(ttfts)[int(len(ttfts)*0.9)]*1000:.1f} ms")
    print(f"    TPOT  p50/p90 : {sorted(tpots)[len(tpots)//2]*1000:.1f} / "
          f"{sorted(tpots)[int(len(tpots)*0.9)]*1000:.1f} ms")
    print(f"    吞吐          : {gen_tokens / t_total:.0f} tok/s（wall {t_total:.2f}s）")
    print(f"    每请求平均    : {sum(all_t)/len(all_t)*1000:.0f} ms")

    # ── 静态批处理对照（serving 进化第一步：一起跑但一起等）──
    batch = tok(prompts_use[:8], return_tensors='pt', padding=True).to(device)
    t0 = time.perf_counter()
    _ = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    t_batch = time.perf_counter() - t0
    print(f"\n[2] 静态批处理（batch=8）: {(8*max_new)/t_batch:.0f} tok/s"
          f"（吞吐↑ 但早完成的请求也要等最慢的 —— 'static batching' 的浪费）")

    print(f"""
═══ 待 vLLM 填空的对比表（跑脚本 02 后回填）═══
  {'指标':<26}{'naive 循环（本脚本）':>22}{'vLLM（脚本 02）':>18}
  {'-'*68}
  {'吞吐 tok/s':<28}{gen_tokens / t_total:>20.0f}{'?':>18}
  {'TTFT p50 ms':<27}{sorted(ttfts)[len(ttfts)//2]*1000:>20.1f}{'?':>18}
  {'TPOT p50 ms':<27}{sorted(tpots)[len(tpots)//2]*1000:>20.1f}{'?':>18}
  对应关系：连续批处理 vs 逐请求循环、PagedAttention vs KV cache dict、
  prefix caching vs 每请求冷启动 —— 每一行都是 Part 8 06 章"手写模拟"的工业对应。
  💡 面试："你怎么证明 vLLM 快？"→ 同模型/同 prompt/同指标的三行对比表，比背论文有力。""")


if __name__ == '__main__':
    main()
