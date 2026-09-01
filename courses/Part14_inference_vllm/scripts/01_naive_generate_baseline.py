#!/usr/bin/env python3
"""
Part 14 - 脚本 01: 朴素生成基线（vLLM 对比实验的"手写侧"）
目标：建立 serving 对比的基准线。用 HuggingFace naive 逐请求生成循环测
      TTFT / TPOT / 吞吐，产出的数字将和 02 章 vLLM 实操（CLI，无脚本）在同一模型、
      同一批 prompt 上对比——这就是"手写 vs 工具"的实验设计。
对应教程：tutorial/01_naive_baseline.md

运行（需要 HF 下载 Qwen2.5-0.5B-Instruct ≈1GB，safetensors；GPU 约 20s）：
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
    import transformers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)  # CPU 无需同步
    print("═══ 朴素生成基线（HF 逐请求循环）═══")
    print(f"  device={device}({gpu}), torch={torch.__version__}, "
          f"transformers={transformers.__version__}")
    print(f"  model={MODEL}, requests={len(PROMPTS)}\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'   # decoder-only 批处理必须左 padding（transformers 会警告你）
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
    if device == 'cpu':            # CPU 上 0.5B 很慢：缩规模保可跑（数字仍具方向性）
        prompts_use, max_new = PROMPTS[:8], 8
    else:
        prompts_use, max_new = PROMPTS, 32

    # ── 逐请求生成（serving 反模式：一次一个请求，其余全在排队）──
    # ⚠️ 异步陷阱：GPU 是异步的，每个计时点前后都必须 torch.cuda.synchronize()，
    #    否则测到的是"提交 kernel 的时间"而不是"算完的时间"（Part 9 01 章）。
    ttfts, tpots, all_t = [], [], []
    gen_tokens = 0
    t_total0 = time.perf_counter()
    for p in prompts_use:
        ids = tok(p, return_tensors='pt').to(device)
        sync()                          # 计时起点干净：排除上一次/输入拷贝的尾巴
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        sync()                          # ← 等这一批 32 个 token 真正算完
        first = time.perf_counter()
        # generate 是一次调用，TTFT 用"首 token 时间"近似：单步探测。
        # ⚠️ 口径：探测在本请求正式计时段之外——先完成上面 32 token 的正式计时，
        #    再做单步探测；探测的 1 个 token 也不计入 gen_tokens（分子分母同口径）。
        sync()                          # 排掉上一段 generate 可能残留的收尾工作
        t1 = time.perf_counter()
        _ = model.generate(**ids, max_new_tokens=1, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        sync()                          # ← 首 token（prefill+1 步 decode）真实耗时
        ttft = time.perf_counter() - t1
        total = first - t0
        ttfts.append(ttft)
        tpots.append((total - ttft) / max(max_new - 1, 1))
        all_t.append(total)
        gen_tokens += max_new
    sync()
    t_total = time.perf_counter() - t_total0
    t_gen = sum(all_t)                  # 正式计时段合计（64 次 32-token generate）：
                                        # TTFT 探测与 tokenize 开销不计入，与 gen_tokens 同口径

    print(f"[1] 逐请求循环（serving 反模式基线）:")
    print(f"    TTFT  p50/p90 : {sorted(ttfts)[len(ttfts)//2]*1000:.1f} / "
          f"{sorted(ttfts)[int(len(ttfts)*0.9)]*1000:.1f} ms")
    print(f"    TPOT  p50/p90 : {sorted(tpots)[len(tpots)//2]*1000:.1f} / "
          f"{sorted(tpots)[int(len(tpots)*0.9)]*1000:.1f} ms")
    print(f"    吞吐          : {gen_tokens / t_gen:.0f} tok/s"
          f"（计时段 {t_gen:.2f}s；wall {t_total:.2f}s 含 TTFT 探测，不作分母）")
    print(f"    每请求平均    : {sum(all_t)/len(all_t)*1000:.0f} ms")

    # ── 静态批处理对照（serving 进化第一步：一起跑但一起等）──
    batch = tok(prompts_use[:8], return_tensors='pt', padding=True).to(device)
    sync()                              # 起点干净：padding 张量拷贝完成后再计时
    t0 = time.perf_counter()
    _ = model.generate(**batch, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    sync()                              # ← 8 路静态批真正算完
    t_batch = time.perf_counter() - t0
    print(f"\n[2] 静态批处理（batch=8）: {(8*max_new)/t_batch:.0f} tok/s"
          f"（吞吐↑ 但早完成的请求也要等最慢的 —— 'static batching' 的浪费）")

    print(f"""
═══ 待 vLLM 填空的对比表（完成 02 章 CLI 实操后回填）═══
  {'指标':<26}{'naive 循环（本脚本）':>22}{'vLLM（02 章实操）':>18}
  {'-'*68}
  {'吞吐 tok/s':<28}{gen_tokens / t_gen:>20.0f}{'?':>18}
  {'TTFT p50 ms':<27}{sorted(ttfts)[len(ttfts)//2]*1000:>20.1f}{'?':>18}
  {'TPOT p50 ms':<27}{sorted(tpots)[len(tpots)//2]*1000:>20.1f}{'?':>18}
  对应关系：连续批处理 vs 逐请求循环、PagedAttention vs KV cache dict、
  prefix caching vs 每请求冷启动 —— 每一行都是 Part 8 06 章"手写模拟"的工业对应。
  💡 面试："你怎么证明 vLLM 快？"→ 同模型/同 prompt/同指标的三行对比表，比背论文有力。""")


if __name__ == '__main__':
    main()
