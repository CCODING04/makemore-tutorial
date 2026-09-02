#!/usr/bin/env python3
"""
Part 9 - 脚本 09: Flash Attention —— 手写 Triton 内核（本部分的"毕业内核"）
目标：把 02 章的 tiling 直觉 + 07 章 Triton 的 softmax 内核合体，写出带 online softmax
      与 causal mask 的 Flash Attention 前向，对照 PyTorch SDPA 的全部后端验收
      正确性与性能，最后看 FlexAttention 怎么把"变体 mask"也变成 Triton 内核。

结构（五段）：
  段 1  基准：naive attention + SDPA 四后端逐一锁定，用 profiler 打印"实际命中者"
  段 2  内核：_fa_fwd —— online softmax（exp2 技巧）+ causal 三阶段分解 + 边界 mask
  段 3  数值验收：与 naive 对照（对齐 / 不对齐 seq、causal / 非因果、泄漏检查）
  段 4  性能：seq={1K,2K,4K} x {causal,non-causal}，验收承诺 >= SDPA 最优后端 50%
  段 5  生态：FlexAttention 复现 causal / sliding 窗口；SageAttention（文字）

运行（需要 GPU + torch 自带的 triton；整脚本约 2-4 分钟，大头是 autotune 编译）:
    python 09_flash_attention_triton.py
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel

LOG2E = 1.44269504  # log2(e)：exp(x) = exp2(x * log2(e))，GPU 上 exp2 快一个量级


# =====================================================================
# 段 1 基准：naive attention（教科书公式，显式物化 T x T 矩阵）
# =====================================================================
def naive_attention(q, k, v, causal, fp32=True):
    """教科书版 attention：S = QK^T/sqrt(d) -> mask -> softmax -> @V。

    q/k/v: (B, H, T, D)。fp32=True 时上转 fp32 计算（数值参考）；
    fp32=False 直接用输入 dtype（bf16，性能基准用，和其余实现同精度）。
    中间矩阵 S 与 P 各占 (B,H,T,T)——Flash Attention 要消灭的就是它们。
    """
    if fp32:
        q, k, v = q.float(), k.float(), v.float()
    T = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    att = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)  <- 巨型中间矩阵
    if causal:
        keep = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        att = att.masked_fill(~keep, float('-inf'))
    p = att.softmax(dim=-1)                          # (B, H, T, T)  <- 又一个
    return p @ v                                     # (B, H, T, D)


def gpu_kernel_names(fn, top=2):
    """跑一次 fn，用 profiler 抓真正执行的 GPU kernel 名（证据，不靠猜）。"""
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ks = [(e.key, e.self_device_time_total) for e in prof.key_averages()
          if e.self_device_time_total > 0]
    ks.sort(key=lambda x: -x[1])
    return ks[:top]


def classify_backend(kernel_names):
    """按 kernel 名里的特征串判断刚才跑的是哪个 SDPA 后端。"""
    joined = ' '.join(kernel_names).lower()
    if 'flash' in joined:
        return 'flash'
    if 'cudnn' in joined:
        return 'cudnn'
    if 'fmha' in joined:
        return 'efficient'           # mem-efficient 的 cutlass fmha_* kernel
    if 'softmax' in joined or 'gemm' in joined or 'elementwise' in joined:
        return 'math'                # math = 拆成多个 eager kernel 的朴素路径
    return 'unknown'


SDPA_BACKENDS = [
    ('flash', SDPBackend.FLASH_ATTENTION),
    ('efficient', SDPBackend.EFFICIENT_ATTENTION),
    ('cudnn', SDPBackend.CUDNN_ATTENTION),
    ('math', SDPBackend.MATH),
]


def probe_sdpa(q, k, v, causal):
    """逐个锁定后端跑 SDPA：能跑则用 profiler 记录证据 kernel，跑不动则记 None。

    返回 {名字: kernel 名列表}。注意不能假设 4090 一定命中 flash——锁谁跑谁，
    锁不住（RuntimeError: No available kernel）就是不可用。
    """
    avail = {}
    for name, be in SDPA_BACKENDS:
        try:
            with sdpa_kernel([be]):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
                kernels = [kn for kn, _t in
                           gpu_kernel_names(lambda: F.scaled_dot_product_attention(
                               q, k, v, is_causal=causal))]
            avail[name] = kernels
        except RuntimeError as e:
            print(f"  [sdpa] lock {name:9s} -> 不可用 ({str(e)[:60]}...)")
            avail[name] = None
    return avail


# =====================================================================
# 段 2 内核：Flash Attention 前向（online softmax + causal 三阶段）
# =====================================================================
@triton.jit
def _fa_fwd_inner(acc, l_i, m_i, q,
                  K, V, base, offs_m,
                  n_lo, n_hi, qk_scale,
                  N_CTX,
                  MASK_MODE: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  HEAD_DIM: tl.constexpr):
    """沿 K/V 扫 [n_lo, n_hi) 这些块，滚动更新 (acc, l_i, m_i)。

    MASK_MODE: 0 = 完全无 mask（causal 的带外块，块内元素必然合法）
               1 = 只挡列越界（非 causal 扫到 T 尾部时）
               2 = causal 对角块 + 列越界
    """
    offs_n = tl.arange(0, BLOCK_N)                       # 块内 key 相对下标 (BLOCK_N,)
    offs_d = tl.arange(0, HEAD_DIM)                      # head 维下标 (HEAD_DIM,)
    for start_n in range(n_lo, n_hi, BLOCK_N):
        offs_kv = start_n + offs_n                       # key/value 的绝对下标 (BLOCK_N,)
        kv_ok = offs_kv < N_CTX                          # 列越界（T 不是 BLOCK_N 倍数时出现）

        # K^T 直接按 (HEAD_DIM, BLOCK_N) 布局加载，喂给 tl.dot 免转置；
        # 越界列填 0（参与 dot 无害，随后会被 where 挡掉）
        kT = tl.load(K + base + offs_kv[None, :] * HEAD_DIM + offs_d[:, None],
                     mask=kv_ok[None, :], other=0.0)     # (HEAD_DIM, BLOCK_N) bf16
        qk = tl.dot(q, kT)                               # (BLOCK_M, BLOCK_N) fp32 累加

        # ---- online softmax 的 5 行核心 ----
        # qk_scale 已乘 LOG2E：把 exp 换成 exp2（见教程推导）
        if MASK_MODE == 0:
            qk_scaled = qk * qk_scale
        elif MASK_MODE == 1:
            qk_scaled = qk * qk_scale + tl.where(kv_ok[None, :], 0.0, -1.0e6)
        else:  # MASK_MODE == 2：causal 对角块（下三角才合法）
            valid = (offs_m[:, None] >= offs_kv[None, :]) & kv_ok[None, :]
            qk_scaled = qk * qk_scale + tl.where(valid, 0.0, -1.0e6)
        # ⚠️ 用 -1.0e6 不用 -inf：全 -inf 行会让 max 与被减项都是 -inf，
        #    -inf - (-inf) = NaN 并污染整个块（见教程"陷阱 1"）
        m_new = tl.maximum(m_i, tl.max(qk_scaled, 1))    # 行最大值（fp32）
        p = tl.math.exp2(qk_scaled - m_new[:, None])     # 稳定 softmax 分子（未归一）
        alpha = tl.math.exp2(m_i - m_new)                # 旧累加和的缩放因子
        l_i = l_i * alpha + tl.sum(p, 1)                 # 分母滚动和（fp32）
        v = tl.load(V + base + offs_kv[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=kv_ok[:, None], other=0.0)      # (BLOCK_N, HEAD_DIM) bf16
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # 输出滚动累加
        m_i = m_new
    return acc, l_i, m_i


# autotune 网格：16 个组合，让编译器替我们选块大小 / warp 数 / 流水线深度
#（02 章"通向 cuBLAS 还差什么"表里的 Autotuning 一行，落到实处）
fa_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_warps=w, num_stages=s)
    for BM in [64, 128]
    for BN in [64, 128]
    for w in [4, 8]
    for s in [2, 3]
]


@triton.autotune(configs=fa_configs, key=['N_CTX', 'HEAD_DIM'])
@triton.jit
def _fa_fwd(Q, K, V, O, sm_scale, N_CTX,
            HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    """一个 program 算 O 的一块（BLOCK_M 行）。Q 块全程驻留"片上"，
    K/V 分块流过，softmax 状态 (m, l) 与输出 acc 滚动更新——从不物化 (T, T) 矩阵。
    布局约定：q/k/v/o 均为 (B, H, T, D) 连续（wrapper 里断言），
    则 q[b,h,t,d] = flat[(b*H+h)*T*D + t*D + d]，一个偏移 base 搞定。
    """
    start_m = tl.program_id(0)                 # 第几个 query 行块
    off_bh = tl.program_id(1)                 # 展平的 (batch, head) 编号
    base = off_bh.to(tl.int64) * N_CTX * HEAD_DIM   # int64：大张量防溢出

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # 本块 query 行号 (BLOCK_M,)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q 块：(BLOCK_M, HEAD_DIM)，行越界（最后一个不完整块）填 0，无碍——那些行不写回
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_m[:, None] < N_CTX, other=0.0)

    # online softmax 状态，全部 fp32（bf16 累加会丢精度，见教程"陷阱 2"）
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # 历史行最大值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 分母滚动和
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # 输出累加器

    qk_scale = sm_scale * 1.44269504           # scale * log2(e)：exp 一次换成 exp2

    # STAGE=1 非因果：整行一趟扫完（只需挡列越界）
    # STAGE=3 causal 三阶段分解（对齐官方 Triton tutorial 06）：
    #   阶段 1 带外块 [0, diag_lo)            —— 无 mask，最快路径
    #   阶段 2 对角块 [diag_lo, 行块末尾)      —— 逐元素 causal mask
    #   阶段 3 [行块末尾, T)                   —— 全非法，直接不进循环（省一半算力）
    if STAGE == 1:
        acc, l_i, m_i = _fa_fwd_inner(
            acc, l_i, m_i, q, K, V, base, offs_m,
            0, N_CTX, qk_scale, N_CTX, 1, BLOCK_M, BLOCK_N, HEAD_DIM)
    else:
        # 对角带起点向下对齐到 BLOCK_N 的倍数：保证阶段 1/2 的块边界不错切
        #（BLOCK_N > BLOCK_M 时若直接用 start_m*BLOCK_M，阶段 1 会越界扫进对角带）
        diag_lo = (start_m * BLOCK_M) // BLOCK_N * BLOCK_N
        acc, l_i, m_i = _fa_fwd_inner(
            acc, l_i, m_i, q, K, V, base, offs_m,
            0, diag_lo, qk_scale, N_CTX, 0, BLOCK_M, BLOCK_N, HEAD_DIM)
        acc, l_i, m_i = _fa_fwd_inner(
            acc, l_i, m_i, q, K, V, base, offs_m,
            diag_lo, (start_m + 1) * BLOCK_M, qk_scale, N_CTX, 2,
            BLOCK_M, BLOCK_N, HEAD_DIM)

    # epilogue：分子 / 分母 = 归一化。合法行的 l_i 至少含自身位置，>0
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < N_CTX)    # 行越界的不写回


def fa_forward(q, k, v, causal=True):
    """Flash Attention 前向（教学版 wrapper）。

    Args:
        q/k/v: (B, H, T, D) bf16（tl.dot 需要 fp16/bf16 输入走 Tensor Core）
        causal: 是否下三角因果 mask
    Returns:
        (B, H, T, D) bf16；softmax 状态与累加全程 fp32。
    """
    assert q.shape == k.shape == v.shape
    B, H, T, D = q.shape
    assert D in (16, 32, 64, 128, 256), "HEAD_DIM 必须是 tl.dot 支持的 2 的幂"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), \
        "教学版只接受连续布局（工业版会用 stride 参数任意支持）"
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    stage = 3 if causal else 1
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_M']), B * H)
    _fa_fwd[grid](q, k, v, o, sm_scale, T,
                  HEAD_DIM=D, STAGE=stage)
    return o


# =====================================================================
# 段 3/4/5 的辅助
# =====================================================================
def bench(fn, warmup=10, iters=50):
    """torch.cuda.Event 计时：预热 warmup 次，测 iters 次取平均（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def attn_flops(B, H, T, D, causal):
    """attention 前向 FLOPs：QK^T 与 PV 两个 T*T*D matmul，causal 减半。"""
    flops = 4.0 * B * H * T * T * D          # 2 个 matmul x 2 FLOP/MAC
    return flops * (0.5 if causal else 1.0)


def max_rel_err(tri, ref, thresh=0.1):
    """在 |ref| >= thresh 的"有意义区间"上算最大相对误差。

    全量逐元素相对误差会被 |ref|->0 处的 bf16 量化噪声主导（分母趋 0，
    SDPA flash 也一样大），那部分误差绝对值极小，交给 assert_close 的
    atol 判定；区间内则反映真实的 exp2/scale 顺序差异（~1e-3 量级）。
    """
    tri, ref = tri.float(), ref.float()
    mask = ref.abs() >= thresh
    return ((tri - ref).abs()[mask] / ref.abs()[mask]).max().item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA GPU required (Part 9 Python 脚本需要 GPU)。")

    torch.manual_seed(42)
    print("=" * 74)
    print("Part 9 - 脚本 09: Flash Attention 手写 Triton 内核")
    print(f"环境: {torch.cuda.get_device_name(0)} | torch {torch.__version__} "
          f"| triton {triton.__version__}")
    print("=" * 74)

    # 统一实验形状：D=64 对齐 minimind 小配置的 head_dim（hidden 512 / 8 heads）
    B, H, T0, D = 2, 8, 1024, 64
    q = torch.randn(B, H, T0, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    print(f"\n输入形状: q/k/v = (B={B}, H={H}, T={T0}, D={D}) bf16"
          f"  <- minimind 风格（D=64）")

    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("段 1  基准: naive attention + SDPA 四后端锁定（打印实际命中者）")
    print("-" * 74)

    t_naive = bench(lambda: naive_attention(q, k, v, causal=True, fp32=False))
    naive_s_mem = B * H * T0 * T0 * q.element_size() / 1e9
    print(f"[naive]  causal 前向 {t_naive:.3f} ms；显式物化两个 (B,H,T,T) 矩阵，"
          f"每个 {naive_s_mem:.2f} GB（bf16）")
    print(f"[naive]  T 翻倍 -> 显存翻 4 倍（T^2），4K 序列时每个矩阵 "
          f"{naive_s_mem * 16:.1f} GB —— 这就是要消灭的东西")

    avail = probe_sdpa(q, k, v, causal=True)
    print("\n  锁定后端        实际命中（profiler 证据 kernel）")
    print("  " + "-" * 66)
    for name, kernels in avail.items():
        if kernels is None:
            continue
        hit = classify_backend(kernels)
        flag = "OK " if hit == name else "?! "
        print(f"  {flag}{name:9s} -> {hit:9s} | {kernels[0][:52]}")
    default_kernels = [n for n, _ in gpu_kernel_names(
        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))]
    print(f"\n  [默认调度（不锁定）] 命中 {classify_backend(default_kernels)}"
          f" | {default_kernels[0][:52]}")
    print("  📝 4090(SM89) 上 flash/efficient/cudnn 都可用；FA3 的 WGMMA 是 SM90 "
          "指令，只有 Hopper 能跑（教程演进表）")

    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("段 2  内核: _fa_fwd（online softmax + causal 三阶段）——编译中...")
    print("-" * 74)
    out = fa_forward(q, k, v, causal=True)
    torch.cuda.synchronize()
    try:
        cfg = _fa_fwd.best_config
        bm, bn = cfg.kwargs['BLOCK_M'], cfg.kwargs['BLOCK_N']
        print(f"[autotune] T={T0} 选中 BLOCK_M={bm} BLOCK_N={bn} "
              f"num_warps={cfg.num_warps} num_stages={cfg.num_stages}"
              f"（16 个组合实测选出）")
        # 示例块号钳制：BM=128 且 T=1024 时只有 0..7 号块，写死"第 8 块"会指向
        # 不存在的行——取"倒数第二块"保证三种 BLOCK 组合下示例都真实存在
        n_blocks = (T0 + bm - 1) // bm
        blk = max(min(8, n_blocks - 2), 0)
        r0, r1 = blk * bm, min((blk + 1) * bm, T0)
        diag_lo = r0 // bn * bn
        print(f"[causal 三阶段] 以第 {blk} 个 query 块（行 {r0}..{r1}）为例：")
        print(f"               带外块 [0, {diag_lo}) 无 mask | 对角块 "
              f"[{diag_lo}, {r1}) 逐元素 mask | [{r1}, {T0}) 直接跳过")
    except AttributeError:
        pass
    print(f"[fa_forward] 输出形状 {tuple(out.shape)}，dtype {out.dtype}"
          f"（内部 m/l/acc 全 fp32）")

    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("段 3  数值验收: 与 naive fp32 参考对照")
    print("-" * 74)
    print("  📝 exp2/scale 折叠顺序 + bf16 量化会引入 1e-3 量级差异，属预期；")
    print("     阈值 rtol=atol=1e-2 与官方 Triton tutorial 一致")

    for T_check in (1024, 1000):    # 1000: T % 64 != 0，专门打边界 mask
        qc = torch.randn(B, H, T_check, D, device='cuda', dtype=torch.bfloat16)
        kc = torch.randn_like(qc)
        vc = torch.randn_like(qc)
        for causal in (True, False):
            ref = naive_attention(qc, kc, vc, causal=causal, fp32=True)
            tri = fa_forward(qc, kc, vc, causal=causal)
            torch.testing.assert_close(tri.float(), ref, rtol=1e-2, atol=1e-2)
            sd = F.scaled_dot_product_attention(qc, kc, vc, is_causal=causal)
            abs_err = (tri.float() - ref).abs().max().item()
            print(f"  [T={T_check} causal={causal!s:5s}] assert_close PASS | "
                  f"max|Δ| {abs_err:.2e} | 相对误差(|ref|>=0.1 处) "
                  f"{max_rel_err(tri, ref):.1e} (SDPA flash 同口径 "
                  f"{max_rel_err(sd, ref):.1e})")
    print("  [结论] 4 组（对齐/不对齐 x 因果/非因果）全部通过")

    # causal 泄漏检查：改动"未来"的 v，历史行输出必须一字不变
    out = fa_forward(q, k, v, causal=True)
    v_future = v.clone()
    v_future[:, :, T0 // 2 + 1:, :] += 1.0       # 污染后半段所有 value
    out2 = fa_forward(q, k, v_future, causal=True)
    past = (out2[:, :, :T0 // 2 + 1, :] - out[:, :, :T0 // 2 + 1, :]).abs().max().item()
    future = (out2[:, :, T0 // 2 + 1:, :] - out[:, :, T0 // 2 + 1:, :]).abs().max().item()
    print(f"  [泄漏检查] 改动 v[j>i]: 行 <=i 输出最大变化 = {past:.2e} (应=0)，"
          f"行 >i 最大变化 = {future:.2f} (应>0) -> "
          f"{'无泄漏 OK' if past == 0.0 and future > 0 else 'FAIL'}")

    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("段 4  性能: 验收承诺 —— 教学版前向吞吐 >= SDPA 最优后端的 50% 合格，"
          ">85% 优秀")
    print("       （依据: PyTorch 官方 FlexAttention 博客：Triton 路径为 FA2 前向")
    print("        的 90%，即 Triton 通用内核的上限；教学版再让一档到 85%）")
    print("-" * 74)

    rows = []
    for T in (1024, 2048, 4096):
        qp = torch.randn(B, H, T, D, device='cuda', dtype=torch.bfloat16)
        kp, vp = torch.randn_like(qp), torch.randn_like(qp)
        for causal in (True, False):
            fl = attn_flops(B, H, T, D, causal)
            t_tri = bench(lambda: fa_forward(qp, kp, vp, causal=causal))
            t_nv = bench(lambda: naive_attention(qp, kp, vp, causal=causal,
                                                  fp32=False))
            best_name, best_ms = None, float('inf')
            parts = []
            for name, be in SDPA_BACKENDS:
                if avail.get(name) is None:
                    continue
                try:
                    # 注意：上下文管理器包在 bench 外面（只进一次），
                    # 否则每次迭代多付一次 backend 开关的 Python 开销
                    with sdpa_kernel([be]):
                        t = bench(lambda: F.scaled_dot_product_attention(
                            qp, kp, vp, is_causal=causal))
                except RuntimeError:
                    continue
                parts.append(f"{name} {t:.3f}")
                if t < best_ms:
                    best_name, best_ms = name, t
            ratio = best_ms / t_tri * 100.0
            verdict = '优秀' if ratio > 85 else ('合格' if ratio >= 50 else '不达标')
            rows.append(ratio)
            print(f"  T={T:5d} {'causal ' if causal else 'full   '}"
                  f"| naive {t_nv:7.3f} ms | triton {t_tri:6.3f} ms "
                  f"({fl/t_tri*1e-9:5.1f} TF) | best {best_name} {best_ms:6.3f} ms "
                  f"({fl/best_ms*1e-9:5.1f} TF) -> {ratio:5.1f}% {verdict}")
            print(f"          SDPA 明细(ms): {' | '.join(parts)}")
    overall = min(rows)
    print(f"\n  [总评] 最慢场景 {overall:.1f}% of SDPA 最优后端 -> "
          f"{'优秀 (>85%)' if overall > 85 else ('合格 (>=50%)' if overall >= 50 else '不达标')}")
    print("  📝 公平性说明: 教学版只做前向（不物化 logsumexp、无 backward），")
    print("     而 SDPA 前向要为 autograd 额外写 LSE，且小序列时内核外的")
    print("     Python 开销占比大——所以教学版可能反超；看量级，别抠个位数。")

    # ---------------------------------------------------------------
    print("\n" + "-" * 74)
    print("段 5a  FlexAttention: 同样的 causal / sliding，几行 PyTorch 搞定")
    print("-" * 74)
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    Bf, Hf, Tf, Df = 1, 4, 1024, 64
    qf = torch.randn(Bf, Hf, Tf, Df, device='cuda', dtype=torch.bfloat16)
    kf, vf = torch.randn_like(qf), torch.randn_like(qf)

    def causal_mod(b, h, q_idx, kv_idx):        # mask_mod: 返回 True = 参与注意
        return q_idx >= kv_idx

    def sliding_mod(b, h, q_idx, kv_idx, W=256):   # causal + 最近 W 个 token
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= W)

    # create_block_mask 把 Python 函数编译成分块 (128x128) 的稀疏 Bitmap
    causal_bm = create_block_mask(causal_mod, None, None, Tf, Tf)
    sliding_bm = create_block_mask(sliding_mod, None, None, Tf, Tf)

    out_flex = flex_attention(qf, kf, vf, block_mask=causal_bm)
    ref_causal = naive_attention(qf, kf, vf, causal=True)
    torch.testing.assert_close(out_flex.float(), ref_causal, rtol=1e-2, atol=1e-2)
    print(f"  [flex causal ] vs naive: assert_close PASS "
          f"(最大相对误差 {max_rel_err(out_flex, ref_causal):.2e})")

    out_flex_s = flex_attention(qf, kf, vf, block_mask=sliding_bm)
    Tl = qf.shape[2]
    # naive 参考手动加同一个 sliding mask：下三角 且 q_idx - kv_idx <= 256
    keep = (torch.ones(Tl, Tl, dtype=torch.bool, device='cuda').tril()
            & ((torch.arange(Tl, device='cuda')[:, None]
                - torch.arange(Tl, device='cuda')[None, :]) <= 256))
    att = (qf.float() @ kf.float().transpose(-2, -1)) / math.sqrt(Df)
    att = att.masked_fill(~keep, float('-inf')).softmax(-1)
    ref_slide = att @ vf.float()
    torch.testing.assert_close(out_flex_s.float(), ref_slide, rtol=1e-2, atol=1e-2)
    print(f"  [flex sliding] vs naive(W=256): assert_close PASS "
          f"(最大相对误差 {max_rel_err(out_flex_s, ref_slide):.2e})")
    print("  📝 FlexAttention = score_mod/mask_mod -> torch.compile 生成 Triton")
    print("     内核；块级 Bitmap 让 sliding 只算需要的块（等价我们的阶段 3 跳过）")

    print("\n" + "-" * 74)
    print("段 5b  SageAttention（只介绍，不实现）")
    print("-" * 74)
    print("  论文: arXiv 2411.10958 (SageAttention2)。QK^T 用 INT8、PV 用 INT4")
    print("  量化 + 平滑 outlier；4090 恰是甜点卡——RTX 40 的 INT4 Tensor Core")
    print("  吞吐极高（论文：INT8 只有 INT4 一半速度），实测量级 ~3x FA2。")
    print("  代价是近似（量化误差），推理可用、训练慎用；当前卡上装")
    print("  sageattention 包即可 pip 体验。FA3 走的是另一条路：Hopper 专属")
    print("  WGMMA/FP8（SM90 指令，4090/SM89 用不了）。")

    print("\n" + "=" * 74)
    print("Key takeaway: Flash Attention = tiling(02 章) + online softmax(本脚本)")
    print("+ 跳过全 mask 块。Triton 教学版 ~几十行就到 SDPA 的五到九成，")
    print("剩下的差距正是 FA2/FA3 手工调优的位置。")
    print("=" * 74)


if __name__ == '__main__':
    main()
