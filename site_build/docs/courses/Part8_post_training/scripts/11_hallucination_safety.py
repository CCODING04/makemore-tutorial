#!/usr/bin/env python3
"""
Part 8 - 脚本 11: 幻觉与安全对齐 —— 把"感觉模型在瞎编"变成可测量的数字
目标：三个实验 + 一个可选前沿方法演示，全部有真实输出：
  实验 A  简化语义熵 semantic entropy（SE）：
          同一问题采样 n 个答案 → 句向量余弦聚类（≥0.8 同簇）→ 聚类分布熵；
          SE 高 = 模型"自己跟自己都不一致" = 大概率瞎编。
          报告 SE 对"多数答案是否正确"的判别力（手写 rank-based AUROC）。
  实验 B  温度 sweep：T ∈ {0.0, 0.3, 0.7, 1.0} × 多数投票幻觉率 → output_hallucination.png
          （破除"低温 = 更事实"的迷思：温度对事实性的影响是任务相关的，Renze 2024）
  实验 C  校准对比：Qwen2.5-0.5B（base）vs Qwen2.5-0.5B-Instruct，同一批 20 道 3 选 1，
          选项 token 概率为置信度 → ECE（10 桶）——呼应 GPT-4 技术报告"RLHF 损害校准"。
  🌟可选  refusal direction（Arditi et al. 2024, arXiv 2406.11717）：
          diff-in-means 提取"拒绝方向" + 方向消融演示。
          ⚠️ 实验数据（refusal/normal 两组提示语）由读者自备 jsonl 文件——
          本脚本与教程均【不内嵌任何具体提示语样本】，只呈现方法本身（防御性视角）。
          数据文件缺失或模型不可用 → 打印说明后跳过，退出码保持 0。

模型：Qwen/Qwen2.5-0.5B-Instruct（生成 + 句向量）、Qwen/Qwen2.5-0.5B（base 对照）。
      本地缺失时打印下载指引并正常退出（rc=0）。

三个可复用函数（后续扩展实验/复用会用到，签名保持稳定）：
    sample_n(prompt, n, t)                同一 prompt 采样 n 个回答（t=0 贪心）
    semantic_entropy(answers, nli_model)  简化版 SE：句向量余弦聚类分布熵
                                          （原论文用 NLI 双向蕴含聚类，本课降级为余弦版）
    ece(conf, correct)                    Expected Calibration Error（10 桶）

叙事锚点（论文号已核实）：
    Farquhar et al., Nature 630, 625-630 (2024)   — semantic entropy 检测幻觉
    Manakul et al., EMNLP 2023, arXiv 2303.14451   — SelfCheckGPT（一致性采样检测）
    Renze, 2024, arXiv 2402.05201                  — 温度对解题表现影响任务相关
    OpenAI GPT-4 Technical Report, 2303.08774      — RLHF 前后校准曲线对比
    Arditi et al., 2024, arXiv 2406.11717          — 拒绝行为由单个方向介导

运行：
    MPLBACKEND=Agg python 11_hallucination_safety.py
预期耗时：GPU（0.5B, fp16）约 4-6 分钟；CPU 会自动缩小采样规模（约 15-25 分钟）。
"""

import os
import re
import sys
import json
import math

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

torch.manual_seed(1337)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CPU_MODE = not torch.cuda.is_available()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTRUCT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
BASE_MODEL = 'Qwen/Qwen2.5-0.5B'

# 采样规模：GPU 全量（论文规模 n=10）；CPU 缩小以保证可跑完
N_SAMPLES = 10 if not CPU_MODE else 5
MAX_NEW_TOKENS = 32 if not CPU_MODE else 20
# 句向量余弦聚类阈值：≥ COS_TH 同簇（论文用 NLI 蕴含，这里是降级实现）
COS_TH = 0.8

# ─── 模型缓存（进程内只加载一次）────────────────────────────
_CACHE = {}


def _load_hf(model_name, dtype=torch.float16):
    """加载 HF 因果语言模型（进程内缓存）。缺失 → 打印指引、返回 None（rc 保持 0）。"""
    if model_name in _CACHE:
        return _CACHE[model_name]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = dtype if not CPU_MODE else torch.float32
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype).to(DEVICE).eval()
        except Exception:  # 网络不可用时回退到纯本地缓存
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, local_files_only=True).to(DEVICE).eval()
        _CACHE[model_name] = (model, tok)
        return model, tok
    except Exception as e:
        print(f"  [MISS] 模型 {model_name} 不可用（{type(e).__name__}: {str(e)[:120]}）")
        print("         指引：huggingface-cli download " + model_name)
        print("         下载后重跑本脚本即可；现在跳过依赖该模型的实验。")
        return None


def _get_generator():
    """实验 A/B 共用的生成模型（Instruct 版）。"""
    if 'gen' not in _CACHE:
        loaded = _load_hf(INSTRUCT_MODEL)
        if loaded is None:
            return None
        _CACHE['gen'] = loaded
    return _CACHE['gen']


# ═══════════════════════════════════════════════════════════
# 可复用函数一：sample_n —— 同一 prompt 采样 n 个回答
# ═══════════════════════════════════════════════════════════
def sample_n(prompt, n, t):
    """对同一 prompt 采样 n 个回答（一致性采样的最小实现）。

    Args:
        prompt:  问题文本（str）
        n:       采样个数
        t:       温度。t > 0 → do_sample=True + num_return_sequences=n；
                 t == 0 → 贪心解码（同答案重复 n 次，作为对照点）
    Returns:
        list[str]，长度 n，每个元素是去掉换行后的首行回答
    """
    gen = _get_generator()
    if gen is None:
        raise RuntimeError("生成模型不可用（Qwen2.5-0.5B-Instruct 未缓存）——见 _load_hf 的下载指引")
    model, tok = gen
    messages = [
        {"role": "system", "content": "你是一个简洁的问答助手。请直接给出答案，不超过一句话，不要解释。"},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors='pt').to(DEVICE)
    input_len = enc['input_ids'].shape[1]  # (1, L_in)

    with torch.no_grad():
        if t and t > 0:
            out = model.generate(
                **enc, do_sample=True, temperature=t, top_p=0.95,
                max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=n,  # → (n, L_in+L_out)
            )
        else:
            out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
            out = out.repeat(n, 1)  # (1, L) → (n, L)：贪心结果复制 n 份

    # 只取新生成部分：(n, L_in+L_out) → 每条 decode 后取首行
    answers = []
    for row in out:
        ans = tok.decode(row[input_len:], skip_special_tokens=True)
        answers.append(ans.strip().split('\n')[0].strip())
    return answers


# ═══════════════════════════════════════════════════════════
# 可复用函数二：semantic_entropy —— 句向量余弦聚类版语义熵
# ═══════════════════════════════════════════════════════════
def embed_sentences(texts, batch_size=32):
    """纯 torch 句向量：Qwen2.5-0.5B-Instruct 最后一层隐状态 mean pooling。

    说明：因果 LM 不是句向量模型，这里只用于"相同答案聚成一簇"的粗聚类——
    教学降级（原论文用 NLI 双向蕴含判断语义等价）。

    Args:  texts: list[str]
    Returns: Tensor (n_texts, d_model=896)，已 L2 归一化
    """
    gen = _get_generator()
    if gen is None:
        raise RuntimeError("生成模型不可用（Qwen2.5-0.5B-Instruct 未缓存）——见 _load_hf 的下载指引")
    model, tok = gen
    if tok.padding_side != 'right':
        tok.padding_side = 'right'  # mean pooling 需要右侧 padding
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors='pt', padding=True).to(DEVICE)
        # input_ids: (B, T)；attention_mask: (B, T)
        with torch.no_grad():
            hs = model.model(**enc).last_hidden_state  # (B, T, d_model)
        mask = enc['attention_mask'].unsqueeze(-1).float()  # (B, T, 1)
        summed = (hs * mask).sum(dim=1)                     # (B, d_model)
        counts = mask.sum(dim=1).clamp(min=1.0)             # (B, 1)
        pooled = summed / counts                             # (B, d_model) mean pooling
        vecs.append(F.normalize(pooled, dim=-1))             # L2 归一化，方便算余弦
    return torch.cat(vecs, dim=0)  # (n_texts, d_model)


def _canonicalize(s):
    """表层规整：去标点/空白，去"大约/约/左右"等填充词。

    短答案的表层噪声（"北京。"vs"北京"、"约1642米"vs"1642米"）会淹没语义差异，
    先规整可以让"同义"至少在字符串层面对齐，再交给句向量聚类。
    """
    t = re.sub(r'[。，、；：,.;:!?！？\s"\'（）()【】\[\]“”]', '', s.strip())
    fillers = ['大约', '大概', '约', '左右', '将近', '整整']
    changed = True
    while changed:
        changed = False
        for f in fillers:
            if t.startswith(f):
                t = t[len(f):]
                changed = True
            if t.endswith(f):
                t = t[:-len(f)]
                changed = True
    return t


def _cosine_clusters(emb):
    """贪心凝聚聚类：按顺序遍历，与某簇质心余弦 ≥ COS_TH 则并入。

    ⚠️ 反各向异性（实测坑）：mean pooling 句向量挤在窄锥里——实测"北京"vs"上海"
    的原始余弦高达 0.95+，任何阈值都分不开。解法：先减去批均值（去掉公共分量）
    再归一化——相同字符串余弦恢复到 1.0，不同答案落到 ≈0 或负数，0.8 阈值即可用。

    Args:  emb: (n, d) 已 L2 归一化
    Returns: list[int] 长度 n，每个样本的簇编号
    """
    # 中心化 + 重归一化：(n, d) - (1, d) → (n, d)
    emb = emb - emb.mean(dim=0, keepdim=True)
    emb = F.normalize(emb, dim=-1)
    n = emb.shape[0]
    centroids, sizes, assign = [], [], [-1] * n
    for i in range(n):
        placed = False
        for c in range(len(centroids)):
            cos = float((emb[i] * centroids[c]).sum())  # 已归一化 → 内积即余弦
            if cos >= COS_TH:
                assign[i] = c
                sizes[c] += 1
                # 增量更新质心（保持单位长度）
                centroids[c] = F.normalize(
                    centroids[c] * (sizes[c] - 1) + emb[i], dim=0)
                placed = True
                break
        if not placed:
            assign[i] = len(centroids)
            centroids.append(emb[i].clone())
            sizes.append(1)
    return assign


def semantic_entropy(answers, nli_model):
    """简化版语义熵（Farquhar et al., Nature 2024 的课堂版）。

    原论文：采样 n 个答案 → NLI 双向蕴含聚成"语义簇" → 簇分布的熵。
    本实现的降级链（换来的：纯 torch、无需额外模型）：
        ① 表层规整 _canonicalize（去标点/填充词，消"北京。"vs"北京"噪声）
        ② 句向量编码 nli_model（mean pooling）+ 批内中心化（反各向异性，见 _cosine_clusters）
        ③ 余弦 ≥ 0.8 同簇——相同字符串恒为 1.0，不同答案实测 ≤0.3，阈值 0.8 够用

    Args:
        answers:   list[str]，同一问题的 n 个采样答案
        nli_model: 句向量编码器（本脚本传 embed_sentences）
    Returns:
        float：SE = -Σ_c (n_c/n) ln(n_c/n)，单位 nat。0 = 所有答案语义一致。
    """
    if len(answers) == 0:
        return 0.0
    cans = [_canonicalize(a) for a in answers]  # 表层规整："约1642米。" → "1642米"
    emb = nli_model(cans)                 # (n, d)
    assign = _cosine_clusters(emb)        # list[int]
    n = len(assign)
    counts = {}
    for c in assign:
        counts[c] = counts.get(c, 0) + 1
    se = 0.0
    for k in counts.values():
        p = k / n
        se -= p * math.log(p)             # 自然对数（nat）
    return se


# ═══════════════════════════════════════════════════════════
# 可复用函数三：ece —— 期望校准误差（10 桶）
# ═══════════════════════════════════════════════════════════
def ece(conf, correct, n_bins=10):
    """Expected Calibration Error：模型"自信程度"与"实际正确率"的加权偏差。

    ECE = Σ_b (n_b / N) × |acc(b) - avg_conf(b)|，b 遍历 [0,1] 的 n_bins 个等宽桶。

    Args:
        conf:    list[float]，每条预测的置信度 ∈ (0,1]
        correct: list[bool]，每条预测是否正确（与 conf 等长）
    Returns:
        float ∈ [0,1]，越小越校准
    """
    assert len(conf) == len(correct)
    N = len(conf)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i in range(N) if (conf[i] > lo or (b == 0 and conf[i] == 0)) and conf[i] <= hi]
        if not idx:
            continue
        acc_b = sum(bool(correct[i]) for i in idx) / len(idx)
        conf_b = sum(conf[i] for i in idx) / len(idx)
        total += (len(idx) / N) * abs(acc_b - conf_b)
    return total


def auroc(scores, labels):
    """手写 rank-based AUROC（Mann-Whitney U 统计量）。

    H0：分数对正类没有区分度 → AUROC ≈ 0.5；>0.5 表示分数越高越可能是正类。
    平局用平均名次处理（与小样本严格解一致）。

    Args:
        scores: list[float]；labels: list[int]（1 = 正类）
    """
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 平均名次从 1 开始
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ─── 题库：30 题（15 易幻觉 trivia + 15 稳定事实）────────────
# checker 三种类型：
#   ("num", [(lo, hi), ...])  提取全部数字（含 万/亿/万亿 后缀换算），命中任一区间即对
#   ("kw",  [子串, ...])      包含任一关键词即对（子串会 lower() 后匹配）
#   ("re",  正则)             正则命中即对
def _num_values(text):
    """从中文回答里提取数字，处理千分位逗号与 万/亿/万亿 量级后缀。"""
    t = text.replace(',', '').replace('，', '').replace('−', '-')
    vals = []
    for m in re.finditer(r'(-?\d+(?:\.\d+)?)\s*(万亿|万|亿)?', t):
        v = float(m.group(1))
        suf = m.group(2) or ''
        mult = {'万': 1e4, '亿': 1e8, '万亿': 1e12}.get(suf, 1)
        vals.append(v * mult)
    return vals


def check_answer(text, checker):
    """用 checker 判定回答是否正确。"""
    kind, payload = checker
    t = text.strip()
    if kind == 'num':
        vals = _num_values(t)
        return any(lo <= v <= hi for v in vals for lo, hi in payload)
    if kind == 'kw':
        low = t.lower()
        return any(str(k).lower() in low for k in payload)
    if kind == 're':
        return re.search(payload, t) is not None
    return False


# 15 个易幻觉题：冷知识/数字类，0.5B 小模型典型的"一本正经瞎编"重灾区
PRONE = [
    ("贝加尔湖最深处大约是多少米？", ("num", [(1542, 1742)])),
    ("科拉超深钻井（最深的人工钻井）大约钻到了多少米深？", ("num", [(11000, 13500)])),
    ("长江全长大约是多少公里？", ("num", [(6000, 6600)])),
    ("中国明代长城的长度大约是多少公里？", ("num", [(8400, 9300)])),
    ("地球与月球的平均距离大约是多少公里？", ("num", [(360000, 400000)])),
    ("马里亚纳海沟最深处大约是多少米？", ("num", [(10000, 11500)])),
    ("一光年大约等于多少公里？", ("num", [(9.0e12, 9.8e12)])),
    ("南极洲记录到的最低气温大约是多少摄氏度？", ("num", [(-95, -78)])),
    ("蜂鸟扇动翅膀的频率大约是每秒多少次？", ("num", [(20, 90)])),
    ("蓝鲸的心脏大约有多重（公斤）？", ("num", [(150, 700)])),
    ("恐龙大约在多少年前灭绝？", ("num", [(5.5e7, 7.5e7)])),
    ("现存最古老的文字系统是什么文字？", ("kw", ["楔形", "苏美尔"])),
    ("按重量算人体最强壮的肌肉是哪块肌肉？", ("kw", ["咬肌"])),
    ("维多利亚瀑布位于非洲的哪个（些）国家？", ("kw", ["赞比亚", "津巴布韦"])),
    ("太平洋的总面积大约是多少平方公里？", ("num", [(1.6e8, 2.0e8)])),
]

# 15 个稳定事实：高频常识，模型训练语料里出现过成千上万次
STABLE = [
    ("中国的首都是哪座城市？", ("kw", ["北京"])),
    ("一年有多少个月？", ("num", [(12, 12)])),
    ("水的化学式是什么？", ("re", r"[Hh][2₂]?[Oo]")),
    ("一周有几天？", ("num", [(7, 7)])),
    ("世界上最高的山峰是哪一座？", ("kw", ["珠穆朗玛", "埃佛勒斯", "everest"])),
    ("大熊猫主要以什么食物为生？", ("kw", ["竹"])),
    ("太阳每天从哪个方向升起？", ("kw", ["东"])),
    ("正常人有两只眼睛、两只耳朵，那么一只手有几根手指？", ("num", [(5, 5)])),
    ("地球绕太阳公转一圈需要多长时间？", ("re", r"一年|365|366|12\s*个?月")),
    ("在标准大气压下，水的沸点是多少摄氏度？", ("num", [(99, 101)])),
    ("中国的国土面积大约是多少万平方公里？", ("num", [(940, 980)])),
    ("世界上面积最大的大洋是哪一个？", ("kw", ["太平"])),
    ("月球绕地球公转一圈大约需要多少天？", ("num", [(27, 31)])),
    ("传统说法中彩虹有几种颜色？", ("num", [(7, 7)])),
    ("鱼用什么器官在水中呼吸？", ("kw", ["鳃"])),
]


def majority_correct(answers, checker):
    """多数投票答案是否正确：取最大语义簇，簇内过半数通过 checker 才算对。

    Returns: (bool, 簇数)
    """
    cans = [_canonicalize(a) for a in answers]
    emb = embed_sentences(cans)              # (n, d)
    assign = _cosine_clusters(emb)           # list[int]
    counts = {}
    for c in assign:
        counts[c] = counts.get(c, 0) + 1
    top = max(counts, key=counts.get)
    members = [a for a, c in zip(cans, assign) if c == top]
    ok = sum(check_answer(m, checker) for m in members)
    return (ok * 2 >= len(members)), len(counts)


# ─── 实验 C 的 20 道 3 选 1 ─────────────────────────────────
MCQ3 = [
    ("地球自转产生了什么现象？", ["昼夜交替", "四季更替", "月相变化"], 0),
    ("下列哪种动物是哺乳动物？", ["蓝鲸", "鲨鱼", "海龟"], 0),
    ("水的沸点在标准大气压下是多少摄氏度？", ["100", "90", "80"], 0),
    ("《静夜思》的作者是谁？", ["李白", "杜甫", "白居易"], 0),
    ("太阳系中体积最大的行星是？", ["木星", "地球", "火星"], 0),
    ("一公里等于多少米？", ["1000", "100", "500"], 0),
    ("下列哪个是中国的传统节日？", ["中秋节", "圣诞节", "感恩节"], 0),
    ("植物进行光合作用主要吸收什么气体？", ["二氧化碳", "氧气", "氮气"], 0),
    ("人体血液中运输氧气的细胞是？", ["红细胞", "白细胞", "血小板"], 0),
    ("光在真空中的速度大约是每秒多少公里？", ["30万", "3万", "300万"], 0),
    ("DPO 算法直接优化模型所用的数据是？", ["成对偏好数据（chosen 与 rejected）", "单条指令文本", "无标注原始语料"], 0),
    ("GRPO 与 PPO 的关键区别是？", ["用组内相对优势替代 Value Network", "不需要任何奖励信号", "只适用于图像任务"], 0),
    ("LoRA 微调的核心做法是？", ["冻结原权重，训练低秩旁路矩阵", "把全部权重量化到 4bit", "删除一半注意力头"], 0),
    ("分组量化（group size 128）的主要作用是？", ["隔离离群通道，减小精度损失", "省掉 KV Cache", "把参数量减为 1/4"], 0),
    ("RAG 缓解幻觉的机制是？", ["生成前检索外部知识拼入上下文", "把温度调为 0", "扩大 tokenizer 词表"], 0),
    ("SFT 阶段计算损失时通常对 prompt 部分？", ["做 mask，只对回答部分算损失", "加倍计算", "随机丢弃"], 0),
    ("奖励模型通常用什么损失训练？", ["Bradley-Terry 成对比较损失", "交叉熵分类损失", "均方误差回归"], 0),
    ("perplexity 对不同 tokenizer 的模型？", ["不可直接比较，应换算 bits-per-byte", "可以直接比较", "只看数值大小"], 0),
    ("KV Cache 的作用是？", ["缓存历史的 K/V，避免重复计算", "压缩模型权重", "加速磁盘读写"], 0),
    ("温度参数调低会让采样分布？", ["更尖锐、更确定", "更平坦、更随机", "完全不变"], 0),
]
# 打乱选项、随机化 gold，避免"永远选 A"的位置偏差
def _shuffle_mcq(seed=1337):
    import random
    rng = random.Random(seed)
    out = []
    for q, opts, gold in MCQ3:
        idx = list(range(len(opts)))
        rng.shuffle(idx)
        new_opts = [opts[i] for i in idx]
        out.append((q, new_opts, idx.index(gold)))
    return out


def mcq_confidence(model_name, use_chat_template):
    """跑一批 3 选 1：输出 token 概率为置信度。

    方法：把题目格式化成
        问题：...
        A. 选项一
        B. 选项二
        C. 选项三
        答案：
    取 next-token logits，只在 A/B/C 三个 token 上做受限 softmax（多选题
    loglikelihood 的单 token 简化版——lm-eval 的 multiple_choice 同思想）。

    Returns: (confs: list[float], corrects: list[bool], acc: float)
    """
    loaded = _load_hf(model_name)
    if loaded is None:
        return None
    model, tok = loaded
    letter_ids = []
    for L in 'ABC':
        ids = tok(L, add_special_tokens=False).input_ids
        assert len(ids) == 1, f"字母 {L} 不是单 token：{ids}"
        letter_ids.append(ids[0])

    confs, corrects = [], []
    for q, opts, gold in _shuffle_mcq():
        lines = [f"问题：{q}"] + [f"{L}. {o}" for L, o in zip('ABC', opts)] + ["答案："]
        raw = "\n".join(lines)
        if use_chat_template:
            msgs = [{"role": "user", "content": "以下是一道选择题，请只回答选项字母。\n\n" + raw}]
            raw = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(raw, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits[0, -1]  # (V,) 最后一个位置的下一 token 分布
        three = logits[letter_ids]               # (3,) 只看 A/B/C 三个 token
        probs = F.softmax(three, dim=0)          # (3,) 受限 softmax = 置信度
        pred = int(probs.argmax())
        confs.append(float(probs[pred]))
        corrects.append(pred == gold)
    return confs, corrects, sum(corrects) / len(corrects)


# ═══════════════════════════════════════════════════════════
# 🌟 可选段：refusal direction（Arditi et al. 2024, 2406.11717）
# ═══════════════════════════════════════════════════════════
def refusal_direction(model_path, prompts_file):
    """拒绝方向提取与消融演示（方法来自 Arditi et al. 2024；防御性视角）。

    ⚠️ 安全说明：本函数【不内嵌任何提示语样本】。实验数据由读者自备 jsonl 文件：
        每行 {"text": "<一句话文本>", "label": "refusal" 或 "normal"}
        normal 组示例类型：普通问答对（如"法国的首都是哪里？——巴黎。"）
        refusal 组示例类型：模型表达拒绝/拒答风格的语句
        具体构造方法见论文附录（论文按危害类别成对构造两类提示）。

    方法（与论文一致的三个步骤）：
        1) 单方向假说：拒绝行为集中在残差流的一个方向 r 上
        2) diff-in-means：r = mean(h_refusal) − mean(h_normal)
           （h = 各提示最后一个 token 的隐状态；论文发现它与"拒绝方向提取"
            等价且更省算力）
        3) 方向消融：把 r 从所有表征中投影掉，模型"失去"拒绝能力
           （权重级实现 = 对每个 Linear 的 W 做 W ← W − r̂ r̂ᵀ W）

    Args:
        model_path:   模型名（7B 级效果最好；0.5B 可跑通方法演示）
        prompts_file: 自备数据 jsonl 路径；缺失 → 打印说明返回 None
    Returns:
        dict 统计结果；数据/模型不可用时返回 None（调用方继续，rc=0）
    """
    if not os.path.exists(prompts_file):
        print("  [SKIP] 未找到自备数据文件：" + prompts_file)
        print("         格式：每行一个 JSON 对象 {\"text\": ..., \"label\": \"refusal\"/\"normal\"}")
        print("         normal 组放普通问答语句，refusal 组放模型拒绝风格语句（各 ≥ 8 条）")
        print("         构造方法见论文 2406.11717 附录；7B 模型可用时效果最好。")
        return None

    lines = [json.loads(l) for l in open(prompts_file, encoding='utf-8') if l.strip()]
    ref = [d['text'] for d in lines if d.get('label') == 'refusal']
    nor = [d['text'] for d in lines if d.get('label') == 'normal']
    if len(ref) < 8 or len(nor) < 8:
        print(f"  [SKIP] 数据不足：refusal={len(ref)} 条、normal={len(nor)} 条（各需 ≥ 8）")
        return None

    loaded = _load_hf(model_path)
    if loaded is None:
        return None
    model, tok = loaded

    def last_token_hs(texts):
        """每个文本最后一个 token 的中层隐状态：list[Tensor(d,)]。"""
        out = []
        for t in texts:
            enc = tok(t, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states
            mid = len(hs) // 2  # 论文：中层（如 Llama-2-70B 的 16/80 层）信息最集中
            out.append(hs[mid][0, -1].float())  # (d,)
        return torch.stack(out)  # (n, d)

    print(f"  提取隐状态：refusal {len(ref)} 条 / normal {len(nor)} 条 ...")
    H_ref, H_nor = last_token_hs(ref), last_token_hs(nor)  # (n, d)

    # ── Step 2: diff-in-means ──
    # r: (d,) = 均值差；归一化后是"拒绝方向"
    r = H_ref.mean(dim=0) - H_nor.mean(dim=0)
    r_hat = F.normalize(r, dim=0)

    # ── 判别力检查：cos(h, r̂) 对两类样本的 AUROC（用可复用的 auroc 函数）──
    cos_all = torch.cat([H_ref, H_nor]) @ r_hat              # (n_ref+n_nor,)
    labels = [1] * len(ref) + [0] * len(nor)
    a_before = auroc(cos_all.tolist(), labels)

    # ── Step 3: 方向消融（embedding 级演示）──
    # 把 r̂ 从每个表征中投影掉：h' = h − (h·r̂) r̂
    # 权重级等价操作（论文做法）见下方 orthogonalize()，对每个 Linear 执行一次即可
    H_ref_abl = H_ref - (H_ref @ r_hat).unsqueeze(1) * r_hat  # (n, d)
    H_nor_abl = H_nor - (H_nor @ r_hat).unsqueeze(1) * r_hat
    cos_abl = torch.cat([H_ref_abl, H_nor_abl]) @ r_hat       # 消融后方向上的投影
    a_after = auroc(cos_abl.tolist(), labels)

    print(f"  |r|（未归一化方向范数）: {float(r.norm()):.4f}")
    print(f"  cos(h, r̂) 的 AUROC：消融前 {a_before:.3f} → 消融后 {a_after:.3f}")
    print("  解读：消融后 AUROC 跌回 ~0.5 = 拒绝信息已被从表征中移除；")
    print("        论文进一步把它写进权重（所有 Linear 正交化），模型即'不会拒绝'——")
    print("        这既解释了拒绝行为的机理，也提示了对齐的脆弱性（故须配合评测与红队）。")
    return {'auroc_before': a_before, 'auroc_after': a_after}


def orthogonalize(W, r):
    """权重级方向消融：W ← W − r̂ r̂ᵀ W（论文的 weight orthogonalization）。

    W: (d_out, d_in)；r: (d,) 方向（残差流维度）。
    返回消融后的 W（shape 不变）。对模型每个 Linear 层执行一次，
    则前向传播中 r 方向的分量永远为 0 —— 'directional ablation'。
    """
    r_hat = F.normalize(r, dim=0)          # (d,)
    return W - torch.outer(r_hat, r_hat) @ W  # (d,d) @ (d_out,d_in) → (d_out,d_in)


# ═══════════════════════════════════════════════════════════
# 实验 A：简化语义熵
# ═══════════════════════════════════════════════════════════
def experiment_A():
    print("\n── 实验 A：简化语义熵（semantic entropy, SE）──")
    print(f"  配置：{INSTRUCT_MODEL}，do_sample=True，T=0.7，每题采样 {N_SAMPLES} 个答案")
    print(f"  聚类：句向量余弦 ≥ {COS_TH}（原论文用 NLI 双向蕴含，此处教学降级）\n")

    questions = [(q, c, 'prone') for q, c in PRONE] + [(q, c, 'stable') for q, c in STABLE]
    rows, ses, wrongs = [], [], []
    for qi, (q, checker, kind) in enumerate(questions):
        answers = sample_n(q, N_SAMPLES, t=0.7)
        se = semantic_entropy(answers, embed_sentences)
        ok, n_clusters = majority_correct(answers, checker)
        rows.append((qi, kind, q, n_clusters, se, ok, answers))
        ses.append(se)
        wrongs.append(0 if ok else 1)  # label=1 表示"多数答案错误"（幻觉）
        print(f"  [{qi:02d}] {kind:6s} 簇={n_clusters} SE={se:.3f} 多数答案{'正确' if ok else '错误'}  {q}")

    # SE 作为"幻觉检测器"的判别力：label=1（错误）的 SE 应更高
    a = auroc(ses, wrongs)
    se_ok = [s for s, w in zip(ses, wrongs) if w == 0]
    se_bad = [s for s, w in zip(ses, wrongs) if w == 1]
    n_wrong = sum(wrongs)
    print(f"\n  30 题汇总：多数答案错误 {n_wrong}/{len(questions)} 题")
    print(f"  SE 均值：多数答案正确 {sum(se_ok)/max(len(se_ok),1):.3f} nat vs 错误 {sum(se_bad)/max(len(se_bad),1):.3f} nat")
    print(f"  AUROC(SE → 预测多数答案错误) = {a:.3f}")
    print("  解读：AUROC > 0.5 ⇒ 采样一致性携带了'对错'的信号（Nature 2024 的核心结论）；")
    print("        我们的单层余弦聚类是降级实现，数值低于论文的 NLI 版是预期内的。")

    # 展示 2 个具体例子（一个稳定、一个高 SE），看真实采样长什么样
    print("\n  采样样例（真实输出）：")
    shown = {'stable': False, 'prone': False}
    for qi, kind, q, n_clusters, se, ok, answers in rows:
        if not shown[kind] and ((kind == 'stable' and se == 0) or (kind == 'prone' and se >= 0.5)):
            shown[kind] = True
            print(f"    Q[{qi:02d}] {q}  （簇={n_clusters}，SE={se:.3f}）")
            for a_i, ans in enumerate(answers[:4]):
                print(f"      采样{a_i+1}: {ans[:50]}")
            if len(answers) > 4:
                print(f"      ... 共 {len(answers)} 个")
    return rows, ses, wrongs


# ═══════════════════════════════════════════════════════════
# 实验 B：温度 sweep（T=0.7 的点直接复用实验 A 的数据）
# ═══════════════════════════════════════════════════════════
def experiment_B(rows_a):
    print("\n── 实验 B：温度 sweep —— '低温 = 更事实'是真的吗？──")
    temps = [0.0, 0.3, 0.7, 1.0]
    questions = [(q, c, 'prone') for q, c in PRONE] + [(q, c, 'stable') for q, c in STABLE]

    # halluc[T][kind] = 该类题目"多数投票答案错误"的比例
    hall = {T: {'prone': [], 'stable': []} for T in temps}
    # T=0.7 复用实验 A 的结果（同种子同配置）
    for qi, kind, q, n_clusters, se, ok, answers in rows_a:
        hall[0.7][kind].append(0 if ok else 1)

    for T in temps:
        if T == 0.7:
            continue
        for qi, (q, checker, kind) in enumerate(questions):
            answers = sample_n(q, N_SAMPLES, t=T)
            ok, _ = majority_correct(answers, checker)
            hall[T][kind].append(0 if ok else 1)

    print(f"\n  {'T':<6}{'总体幻觉率':<12}{'易幻觉题':<12}{'稳定事实题':<12}")
    print("  " + "─" * 42)
    xs, ys_all, ys_prone, ys_stable = [], [], [], []
    for T in temps:
        all_v = hall[T]['prone'] + hall[T]['stable']
        r_all = sum(all_v) / len(all_v)
        r_pro = sum(hall[T]['prone']) / max(len(hall[T]['prone']), 1)
        r_sta = sum(hall[T]['stable']) / max(len(hall[T]['stable']), 1)
        xs.append(T); ys_all.append(r_all); ys_prone.append(r_pro); ys_stable.append(r_sta)
        print(f"  {T:<6}{r_all:<12.1%}{r_pro:<12.1%}{r_sta:<12.1%}")

    print("\n  解读：如果'低温=更事实'成立，幻觉率应随 T 单调上升；实测曲线平缓/非单调，")
    print("        与 Renze 2024（2402.05201）一致：温度的影响是任务相关的——")
    print("        知识没存进权重里，把采样调'保守'也编不出正确答案。")

    # 绘图（英文标题，存到脚本同目录）
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys_prone, 'o-', label='Hallucination-prone trivia (15)')
        ax.plot(xs, ys_stable, 's-', label='Stable facts (15)')
        ax.plot(xs, ys_all, '^--', label='Overall (30)')
        ax.set_xlabel('Sampling temperature T')
        ax.set_ylabel('Majority-vote error rate')
        ax.set_title('Temperature vs Majority-Vote Hallucination Rate (Qwen2.5-0.5B-Instruct)')
        ax.set_xticks(temps)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.grid(alpha=0.3)
        ax.legend()
        out = os.path.join(SCRIPT_DIR, 'output_hallucination.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  [OK] 曲线已保存：{out}")
    except Exception as e:
        print(f"  [WARN] 绘图失败（{e}），表格结果仍然有效")


# ═══════════════════════════════════════════════════════════
# 实验 C：base vs Instruct 的 ECE 对比
# ═══════════════════════════════════════════════════════════
def experiment_C():
    print("\n── 实验 C：校准对比 —— RLHF/SFT 会损害校准吗？──")
    print("  方法：同一批 20 道 3 选 1，选项 token 的受限 softmax 概率为置信度，ECE 取 10 桶")
    print("  叙事锚点：GPT-4 技术报告（2303.08774）中 RLHF 后校准曲线明显劣化")
    print("           （GPT-4 时代结论；DPO 系新证据见 TruthRL 2509.25760）\n")

    results = {}
    for name, chat in [(BASE_MODEL, False), (INSTRUCT_MODEL, True)]:
        out = mcq_confidence(name, chat)
        if out is None:
            print("  [SKIP] 模型缺失，实验 C 跳过")
            return
        confs, corrects, acc = out
        e = ece(confs, corrects)
        avg_conf = sum(confs) / len(confs)
        results[name] = (acc, avg_conf, e, confs, corrects)

    print(f"  {'模型':<32}{'准确率':<10}{'平均置信度':<12}{'ECE(10桶)':<10}")
    print("  " + "─" * 64)
    for name, (acc, avg_conf, e, _, _) in results.items():
        tag = '（base）' if 'Instruct' not in name else '（对齐后）'
        print(f"  {name + tag:<34}{acc:<10.1%}{avg_conf:<12.1%}{e:<10.3f}")

    names = list(results)
    if len(names) == 2:
        a = results[names[0]][2]
        b = results[names[1]][2]
        direction = "Instruct 更校准" if b < a else "Instruct 校准更差"
        print(f"\n  ECE 对比：base={a:.3f} vs instruct={b:.3f} → {direction}")
        print("  解读：GPT-4 报告观察到对齐损害校准（自信但不见得更对）；我们的 0.5B 玩具复现")
        print("        方向可能一致也可能相反——小模型 + 20 题样本量，只看方法不看结论强度。")
    return results


# ─── Main ───────────────────────────────────────────────────
def main():
    print("═══ Part 8 脚本 11: 幻觉与安全对齐 ═══")
    print(f"  device={DEVICE}, 采样 n={N_SAMPLES}, max_new_tokens={MAX_NEW_TOKENS}, seed=1337")

    if _get_generator() is None:
        print("\n生成模型不可用，本次运行到此结束（rc=0）。")
        return

    rows_a, _, _ = experiment_A()
    experiment_B(rows_a)
    experiment_C()

    # 🌟 可选段：refusal direction（数据自备，缺失自动跳过）
    print("\n── 🌟 可选段：refusal direction（Arditi 2024, 2406.11717）──")
    print("  论文方法：① 单方向假说 ② diff-in-means 提取 ③ 方向消融（见 refusal_direction 文档串）")
    print("  本脚本不内嵌任何提示语样本——实验数据由读者自备（防御性视角）：\n")
    demo_file = os.path.join(SCRIPT_DIR, 'refusal_prompts.jsonl')  # 读者自备
    # 模型优先 7B（论文规模效应明显），未就绪则用 0.5B 演示方法
    refusal_direction('Qwen/Qwen2.5-7B-Instruct', demo_file)
    print("\n  [方法演示] orthogonalize(W, r)：W − r̂r̂ᵀW，对全部 Linear 执行一次即'方向消融'——")
    W = torch.randn(896, 896)
    r = torch.randn(896)
    W_abl = orthogonalize(W, r)
    proj = float((F.normalize(r, dim=0) @ W_abl).norm() / r.norm())
    print(f"  随机矩阵演示：消融后 W 在 r 方向的投影范数 ≈ {proj:.2e}（≈0 ⇒ 方向被移除；shape 全程 (d,d)@(d_out,d_in)）")

    print("""
═══ 总结 ═══

本脚本把"幻觉"和"校准"从形容词变成数字：
  A. 语义熵 SE：采样一致性检测幻觉（Nature 630, 2024）——不依赖任何外部知识库，
     只问模型"你自己一致吗"；AUROC 是它的判别力度量。
  B. 温度 sweep：幻觉率对温度不敏感（任务相关，Renze 2024）——
     "把温度调低"不是事实性方案，知识在不在权重里才是关键。
  C. ECE：base vs Instruct 的校准对比（GPT-4 报告的"对齐损害校准"）——
     自信 ≠ 正确，上线前要看校准而不只看准确率。
  🌟 refusal direction：拒绝行为集中在单一方向（diff-in-means 可提取、可消融）——
     既是机理发现，也是"对齐是脆弱的"的量化注脚（数据自备、防御性视角）。

缓解幻觉的工程正道（教程 07 章 §6 展开）：RAG 提供外部依据、弃权训练
（"我不知道"也是答案）、采样一致性做低置信过滤。""")


if __name__ == '__main__':
    main()
