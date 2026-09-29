#!/usr/bin/env python3
"""
你的 minimind —— 实验驱动生长版（教程 02-11 章配套，单文件）

设计：一个模型类 + 一组配置开关。每个现代组件（RoPE / GQA / QK-Norm / SwiGLU / RMSNorm）
都是一个开关：先跑通"没有它"的基线，发现问题，再加开关、用实验对比前后变化——
像搭积木一样，从最基础的模型逐步长成官方 minimind。

用法（每个 stage 对应一章末尾的运行点）：
  python 00_download_data.py          # 先拿官方数据 + 官方 tokenizer（一次性）
  python my_minimind.py --stage 1     # 02 基线诞生：BPE + MHA + learned PE + ReLU + LayerNorm，第一次训练
  python my_minimind.py --stage 2     # 03 实验仪器：验证集 ppl + metric 记录（CSV/tensorboard）→ 问题清单
  python my_minimind.py --stage 3     # 04 稳定提速：fp32 / fp16+GradScaler / bf16 对比 + 双卡 DDP 演示
  python my_minimind.py --stage 4     # 05/06 注意力进化（05 RoPE 外推 ｜ 06 GQA/QK-Norm）
  python my_minimind.py --stage 5     # 07 FFN 进化：ReLU→SwiGLU 同参对比、MoE 路由/aux/负载均衡扫描
  python my_minimind.py --stage 6     # 08 组装对账：现代版全开 vs 基线、25.83M 参数账本、生成
  python my_minimind.py --stage 7     # 09 阶段一 Pretrain（完整工程上阵）
  python my_minimind.py --stage 8     # 10 阶段二 SFT（加载 stage 7 权重）
  python my_minimind.py --stage 9     # 11 阶段三 DPO（训练池 + heldout 泛化监控）→ 毕业

档位：
  --profile quick   快速档（默认）：官方 26M 配置 + 数据采样，每章几分钟——先跑通，再谈复现
  --profile full    复现档：官方默认超参 + 全量数据（9/10/11 章与官方 argparse 逐项对齐，2026-09 核对）
  无 GPU 自动切 toy 档（hidden 64），只保流程。P7_STEPS 可压训练步数。

实验记录（03 章起全程启用）：
  CSV 恒定写入 ../temp/out/logs/<run>.csv（零依赖，永远可用）；
  检测到 tensorboard 则并行写事件文件：tensorboard --logdir ../temp/out/logs --port 6006
  P7_SWANLAB=1 且装有 swanlab 时同步上报（官方同款，默认关闭——不偷传云端）。
"""

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
if not torch.cuda.is_available():
    torch.set_num_threads(8)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset'))
OUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'temp', 'out'))
LOG_DIR = os.path.join(OUT_DIR, 'logs')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════════
# 公共：tokenizer 与词表报告（02 章）
# ══════════════════════════════════════════════════════════════════

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))


def vocab_report(tok):
    """02 章 §1.2：词表体检——6400 个槽位里都住了谁。"""
    print(f"  词表大小: {len(tok)}")
    print(f"  三个角色 token: bos={tok.bos_token!r}(id={tok.bos_token_id})  "
          f"eos={tok.eos_token!r}(id={tok.eos_token_id})  pad={tok.pad_token!r}(id={tok.pad_token_id})")
    print(f"  保留的特殊 token: {len(tok.added_tokens_decoder)} 个（im/vision/audio/tool_call/think…）")
    samples = ["人工智能是计算机科学的一个分支，它企图了解智能的实质。",
               "Large language models are trained on vast amounts of text data."]
    for s in samples:
        n = len(tok(s).input_ids)
        print(f"  压缩率样例: {len(s)} 字 → {n} token（{len(s)/n:.1f} 字/token）")


# ══════════════════════════════════════════════════════════════════
# 模型：一个类 + 组件开关（02 章基线 → 05-08 章逐个打开）
#   cfg 键：pos='learned'|'rope'   attn='mha'|'gqa'   qk_norm=bool
#           ffn='relu'|'swiglu'   norm='ln'|'rms'
# ══════════════════════════════════════════════════════════════════

def precompute_cos_sin(head_dim, max_seq_len, theta):
    """05 章：RoPE 角度表。第 i 维频率 ∝ θ^(-2i/d)，θ 越大低频维转得越慢。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)   # 两份拼接 → head_dim 宽
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RMSNorm(nn.Module):
    """08 章：LayerNorm 减掉均值中心化与 bias，只留均方根缩放；norm 在 fp32 里算。"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def make_norm(kind, dim, eps):
    """02 章是 LayerNorm，08 章换 RMSNorm——一个工厂函数，开关式替换。"""
    return nn.LayerNorm(dim, eps=eps) if kind == 'ln' else RMSNorm(dim, eps)


class Attention(nn.Module):
    """02 章：MHA + learned PE 的原始形态。
    05 章 pos 换 RoPE；06 章 attn 换 GQA、加 qk_norm——三个开关逐个打开，每次都做实验对比。"""
    def __init__(self, cfg):
        super().__init__()
        self.pos, self.qk_norm_on = cfg['pos'], cfg['qk_norm']
        self.n_q = cfg['n_heads']
        self.n_kv = self.n_q if cfg['attn'] == 'mha' else cfg['n_kv_heads']   # GQA：K/V 只留 2 组
        self.hd = cfg['hidden_size'] // self.n_q
        d = cfg['hidden_size']
        self.q_proj = nn.Linear(d, self.n_q * self.hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.hd, d, bias=False)
        if self.qk_norm_on:                                              # 只 norm q/k，不 norm v
            self.q_norm = RMSNorm(self.hd, cfg['eps'])
            self.k_norm = RMSNorm(self.hd, cfg['eps'])
        if self.pos == 'rope':
            cos, sin = precompute_cos_sin(self.hd, cfg['max_pos'], cfg['theta'])
            self.register_buffer('rope_cos', cos, persistent=False)      # 现算的表，不进权重文件
            self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q, self.hd)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd)
        if self.qk_norm_on:
            q, k = self.q_norm(q), self.k_norm(k)                        # 先 norm 再 RoPE（顺序不能反）
        if self.pos == 'rope':
            cos = self.rope_cos[:T].unsqueeze(0).unsqueeze(2)
            sin = self.rope_sin[:T].unsqueeze(0).unsqueeze(2)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        rep = self.n_q // self.n_kv
        if rep > 1:                                                      # repeat_kv：KV 复制给每个 Q 组
            k = k[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_q, self.hd)
            v = v[:, :, :, None, :].expand(B, T, self.n_kv, rep, self.hd).reshape(B, T, self.n_q, self.hd)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        scores = scores + torch.triu(torch.full((T, T), float('-inf'), device=x.device), 1)
        out = F.softmax(scores.float(), dim=-1).type_as(q) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.n_q * self.hd))


class FFN(nn.Module):
    """02 章：ReLU 版（中间 4d）。07 章：fffn 开关换 SwiGLU（中间 8/3·d 对齐 64）。"""
    def __init__(self, cfg):
        super().__init__()
        d = cfg['hidden_size']
        if cfg['ffn'] == 'relu':
            self.up = nn.Linear(d, 4 * d, bias=False)
            self.down = nn.Linear(4 * d, d, bias=False)
            self.forward = self._relu_forward
        else:
            i = cfg['intermediate_size']
            self.gate = nn.Linear(d, i, bias=False)
            self.up = nn.Linear(d, i, bias=False)
            self.down = nn.Linear(i, d, bias=False)
            self.forward = self._swiglu_forward

    def _relu_forward(self, x):
        return self.down(F.relu(self.up(x)))

    def _swiglu_forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """pre-norm 残差块：x + attn(norm(x)); x + ffn(norm(x))。骨架 01→09 一行不改，改的全是零件。"""
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = make_norm(cfg['norm'], cfg['hidden_size'], cfg['eps'])
        self.post_attention_layernorm = make_norm(cfg['norm'], cfg['hidden_size'], cfg['eps'])
        self.self_attn = Attention(cfg)
        self.mlp = FFN(cfg)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MiniMindForCausalLM(nn.Module):
    """你的 minimind——基线与现代版共用这一个类，差别全在 cfg 开关。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        self.pos_emb = (nn.Embedding(cfg['max_pos'], cfg['hidden_size'])
                        if cfg['pos'] == 'learned' else None)            # 02 章：学一张位置表；05 章退役
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg['num_hidden_layers']))
        self.norm = make_norm(cfg['norm'], cfg['hidden_size'], cfg['eps'])
        self.lm_head = nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)
        if cfg.get('tie', True):
            self.lm_head.weight = self.embed_tokens.weight               # 02 章：权重绑定，省一份 V×D
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        """labels[j]=input_ids[j]（该位置要被预测出的 token）；shift 只在这里做一次。"""
        x = self.embed_tokens(input_ids)
        if self.pos_emb is not None:
            T = input_ids.size(1)
            x = x + self.pos_emb(torch.arange(T, device=input_ids.device))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            x_, y_ = logits[:, :-1, :], labels[:, 1:]
            loss = F.cross_entropy(x_.reshape(-1, x_.size(-1)), y_.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=48, temperature=0.85, top_k=50, eos_id=None):
        """自回归生成（整段重算的教学版；KV Cache 版是 06 章 GQA 的伏笔）。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.cfg['max_pos']:]
            logits = self(idx)[0][:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat((input_ids, nxt), dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return input_ids

    def param_report(self):
        groups = {'embedding': [], 'pos_emb': [], 'attention': [], 'ffn': [], 'norm': []}
        for name, p in self.named_parameters():
            key = ('embedding' if 'embed_tokens' in name
                   else 'pos_emb' if 'pos_emb' in name
                   else 'attention' if 'self_attn' in name
                   else 'ffn' if 'mlp' in name else 'norm')
            groups[key].append(p.numel())
        total = sum(sum(v) for v in groups.values())
        lines = [f"    {k:<10} {sum(v)/1e6:7.3f} M  ({sum(v)/total*100:5.1f}%)"
                 for k, v in groups.items() if v]
        return total, '\n'.join(lines)


def intermediate_of(hidden):
    """07 章：SwiGLU 参数守恒 → 8/3·d 再向上对齐 64（tensor core tile）。"""
    v = int(hidden * 8 / 3)
    return 64 * ((v + 63) // 64)


def make_config(profile, modern=False, **over):
    """两代配置 + 组件开关。baseline = 02 章的最小可用模型；modern = 05-08 章逐个开开关后的完全体。"""
    if profile == 'gpu26m':
        cfg = dict(hidden_size=512, num_hidden_layers=8, n_heads=8, n_kv_heads=2,
                   vocab_size=6400, intermediate_size=intermediate_of(512),
                   eps=1e-5, theta=1e6, max_pos=1024, tie=True,
                   pos='learned', attn='mha', qk_norm=False, ffn='relu', norm='ln')
        if modern:
            cfg.update(pos='rope', attn='gqa', qk_norm=True, ffn='swiglu', norm='rms')
    else:
        cfg = dict(hidden_size=64, num_hidden_layers=2, n_heads=4, n_kv_heads=2,
                   vocab_size=6400, intermediate_size=intermediate_of(64),
                   eps=1e-5, theta=1e4, max_pos=512, tie=True,
                   pos='learned', attn='mha', qk_norm=False, ffn='relu', norm='ln')
        if modern:
            cfg.update(pos='rope', attn='gqa', qk_norm=True, ffn='swiglu', norm='rms')
    cfg.update(over)
    return cfg


def build_model(cfg, seed=1337):
    torch.manual_seed(seed)
    return MiniMindForCausalLM(cfg).to(DEVICE)


def current_profile():
    return 'gpu26m' if DEVICE == 'cuda' else 'cpu_toy'


# ══════════════════════════════════════════════════════════════════
# 实验仪器（03 章）：MetricLogger + 验证集评估
# ══════════════════════════════════════════════════════════════════

class MetricLogger:
    """先写 CSV（零依赖、永远可用），检测到 tensorboard 再并行写事件文件。
    列结构显式声明（fields），没值的位置留空——train 行与 eval 行各填各的列。
    用法：logger = MetricLogger('run') → logger.log(step, train_loss=…, lr=…) → logger.close()"""

    FIELDS = ['step', 'train_loss', 'lr', 'tok_s', 'grad_scale', 'val_loss', 'val_ppl']

    def __init__(self, run_name):
        self.run = run_name
        os.makedirs(LOG_DIR, exist_ok=True)
        self.csv_path = os.path.join(LOG_DIR, f'{run_name}.csv')
        self._csv = open(self.csv_path, 'w', encoding='utf-8')
        self._csv.write(','.join(self.FIELDS) + '\n')
        self._tb = None
        if os.environ.get('P7_NO_TB') != '1':
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(os.path.join(LOG_DIR, 'tb', run_name))
            except ImportError:
                pass
        self._swan = None
        if os.environ.get('P7_SWANLAB') == '1':                          # 官方同款可选上报，默认关
            try:
                import swanlab
                swanlab.init(project='my-minimind', experiment_name=run_name)
                self._swan = swanlab
            except ImportError:
                print("  ⚠️ P7_SWANLAB=1 但未安装 swanlab（pip install swanlab）——跳过云端记录")

    def log(self, step, **tags):
        row = {'step': step, **tags}
        self._csv.write(','.join(
            f'{row[k]:.6g}' if isinstance(row.get(k), float) else str(row.get(k, ''))
            for k in self.FIELDS) + '\n')
        if self._tb is not None:
            for k, v in tags.items():
                self._tb.add_scalar(k, v, step)
        if self._swan is not None:
            self._swan.log(tags, step=step)

    def close(self):
        self._csv.close()
        if self._tb is not None:
            self._tb.close()
        if self._swan is not None:
            self._swan.finish()

    @staticmethod
    def view_hint():
        print(f"  📈 曲线怎么看：tensorboard --logdir {os.path.relpath(LOG_DIR, SCRIPT_DIR)} --port 6006"
              f"（或直接打开 {os.path.relpath(LOG_DIR, SCRIPT_DIR)}/*.csv）")


@torch.no_grad()
def evaluate(model, data, batch=32, max_batches=None):
    """验证集 loss / ppl：评测时 model.eval() + no_grad，与训练严格隔离。"""
    model.eval()
    losses, n = [], 0
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        xb = torch.stack([x for x, _ in chunk]).to(DEVICE)
        yb = torch.stack([y for _, y in chunk]).to(DEVICE)
        _, loss = model(xb, yb)
        losses.append(loss.item())
        n += 1
        if max_batches and n >= max_batches:
            break
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)


# ══════════════════════════════════════════════════════════════════
# 数据机器（09-11 章）——官方 PretrainDataset / SFTDataset / DPODataset 逐行同款
# ══════════════════════════════════════════════════════════════════

def get_lr(current_step, total_steps, lr):
    """09 章：官方调度——无 warmup，1.0×→0.1× 余弦（三个读数 1.0 / 0.55 / 0.1）。"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def find_data(name, sample_first=True):
    cands = []
    stem = name.replace('.jsonl', '')
    if sample_first:
        cands.append(os.path.join(DATA_DIR, stem + '_sample.jsonl'))
    cands.append(os.path.join(DATA_DIR, name))
    for p in cands:
        if os.path.exists(p):
            return p
    raise SystemExit(f"未找到 {name}——先跑 python 00_download_data.py")


def build_pretrain_dataset(tok, seq, max_samples):
    """09 章：四步流水线——①截断到 seq-2 ②手工包 bos/eos ③右补 pad ④labels 仅 pad 置 -100。"""
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    data, truncated = [], 0
    with open(find_data('pretrain_t2t_mini.jsonl'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                text = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tok(text, add_special_tokens=False, max_length=seq - 2, truncation=True).input_ids
            truncated += (len(ids) == seq - 2)
            ids = [bos] + ids + [eos]
            ids += [pad] * (seq - len(ids))
            labels = [t if t != pad else -100 for t in ids]
            data.append((torch.tensor(ids), torch.tensor(labels)))
    print(f"  数据: {len(data):,} 条 × seq={seq}（截断 {truncated:,} 条 = {truncated/len(data)*100:.0f}%；"
          f"bos={bos} eos={eos} pad={pad}）")
    return data


def split_train_val(data, val_ratio=0.05):
    """03 章：训练前先切出验证集（取尾部 5%，切分必须发生在训练开始之前）。"""
    n_val = max(1, int(len(data) * val_ratio))
    return data[:-n_val], data[-n_val:]


def build_extrap_eval(tok, seq, n=150):
    """05 章：外推评估集——专挑比训练长度长的文本（token 数 341~510），pack 到 seq=512。"""
    bos, eos, pad = tok.bos_token_id, tok.eos_token_id, tok.pad_token_id
    data = []
    with open(find_data('pretrain_t2t_mini.jsonl'), encoding='utf-8') as f:
        for line in f:
            if len(data) >= n:
                break
            try:
                text = json.loads(line)['text']
            except (json.JSONDecodeError, KeyError):
                continue
            ids = tok(text, add_special_tokens=False).input_ids
            if not (341 <= len(ids) <= seq - 2):
                continue
            ids = [bos] + ids[:seq - 2] + [eos]
            ids += [pad] * (seq - len(ids))
            labels = [t if t != pad else -100 for t in ids]
            data.append((torch.tensor(ids), torch.tensor(labels)))
    print(f"  外推评估集: {len(data)} 条 × seq={seq}（全部长于训练长度 340）")
    return data


SYSTEM_PROMPTS = [
    "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
    "你是minimind，一个小巧但有用的语言模型。",
    "你是一个专业的AI助手，请提供有价值的回答。",
    "你是minimind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    "You are a helpful AI assistant.",
    "You are minimind, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are minimind, a small but useful language model."
]


def pre_processing_chat(conversations, rng, ratio=0.2):
    """10 章：20% 概率注入随机 system prompt——训练分布覆盖推理分布。"""
    if any(c.get('tools') for c in conversations):
        return conversations
    if conversations[0].get('role') != 'system' and rng.random() < ratio:
        return [{'role': 'system', 'content': rng.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(text, rng, keep_empty_think=0.2):
    """10 章：80% 概率删掉空 <think>\\n\\n</think>——两种格式都见过，推理兼容。"""
    if '<think>\n\n</think>\n\n' in text and rng.random() > keep_empty_think:
        return text.replace('<think>\n\n</think>\n\n', '')
    return text


def generate_labels(input_ids, bos_id, eos_id, max_length):
    """10 章：token 子序列扫描——滑窗找 <|im_start|>assistant\\n，圈到 <|im_end|>\\n，区间外 -100。"""
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1
            for j in range(start, min(end + len(eos_id), max_length)):
                labels[j] = input_ids[j]
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


def build_sft_dataset(tok, seq, max_samples):
    rng = random.Random(42)
    bos_id = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    eos_id = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids
    pad = tok.pad_token_id
    data, ratios = [], []
    with open(find_data('sft_t2t_mini.jsonl'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                conv = json.loads(line)['conversations']
            except (json.JSONDecodeError, KeyError):
                continue
            conv = pre_processing_chat(conv, rng)
            prompt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            prompt = post_processing_chat(prompt, rng)
            ids = tok(prompt).input_ids[:seq]
            ids += [pad] * (seq - len(ids))
            labels = generate_labels(ids, bos_id, eos_id, seq)
            sup = sum(1 for l in labels if l != -100)
            if sup:
                data.append((torch.tensor(ids), torch.tensor(labels)))
                ratios.append(sup / len(labels))
    print(f"  数据: {len(data):,} 条 × seq={seq}，有效监督占比 ≈{sum(ratios)/len(ratios)*100:.0f}%")
    return data


def chat_input(tok, question):
    text = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    return tok(text).input_ids


def chat_output(tok, ids, prompt_len):
    out = ids[0, prompt_len:].tolist()
    if tok.eos_token_id in out:
        out = out[:out.index(tok.eos_token_id)]
    text = tok.decode(out)
    if '</think>' in text:
        text = text.split('</think>')[-1]
    return text


def build_dpo_dataset(tok, seq, max_samples):
    """11 章：chosen/rejected 各渲染成定长，预切分 (x, y, mask) 三根张量。"""
    bos_id = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    eos_id = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids
    data = []
    with open(find_data('dpo.jsonl'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                sample = json.loads(line)
                pair = []
                for conv in (sample['chosen'], sample['rejected']):
                    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                    ids = tok(text, truncation=True, max_length=seq, padding='max_length').input_ids
                    mask = generate_labels(ids, bos_id, eos_id, seq)
                    pair.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:]),
                                 torch.tensor(mask[1:], dtype=torch.float)))
                data.append(tuple(pair))
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  数据: {len(data):,} 对偏好 × seq={seq}")
    return data


# ══════════════════════════════════════════════════════════════════
# 统一训练引擎（02/04 章逐件讲，09 章全件上阵）
# ══════════════════════════════════════════════════════════════════

def amp_ctx(amp):
    """04 章：AMP 上下文。fp16 需配 GradScaler；bf16 指数位同 fp32 不需要；'off' 即 fp32。"""
    if DEVICE != 'cuda' or amp == 'off':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if amp == 'fp16' else torch.bfloat16)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr)                  # 官方同款默认 betas/wd


def train_model(model, prof, run_name, train_data, val_data=None,
                amp='bf16', seed=1337, quiet=False):
    """03 章的完整训练循环（04 章逐件拆解）：shuffle → 累积 → AMP → 裁剪 → step；
    每 eval_every 步验证集 ppl；每 log_every 步 metric 入 CSV/tensorboard。"""
    torch.manual_seed(seed)
    steps = prof['steps']
    if 'epochs' in prof:                                                 # full 档：步数 = epoch × 样本 / 有效 batch
        steps = prof['epochs'] * len(train_data) // (prof['batch_size'] * prof['accum'])
        prof = dict(prof, steps=steps)
        if not quiet:
            print(f"  full 档: {prof['epochs']} epoch × {len(train_data):,} 条 ÷ 有效 batch "
                  f"{prof['batch_size']}×{prof['accum']} → {steps:,} 步")
    lr = prof['lr'] if DEVICE == 'cuda' else prof.get('lr_cpu', prof['lr'])
    bs, accum = prof['batch_size'], prof.get('accum', 1)
    eval_every, log_every = prof.get('eval_every', 0), prof.get('log_every', 20)
    optimizer = make_optimizer(model, lr)
    scaler = make_scaler(enabled=(amp == 'fp16' and DEVICE == 'cuda'))
    logger = MetricLogger(run_name)
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    def stream():
        while True:
            order = torch.randperm(len(train_data)).tolist()
            for i in order:
                yield train_data[i]
    gen = stream()
    t0, losses = time.time(), []
    if not quiet:
        amp_tag = {'off': 'fp32', 'fp16': 'fp16+GradScaler', 'bf16': 'bf16'}[amp if DEVICE == 'cuda' else 'off']
        print(f"  训练: steps={steps} lr={lr} batch={bs}×accum{accum} amp={amp_tag} "
              f"clip=1.0 {'(无 warmup 余弦→0.1×)' if prof.get('sched', True) else ''}")
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(accum):
            xs, ys = zip(*[next(gen) for _ in range(bs)])
            xb = torch.stack(xs).to(DEVICE)
            yb = torch.stack(ys).to(DEVICE)
            with amp_ctx(amp):
                _, loss = model(xb, yb)
                loss = loss / accum
            scaler.scale(loss).backward()                                # fp16: 放大 loss 护梯度；bf16/fp32: 直通
            accum_loss += loss.item()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)                                   # 裁剪前先还原真实梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if prof.get('sched', True):
            for g in optimizer.param_groups:
                g['lr'] = get_lr(step, steps, lr)
        scaler.step(optimizer)
        scaler.update()
        losses.append(accum_loss)
        tok_s = bs * prof['seq'] * accum / max((time.time() - t0) / (step + 1), 1e-9)
        tags = dict(train_loss=accum_loss, lr=optimizer.param_groups[0]['lr'], tok_s=tok_s)
        if scaler.is_enabled() and step % log_every == 0:
            tags['grad_scale'] = scaler.get_scale()                      # 04 章：看这根"保险丝"读数
        logger.log(step, **tags)
        if not quiet and (step % log_every == 0 or step == steps - 1):
            extra = f" scale {scaler.get_scale():.0f}" if scaler.is_enabled() else ""
            print(f"  step {step:4d}/{steps}: loss {accum_loss:.4f} "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} tok/s {tok_s:.0f}{extra}")
        if eval_every and val_data and (step % eval_every == 0 or step == steps - 1):
            vl, vp = evaluate(model, val_data)
            logger.log(step, val_loss=vl, val_ppl=vp)
            if not quiet:
                print(f"    ↳ 验证集: loss {vl:.4f}  ppl {vp:.2f}")
    csv_rel = os.path.relpath(logger.csv_path, SCRIPT_DIR)
    logger.close()
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else 0
    final = sum(losses[-5:]) / 5
    if not quiet:
        print(f"  📉 loss {losses[0]:.4f} → {final:.4f} ｜ 峰值显存 {peak:.1f} GB ｜ 日志 {csv_rel}")
    return final, peak


def save_model(model, cfg, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{name}_{cfg['hidden_size']}.pth")
    torch.save({'model': {k: v.half().cpu() for k, v in model.state_dict().items()},
                'config': cfg}, out)
    print(f"  ✅ 权重 → {out}")


def load_weights(model, name):
    p = os.path.join(OUT_DIR, f"{name}_{model.cfg['hidden_size']}.pth")
    if not os.path.exists(p):
        print(f"  ⚠️ 未找到 {os.path.basename(p)}（先跑上一个 stage），本次随机初始化——流程仍可看")
        return False
    ckpt = torch.load(p, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    return True


# ══════════════════════════════════════════════════════════════════
# 04 章素材：AMP 三连对比 + 双卡 DDP/DistributedSampler 演示
# ══════════════════════════════════════════════════════════════════

def amp_shootout(cfg, prof):
    """04 章 §3.1：同一模型、同一数据、同一 seed，只换数值格式——fp32 / fp16+GradScaler / bf16。"""
    prof = dict(prof, sched=False, log_every=40)
    tok = load_tokenizer()
    data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
    train_data, val_data = split_train_val(data)
    results = {}
    for amp in ('off', 'fp16', 'bf16'):
        tag = {'off': 'fp32', 'fp16': 'fp16+scaler', 'bf16': 'bf16'}[amp]
        model = build_model(cfg)
        print(f"  ── {tag} ──")
        loss, peak = train_model(model, prof, f's3_amp_{tag}', train_data, val_data,
                                 amp=amp, quiet=True)
        scale_note = ''
        results[tag] = (loss, peak)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print("  ────────────────────────────────")
    print("  格式            末段 loss   峰值显存")
    for tag, (loss, peak) in results.items():
        print(f"  {tag:<14}  {loss:.4f}    {peak:.1f} GB")
    print("  ↳ 精度换速度：loss 三者同水平（bf16 指数位=fp32 所以不需要 scaler；fp16 的 GradScaler")
    print("    是保险丝——本实验没触发溢出，scale 恒 65536；深网络/大 lr 才见它出手）")


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def ddp_demo(cfg, prof):
    """04 章 §3.4：torchrun 双卡演示——DistributedSampler 为什么是 DDP 标配。"""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    dev = f'cuda:{local}'

    tok = load_tokenizer()
    data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
    dataset = ListDataset(data)

    model = MiniMindForCausalLM(cfg).to(dev)
    ddp_model = DDP(model, device_ids=[local])
    optimizer = make_optimizer(ddp_model, prof['lr'])
    scaler = make_scaler(enabled=False)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                 shuffle=True, drop_last=True)         # 三个关键参数，正文逐个讲
    bs = prof['batch_size']
    loader = DataLoader(dataset, batch_size=bs, sampler=sampler)
    first_indices = list(iter(sampler))[:3]                            # 本 rank 在这个 epoch 的前 3 个样本
    print(f"  [rank {rank}] epoch0 前 3 个样本下标: {first_indices}", flush=True)
    if rank == 0 and world > 1:
        other = list(iter(DistributedSampler(dataset, num_replicas=world, rank=1,
                                             shuffle=True, drop_last=True)))[:3]
        print(f"  [rank 0 视角] rank1 的前 3 个: {other} → 两张卡看到不同数据，各训一半", flush=True)
    sampler.set_epoch(1)                                               # 不调它：下个 epoch 顺序与上个完全相同
    print(f"  [rank {rank}] set_epoch(1) 后前 3 个: {list(iter(sampler))[:3]}", flush=True)

    steps, t0, seen = 0, time.time(), 0
    ddp_model.train()
    done = False
    while not done:
        sampler.set_epoch(steps)
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = ddp_model(xb, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            optimizer.step()
            seen += xb.size(0)
            steps += 1
            if steps >= prof['steps']:
                done = True
                break
    dt = time.time() - t0
    print(f"  [rank {rank}] {steps} 步 × batch {bs} ｜ {dt:.1f}s ｜ 本卡 tok/s {seen*prof['seq']/dt:.0f}",
          flush=True)
    dist.barrier()
    if rank == 0:
        print(f"  ↳ 注意：双卡合计 tok/s 并没有 ×2——26M 小模型的梯度 all-reduce 通信开销"
              f"占比很大，模型越大 DDP 越划算（04 章正文展开这笔账）")
        print("  ↳ loss 各 rank 各算一半、反向时梯度 all-reduce 求平均——等效 batch 翻倍")
    dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════
# 04/06 章素材：RoPE 外推对比 / GQA 对比 / QK-Norm 实验
# ══════════════════════════════════════════════════════════════════

def rope_vs_learned(prof):
    """05 章实验 ①：learned PE vs RoPE。同 seed 各训一次，比「训练长度内 ppl」和「外推 ppl」。"""
    tok = load_tokenizer()
    data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
    train_data, val_data = split_train_val(data)
    ext_data = build_extrap_eval(tok, seq=512)
    base = make_config(current_profile(), ffn='relu', norm='ln', attn='mha')
    prof = dict(prof, log_every=100)
    results = {}
    for pos in ('learned', 'rope'):
        cfg = dict(base, pos=pos)
        model = build_model(cfg)
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ── pos={pos}（{n_param:.2f}M）──")
        train_model(model, prof, f's4_{pos}', train_data, None, amp='bf16', quiet=True)
        _, ppl340 = evaluate(model, val_data)
        _, ppl512 = evaluate(model, ext_data)
        results[pos] = (ppl340, ppl512)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print("  ────────────────────────────────────────")
    print("  位置编码       ppl@340(训练长内)   ppl@512(外推)")
    for pos, (a, b) in results.items():
        print(f"  {pos:<12}    {a:8.2f}          {b:8.2f}")
    la, lb = results['learned']
    ra, rb = results['rope']
    print(f"  ↳ 外推变化: learned {(lb/la-1)*100:+.1f}%、rope {(rb/ra-1)*100:+.1f}%——"
          f"learned 退化、RoPE 纹丝不动")
    print("  ↳ learned 为什么只「钝痛」不「猝死」：未训练的位置行≈初始化噪声(0.02)，")
    print("    远处 token 丢失位置信息但不被投毒；真正的硬上限是表长本身——")
    print("    max_pos 之外连算都不能算，RoPE 的角度表却能现算到 32768（05 章正文）")
    print("  ↳ 本轮 rope@340 也更低（单次 300 步含随机波动）；但外推行为差异是稳定的")


def gqa_vs_mha(prof):
    """06 章实验 ②：MHA vs GQA。短训比 loss（质量），账本比参数与 KV Cache（推理成本）。"""
    cfg_mha = make_config(current_profile(), modern=False, pos='rope')
    cfg_gqa = dict(cfg_mha, attn='gqa')
    d, n_q, n_kv, hd = cfg_mha['hidden_size'], cfg_mha['n_heads'], 2, cfg_mha['hidden_size'] // cfg_mha['n_heads']
    wq = d * d
    kv_mha = 2 * d * d
    kv_gqa = 2 * d * (n_kv * hd)
    seq = 2048
    print(f"  单层 K+V 投影参数:  MHA {kv_mha/1e6:.2f}M → GQA {kv_gqa/1e6:.2f}M（÷{n_q//n_kv}）")
    print(f"  KV Cache(seq={seq}, fp16): MHA {2*seq*n_q*hd*2/1e6:.1f}MB/层 → GQA {2*seq*n_kv*hd*2/1e6:.1f}MB/层")
    tok = load_tokenizer()
    data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
    train_data, val_data = split_train_val(data)
    prof = dict(prof, log_every=100, sched=False)
    res = {}
    for tag, cfg in (('mha', cfg_mha), ('gqa', cfg_gqa)):
        model = build_model(cfg)
        loss, _ = train_model(model, prof, f's4_{tag}', train_data, val_data, amp='bf16', quiet=True)
        _, ppl = evaluate(model, val_data)
        res[tag] = ppl
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print(f"  短训后验证集 ppl: MHA {res['mha']:.2f} vs GQA {res['gqa']:.2f} —— "
          f"tiny 规模质量损失几乎为零，收益全在推理（GQA 论文结论一致）")


def qk_norm_experiment():
    """06 章实验 ③：模拟「范数漂移 3 倍」的 q/k，看注意力打分尺度。"""
    hd, B, T = 64, 2, 16
    torch.manual_seed(0)
    q = torch.randn(B, T, hd) * 3.0
    k = torch.randn(B, T, hd) * 3.0
    raw = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    qn = q / (q.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    kn = k / (k.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    normed = (qn @ kn.transpose(-1, -2)) / math.sqrt(hd)
    print(f"  未归一化 logits: std={raw.std():.2f}（softmax 输入这么大≈one-hot，梯度消失）")
    print(f"  q/k RMSNorm 后:  std={normed.std():.2f}（尺度钉死——先 norm 再 RoPE，v 不 norm）")


# ══════════════════════════════════════════════════════════════════
# 07 章素材：ReLU vs SwiGLU 同参对比 / MoE 路由与 aux
# ══════════════════════════════════════════════════════════════════

def relu_vs_swiglu(prof):
    """07 章实验 ①：同参数预算下 ReLU(4d) vs SwiGLU(8/3·d)——对照 Shazeer 2020 的结论。"""
    tok = load_tokenizer()
    data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
    train_data, val_data = split_train_val(data)
    base = make_config(current_profile(), modern=True, ffn='relu')       # 现代底座，只换 FFN
    prof = dict(prof, log_every=100)
    res = {}
    for ffn in ('relu', 'swiglu'):
        cfg = dict(base, ffn=ffn)
        model = build_model(cfg)
        n_ffn = sum(p.numel() for n, p in model.named_parameters() if '.mlp.' in n) / 1e6
        i_mid = 4 * cfg['hidden_size'] if ffn == 'relu' else cfg['intermediate_size']
        print(f"  ── FFN={ffn}（中间 {i_mid} = {i_mid/cfg['hidden_size']:.2f}d，每层 {n_ffn/8:.2f}M）──")
        loss, _ = train_model(model, prof, f's5_{ffn}', train_data, val_data, amp='bf16', quiet=True)
        _, ppl = evaluate(model, val_data)
        res[ffn] = (n_ffn, loss, ppl)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
    print("  ────────────────────────────────────────")
    print("  FFN       每层参数   末段 loss   验证集 ppl")
    for ffn, (n, l, p) in res.items():
        print(f"  {ffn:<8}  {n:6.2f}M    {l:.4f}    {p:8.2f}")
    rl, sl = res['relu'][2], res['swiglu'][2]
    print(f"  ↳ 同参数预算（每层 2.10M vs 2.16M）换「形状」：ppl {rl:.0f} → {sl:.0f}"
          f"（{(sl/rl-1)*100:+.0f}%）——门控胜出，方向与 GLU Variants 论文一致；")
    print("    论文中的增益要充分训练才完全显现（读论文实验必修课：注意规模差距）")


class MoEExperiment(nn.Module):
    """07 章：最小 MoE——router 打分 → top-1 选专家 → 加权输出（含 aux loss 与直通梯度）。"""
    def __init__(self, d=64, n_expert=4, n_expert_per_tok=1, coef=5e-4):
        super().__init__()
        self.gate = nn.Linear(d, n_expert, bias=False)
        self.experts = nn.ModuleList(nn.Sequential(
            nn.Linear(d, 4 * d, bias=False), nn.SiLU(), nn.Linear(4 * d, d, bias=False))
            for _ in range(n_expert))
        self.k, self.coef, self.n = n_expert_per_tok, coef, n_expert
        self.aux_loss = torch.zeros(())

    def forward(self, x):
        B, T, D = x.shape
        xf = x.view(-1, D)
        scores = F.softmax(self.gate(xf), dim=-1)
        top_w, top_i = torch.topk(scores, self.k, dim=-1)
        top_w = top_w - top_w.detach() + 1.0                             # 直通梯度：前向=1，反向直通 router
        top_w = top_w / (top_w.sum(-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            sel = (top_i == e)
            if sel.any():
                w = (top_w * sel).sum(-1, keepdim=True)
                y = y + expert(xf) * w
        load = F.one_hot(top_i, self.n).float().mean(0).sum(0)
        self.aux_loss = (load * scores.mean(0)).sum() * self.n * self.coef
        return y.view(B, T, D)


def moe_experiment():
    torch.manual_seed(0)
    d, tokens = 64, 256
    moe = MoEExperiment(d)
    x = torch.randn(1, tokens, d)
    y = moe(x)
    route = moe.gate(x.view(-1, d)).argmax(-1)
    share = torch.bincount(route, minlength=4).float() / route.numel()
    print(f"  256 个 token 的路由分布: {[f'{s:.0%}' for s in share.tolist()]}"
          f"（随机初始化≈均匀；训练中「一家独大」才要 aux 出手）")
    print(f"  aux_loss = N·Σf_i·P_i·α = {moe.aux_loss.item():.4f}")
    print(f"  MoE 的账：容量 ×N、单个 token 算力不变——minimind-3-moe 198M 总参只激活 64M")

# ── 07 章实验②b：负载均衡 α 扫描（从教学版 10 号脚本并入）──────────────

def _gini(f):
    """基尼系数：0=绝对均匀。Σ|f_i-f_j| / (2·N²·mean(f))"""
    n = f.numel()
    diff = (f.view(1, -1) - f.view(-1, 1)).abs().sum()
    return (diff / (2 * n * n * f.mean())).item()


def _make_cluster_data(hidden, batch):
    """4 簇输入 → 天然存在"某些专家更好"的诱惑，路由塌缩才有土壤（真实语料同理）。"""
    centers = torch.randn(4, hidden) * 3
    assign = torch.randint(0, 4, (batch,))
    x = centers[assign] + 0.3 * torch.randn(batch, hidden)
    w = torch.randn(4, hidden, hidden) / (hidden ** 0.5)
    target = torch.einsum('bd,bdh->bh', x - centers[assign], w[assign]) + centers[assign] * 0.1
    return x, target


def moe_load_balance(hidden=128, n_experts=8, steps=400, batch=256):
    """α = 0 / 0.01(Switch 推荐) / 5e-4(minimind 默认) 三档：看 aux loss 把负载从
    "一家独大"拉向均匀，代价是任务 loss 略升。"""
    class _Expert(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                     nn.Linear(2 * dim, dim))

        def forward(self, x):
            return self.net(x)

    class _MoE(nn.Module):
        def __init__(self, dim, n, aux_coef):
            super().__init__()
            self.router = nn.Linear(dim, n, bias=False)
            self.experts = nn.ModuleList([_Expert(dim) for _ in range(n)])
            self.n, self.aux_coef, self.last_f = n, aux_coef, None

        def forward(self, x, target):
            logits = self.router(x)
            probs = F.softmax(logits, dim=-1)
            idx = probs.argmax(-1)                      # top-1 路由
            f = F.one_hot(idx, self.n).float().mean(0)
            self.last_f = f.detach()
            y = torch.zeros_like(x)
            for e in range(self.n):
                m = idx == e
                if m.any():
                    y[m] = self.experts[e](x[m])
            loss = F.mse_loss(y, target)
            if self.aux_coef > 0:                       # L_aux = α·N·Σ(f_i·P_i)
                aux = self.n * (f * probs.mean(0)).sum()
                loss = loss + self.aux_coef * aux
            return loss

    print(f"  α 扫描: experts={n_experts}, tokens/step={batch}, steps={steps}"
          f"（均匀基线 f_i=1/{n_experts}={1/n_experts:.3f}, gini=0）")
    x0, y0 = _make_cluster_data(hidden, batch)
    x0, y0 = x0.to(DEVICE), y0.to(DEVICE)
    print(f"  {'α (aux系数)':<12}{'gini ↓':<10}{'max/mean ↓':<12}{'任务 loss':<12}专家负载 f_i")
    print("  " + "─" * 80)
    for alpha in (0.0, 0.01, 5e-4):
        torch.manual_seed(0)                            # 三档同初始化，公平对比
        moe = _MoE(hidden, n_experts, alpha).to(DEVICE)
        opt = torch.optim.AdamW(moe.parameters(), lr=3e-3)
        for _ in range(steps):
            loss = moe(x0, y0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        f = moe.last_f
        print(f"  {alpha:<12.4g}{_gini(f):<10.3f}{(f.max() * n_experts).item():<12.2f}"
              f"{loss.item():<12.4f}{[f'{v:.2f}' for v in f.tolist()]}")
    print("  ↳ α=0 贫富分化最大（rich-get-richer）；α>0 拉平负载、任务 loss 略升；"
          "P_i 可微、f_i 不可微——aux 公式两个都要有")



# ══════════════════════════════════════════════════════════════════
# 09-11 章素材：行为验收
# ══════════════════════════════════════════════════════════════════

def demo_continuation(model, tok):
    model.eval()
    prompt = "如何才能摆脱拖延症？"
    ids = torch.tensor([[tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids],
                       device=DEVICE)
    gen = model.generate(ids, max_new_tokens=40, eos_id=tok.eos_token_id)
    print(f"  prompt: {prompt!r}")
    print(f"  续写  : {tok.decode(gen[0].tolist()[1:])!r}")
    print("  ↳ 它在『续写文本』而不是『回答问题』——base model 的宿命，10 章 SFT 来修")


def demo_chat(model, tok):
    model.eval()
    for q in ["你是谁？", "怎么才能坚持跑步？"]:
        ids = torch.tensor([chat_input(tok, q)], device=DEVICE)
        gen = model.generate(ids, max_new_tokens=64, eos_id=tok.eos_token_id)
        print(f"  Q: {q}")
        print(f"  A: {chat_output(tok, gen, ids.shape[1])!r}")


# ══════════════════════════════════════════════════════════════════
# 11 章素材：DPO（sum 口径 + 双模型 + 泛化监控）
# ══════════════════════════════════════════════════════════════════

def logits_to_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """11 章：①sum 口径（隐式奖励∝长度）②拼批 ③无 SFT 正则。"""
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    half = ref_log_probs.shape[0] // 2
    pi = policy_log_probs[:half] - policy_log_probs[half:]
    rf = ref_log_probs[:half] - ref_log_probs[half:]
    logits = pi - rf
    return -F.logsigmoid(beta * logits).mean(), logits.detach()


def pair_scores(model, pairs, device):
    model.eval()
    cs, rs = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), 8):
            chunk = pairs[i:i + 8]
            xc = torch.stack([p[0][0] for p in chunk]).to(device)
            yc = torch.stack([p[0][1] for p in chunk]).to(device)
            mc = torch.stack([p[0][2] for p in chunk]).to(device)
            xr = torch.stack([p[1][0] for p in chunk]).to(device)
            yr = torch.stack([p[1][1] for p in chunk]).to(device)
            mr = torch.stack([p[1][2] for p in chunk]).to(device)
            cs += (logits_to_log_probs(model(xc)[0], yc) * mc).sum(dim=1).tolist()
            rs += (logits_to_log_probs(model(xr)[0], yr) * mr).sum(dim=1).tolist()
    model.train()
    return cs, rs


def dpo_report(tag, pairs, ref_scores, after):
    """11 章：绝对序 acc（模型自己的偏好）与隐式偏好 acc（相对 ref 拉开了多少）。"""
    pc, pr = pair_scores_eval(pairs, ref_scores)
    rc, rr = ref_scores
    abs_acc = sum(c > r for c, r in zip(pc, pr)) / len(pc) * 100
    if not after:
        print(f"  {tag}: policy=ref，绝对序 acc {abs_acc:.0f}%")
        return abs_acc
    implicit = [(c - r) - (a - b) for c, r, a, b in zip(pc, pr, rc, rr)]
    imp_acc = sum(d > 0 for d in implicit) / len(implicit) * 100
    print(f"  {tag}: 隐式偏好 acc {abs_acc:.0f}%→{imp_acc:.0f}%（隐式奖励 margin "
          f"{sum(implicit)/len(implicit):+.1f}）")
    return imp_acc


def pair_scores_eval(pairs, ref_scores):
    """训练后重新打分（policy 已更新）。"""
    model = dpo_report.current_model
    pc, pr = pair_scores(model, pairs, DEVICE)
    return pc, pr


def train_stage9(model, cfg, tok, prof):
    """11 章：唯一有两个模型的阶段——policy 在训，ref 是 10 章自己的冻结复印件。"""
    lr = prof['lr'] if DEVICE == 'cuda' else prof.get('lr_cpu', prof['lr'])
    steps = prof['steps']
    data = build_dpo_dataset(tok, prof['seq'], prof['max_samples'])
    pool = data[:prof['pool']]
    heldout = data[prof['pool']:prof['pool'] * 2]                       # 从没参与训练——泛化的考官

    ref = MiniMindForCausalLM(cfg).to(DEVICE)
    ref.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    ref.eval()
    ref.requires_grad_(False)

    ref_pool = pair_scores(ref, pool, DEVICE)
    ref_heldout = pair_scores(ref, heldout, DEVICE)
    optimizer = make_optimizer(model, lr)
    print(f"  训练: steps={steps} lr={lr} β={prof['beta']}（quick 放大档；官方 full=4e-8/β0.15/1 epoch）")
    dpo_report.current_model = model
    model.train()
    print("  训练前（policy=ref）:")
    dpo_report(f"  训练池 {len(pool)} 对", pool, ref_pool, after=False)
    dpo_report(f"  heldout {len(heldout)} 对", heldout, ref_heldout, after=False)

    order, pos, t0, losses = torch.randperm(len(pool)).tolist(), 0, time.time(), []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pairs = [pool[order[(pos + k) % len(order)]] for k in range(prof['batch_size'])]
        pos += prof['batch_size']
        x = torch.cat([torch.stack([p[0][0] for p in pairs]),
                       torch.stack([p[1][0] for p in pairs])]).to(DEVICE)
        y = torch.cat([torch.stack([p[0][1] for p in pairs]),
                       torch.stack([p[1][1] for p in pairs])]).to(DEVICE)
        m = torch.cat([torch.stack([p[0][2] for p in pairs]),
                       torch.stack([p[1][2] for p in pairs])]).to(DEVICE)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
            with torch.no_grad():
                ref_lp = logits_to_log_probs(ref(x)[0], y)
            pol_lp = logits_to_log_probs(model(x)[0], y)
            loss, _ = dpo_loss(ref_lp, pol_lp, m, prof['beta'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}: dpo loss {loss.item():.4f}")
    print(f"  📉 dpo loss {losses[0]:.4f} → {sum(losses[-5:])/5:.4f}")
    model.eval()
    print("  训练后:")
    dpo_report(f"  训练池 {len(pool)} 对（见过的偏好）", pool, ref_pool, after=True)
    dpo_report(f"  heldout {len(heldout)} 对（没见过的偏好）", heldout, ref_heldout, after=True)
    return model


def graduate(model, tok):
    model.eval()
    print("  三阶段行为对照（同一条问题）:")
    q = "你是谁？"
    ids = torch.tensor([chat_input(tok, q)], device=DEVICE)
    gen = model.generate(ids, max_new_tokens=64, eos_id=tok.eos_token_id)
    print(f"  Q: {q}")
    print(f"  A(DPO 后): {chat_output(tok, gen, ids.shape[1])!r}")
    print("""
  你自己的 minimind 现在会什么：
    v1 基线能训 → v2 有仪器 → v3 稳又快 → v4 注意力现代 → v5 FFN 现代
    → v6 完整架构 25.83M → v7 会说话 → v8 会对话 → v9 偏好更讨喜 🎉
  接下来去哪：Part 8 后训练全家桶 ｜ Part 11/17 在线对齐 ｜ Part 14 用 vLLM 部署它""")


# ══════════════════════════════════════════════════════════════════
# stage 档位（quick 教学档 / full 官方对齐档，2026-09 核对官方 argparse）
# ══════════════════════════════════════════════════════════════════

def stage_profiles(stage, profile):
    if profile == 'full':
        if stage == 7:   # 官方 train_pretrain.py 默认: epochs2/batch32/accum8/lr5e-4/seq340
            return dict(epochs=2, batch_size=32, accum=8, lr=5e-4, seq=340,
                        max_samples=None, amp='bf16', eval_every=500)
        if stage == 8:   # 官方 train_full_sft.py 默认: epochs2/batch16/accum1/lr1e-5/seq768
            return dict(epochs=2, batch_size=16, accum=1, lr=1e-5, seq=768,
                        max_samples=None, amp='bf16', eval_every=500)
        if stage == 9:   # 官方 train_dpo.py 默认: epochs1/batch4/lr4e-8/beta0.15/seq1024
            return dict(epochs=1, batch_size=4, lr=4e-8, beta=0.15, seq=1024,
                        max_samples=None, pool=None, amp='bf16')
        return {}
    # quick 教学档
    return {
        1: dict(steps=60, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=8000, amp='off'),
        2: dict(steps=300, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=20000, amp='bf16', eval_every=100),
        3: dict(steps=120, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=20000),
        4: dict(steps=300, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=20000),
        5: dict(steps=300, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=20000),
        6: {},
        7: dict(steps=300, batch_size=16, accum=2, lr=5e-4, lr_cpu=1e-3, seq=340,
                max_samples=20000, amp='bf16', eval_every=100),
        8: dict(steps=300, batch_size=16, accum=1, lr=5e-5, lr_cpu=1e-3, seq=768,
                max_samples=6000, amp='bf16', eval_every=100),
        9: dict(steps=200, batch_size=4, lr=1e-6, lr_cpu=5e-5, beta=0.15, seq=1024,
                max_samples=2000, pool=200),
    }.get(stage, {})


# ══════════════════════════════════════════════════════════════════
# stage 调度
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="你的 minimind——实验驱动生长版（教程 02-11 章配套）")
    ap.add_argument('--stage', type=int, required=True, choices=range(1, 10))
    ap.add_argument('--profile', choices=['quick', 'full'], default='quick',
                    help='quick=教学快速档；full=官方默认超参+全量数据（9/10/11 章生效）')
    ap.add_argument('--ddp-demo', action='store_true',
                    help='stage 3：用 torchrun --nproc_per_node=2 启动，演示 DistributedSampler')
    args = ap.parse_args()
    stage = args.stage
    prof = stage_profiles(stage, args.profile)
    if os.environ.get('P7_STEPS'):
        prof = dict(prof, steps=int(os.environ['P7_STEPS']))

    print(f"═══ 你的 minimind · stage {stage}（{args.profile}）═══")
    print(f"  device={DEVICE}  profile={current_profile()}")

    if stage == 1:
        tok = load_tokenizer()
        vocab_report(tok)                                                # 02 章 §1.2
        cfg = make_config(current_profile(), modern=False)
        model = build_model(cfg)
        total, rep = model.param_report()
        print(f"  基线模型（MHA+learned PE+ReLU+LayerNorm）参数账本:")
        print(rep)
        print(f"  总参数: {total/1e6:.2f} M（对照官方现代版 25.83M——升级之路先记账）")
        print("  ── 第一次训练：60 步，看 loss 从「均匀猜测」往下走 ──")
        data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
        train_data, val_data = split_train_val(data)
        train_model(model, prof, 's1_baseline_first', train_data, val_data,
                    amp=prof.get('amp', 'off'))
        print("  ↳ v1 达成：最小可用 minimind——能建、能训、loss 会降。先跑通，再变强（03 章起）。")

    elif stage == 2:
        tok = load_tokenizer()
        cfg = make_config(current_profile(), modern=False)
        model = build_model(cfg)
        data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
        train_data, val_data = split_train_val(data)
        print(f"  切分: 训练 {len(train_data):,} 条 ｜ 验证 {len(val_data):,} 条（尾部 5%，训练前切好）")
        train_model(model, prof, 's2_baseline_instrumented', train_data, val_data,
                    amp=prof.get('amp', 'bf16'))
        MetricLogger.view_hint()
        print("""  ── 问题清单（后面每一章修一个）──
  ① fp32 训练慢、显存高            → 04 章 AMP（fp16+GradScaler / bf16）
  ② 训练 340 长、推理 512 会怎样？ → 05 章 learned PE 的外推崩溃 → RoPE
  ③ 同参数还能更 low loss 吗？     → 07 章 ReLU→SwiGLU（论文结论复现）
  ④ 想要官方 25.83M 的身材         → 06-08 章 GQA/SwiGLU/RMSNorm 减肥账本""")

    elif stage == 3:
        if args.ddp_demo:
            world = int(os.environ.get('WORLD_SIZE', 1))
            if world < 2:
                raise SystemExit("单进程没有 DDP 可演示。请用："
                                 "torchrun --nproc_per_node=2 my_minimind.py --stage 3 --ddp-demo")
            ddp_demo(make_config(current_profile(), modern=False), prof)
            return
        amp_shootout(make_config(current_profile(), modern=False), prof)
        print("  ── 梯度累积的算术 ──")
        print("  教学档 batch 16 × accum 2 = 有效 32；官方 batch 32 × accum 8 = 有效 256")
        print("  显存只付 batch 16/32 激活的钱，梯度噪声却按 256 压——等效大 batch 是「穷人法宝」")
        print("  ── 双卡 DistributedSampler 演示 ──")
        print("  torchrun --nproc_per_node=2 my_minimind.py --stage 3 --ddp-demo")
        print("  ↳ v3 达成：fp16/bf16、GradScaler、裁剪、累积、DDP 切分——训练工程的稳定提速件齐了。")

    elif stage == 4:
        print("══ 实验 ①：learned PE vs RoPE（同 seed 各训一次）══")
        rope_vs_learned(prof)
        print()
        print("══ 实验 ②：MHA vs GQA（短训比质量，账本比推理）══")
        gqa_vs_mha(prof)
        print()
        print("══ 实验 ③：QK-Norm（模拟范数漂移）══")
        qk_norm_experiment()
        print("  ↳ v4 达成：注意力三次进化，每一次都有数字背书。")

    elif stage == 5:
        print("══ 实验 ①：ReLU vs SwiGLU（同参数预算）══")
        relu_vs_swiglu(prof)
        print()
        print("══ 实验 ②：MoE 路由与 aux loss（随机初始化下的最小样本）══")
        moe_experiment()
        print()
        print("══ 实验 ③：MoE 负载均衡 α 扫描（0 / 0.01 / 5e-4）══")
        moe_load_balance()
        print("  ↳ v5 达成：FFN 两次升级——形状换收益（SwiGLU）、容量解耦算力（MoE）。")

    elif stage == 6:
        tok = load_tokenizer()
        base = build_model(make_config(current_profile(), modern=False))
        modern = build_model(make_config(current_profile(), modern=True))
        tb, rb = base.param_report()
        tm, rm = modern.param_report()
        print(f"  基线（02 章）合计 {tb/1e6:.2f} M：")
        print(rb)
        print(f"  现代版（05-08 章开关全开）合计 {tm/1e6:.2f} M：")
        print(rm)
        print(f"  对照: 官方 26M 口径 ≈25.8M ✓（GQA 省注意力、SwiGLU 守恒、RMSNorm 减法、RoPE 零参数）")
        ids = torch.tensor([[tok.bos_token_id]], device=DEVICE)
        gen = modern.generate(ids, max_new_tokens=24)
        print(f"  随机权重生成: {tok.decode(gen[0].tolist())!r}")
        print("  ↳ v6 达成：你的 minimind 完全体——架构现代、身材对账官方。训练在 09-11 章。")

    elif stage == 7:
        model = build_model(make_config(current_profile(), modern=True))
        tok = load_tokenizer()
        data = build_pretrain_dataset(tok, prof['seq'], prof['max_samples'])
        train_data, val_data = split_train_val(data)
        train_model(model, prof, 's7_pretrain', train_data, val_data,
                    amp=prof.get('amp', 'bf16'))
        save_model(model, model.cfg, 'pretrain')
        demo_continuation(model, tok)
        print("  ↳ v7 达成·阶段一毕业：会『说话』了（续写）。")

    elif stage == 8:
        model = build_model(make_config(current_profile(), modern=True))
        tok = load_tokenizer()
        load_weights(model, 'pretrain')
        data = build_sft_dataset(tok, prof['seq'], prof['max_samples'])
        train_data, val_data = split_train_val(data)
        train_model(model, prof, 's8_sft', train_data, val_data,
                    amp=prof.get('amp', 'bf16'))
        save_model(model, model.cfg, 'full_sft')
        demo_chat(model, tok)
        print("  ↳ v8 达成·阶段二毕业：会一问一答了。")

    elif stage == 9:
        model = build_model(make_config(current_profile(), modern=True))
        tok = load_tokenizer()
        load_weights(model, 'full_sft')
        model = train_stage9(model, model.cfg, tok, prof)
        save_model(model, model.cfg, 'dpo')
        graduate(model, tok)
        print("  ↳ v9 达成·阶段三毕业：🎉 这是你的 minimind。")


if __name__ == '__main__':
    main()
