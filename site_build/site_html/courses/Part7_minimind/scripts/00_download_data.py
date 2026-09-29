#!/usr/bin/env python3
"""
Part 7 - 脚本 00：下载 minimind 官方 mini 数据 + 官方分词器

Part 7 全系列脚本吃官方 jsonl（不依赖任何莎士比亚语料）：
  pretrain_t2t_mini.jsonl  ~1.2GB  {"text": "问句？答句。"}
  sft_t2t_mini.jsonl       ~1.6GB  {"conversations": [{role, content, ...}, ...]}
  dpo.jsonl                ~53MB   {"chosen": [...], "rejected": [...]}
数据放 ../dataset/（即 Part7_minimind/dataset/，已加入 .gitignore，不入库）。

用法：
  python 00_download_data.py                 # 下载全量三件 + tokenizer（约 2.9GB，可断点续传）
  python 00_download_data.py --sample-only   # 只切小样本（从已下载的全量文件取样）
  python 00_download_data.py --sample-only --no-full      # 边下边切小样本，不下全量（省流量）
  python 00_download_data.py --check         # 体检：行数/大小/首行预览

小样本（*_sample.jsonl）供 CPU/快速验证：默认 pretrain 20000 行 / sft 6000 行 / dpo 4000 行。
官方不建议重训 tokenizer；01 章会在 mini 语料上自训一个教学版与它对照，脚本默认直接用官方版。
"""

import argparse
import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'dataset'))
TOK_DIR = os.path.join(DATA_DIR, 'tokenizer')

MS_BASE = "https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master"
HF_BASE = "https://hf-mirror.com/datasets/jingyaogong/minimind_dataset/resolve/main"
GH_RAW = "https://raw.githubusercontent.com/jingyaogong/minimind/master/model"

DATA_FILES = ['pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl', 'dpo.jsonl']
TOK_FILES = ['tokenizer.json', 'tokenizer_config.json']
SAMPLE_LINES = {'pretrain_t2t_mini.jsonl': 20000, 'sft_t2t_mini.jsonl': 6000, 'dpo.jsonl': 4000}


def http_get_stream(url, start_from=0):
    """带 Range 续传的流式 GET。返回 (response, 是否续传)。"""
    req = urllib.request.Request(url, headers={'User-Agent': 'p7-tutorial'})
    if start_from > 0:
        req.add_header('Range', f'bytes={start_from}-')
    resp = urllib.request.urlopen(req, timeout=60)
    return resp, start_from > 0


def download_full(url, path):
    """下载完整文件，已存在且不完整时续传（ModelScope CDN 支持 Accept-Ranges）。"""
    if os.path.exists(path):
        local = os.path.getsize(path)
        try:
            req = urllib.request.Request(url, method='HEAD')
            total = int(urllib.request.urlopen(req, timeout=30).headers.get('Content-Length', 0))
        except Exception:
            total = 0
        if total and local == total:
            print(f"  ✅ {os.path.basename(path)} 已完整（{local/1e9:.2f} GB），跳过")
            return
        if total and local < total:
            print(f"  ⏳ 续传 {os.path.basename(path)}：本地 {local/1e6:.0f}MB / 远端 {total/1e6:.0f}MB")
            resp, _ = http_get_stream(url, start_from=local)
            mode, done = 'ab', local
        else:
            resp, mode, done = http_get_stream(url), 'wb', 0
    else:
        resp, mode, done = http_get_stream(url), 'wb', 0
    with open(path, mode) as f, resp:
        got = 0
        while True:
            chunk = resp.read(1 << 20)          # 1MB
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            done += len(chunk)
            if got // (100 << 20) != (got - len(chunk)) // (100 << 20):
                print(f"    ...已下载 {done/1e6:.0f} MB")
    print(f"  ✅ {os.path.basename(path)} 完成（{os.path.getsize(path)/1e6:.0f} MB）")


def cut_sample(src_path, dst_path, n_lines):
    """取前 n_lines 个完整行存成 *_sample.jsonl。全量文件在本地就直接切，不必再联网。"""
    if os.path.exists(dst_path) and sum(1 for _ in open(dst_path, encoding='utf-8')) >= n_lines:
        print(f"  ✅ {os.path.basename(dst_path)} 已存在（≥{n_lines} 行），跳过")
        return
    if os.path.exists(src_path):                       # 从本地全量文件切
        with open(src_path, encoding='utf-8') as f, open(dst_path, 'w', encoding='utf-8') as out:
            for i, line in enumerate(f):
                if i >= n_lines:
                    break
                out.write(line)
    else:                                              # 边下边切（--no-full 模式）
        url = data_url(src_path)
        resp = http_get_stream(url)[0]
        with open(dst_path, 'w', encoding='utf-8') as out, resp:
            buf, count = b'', 0
            for chunk in resp:
                buf += chunk
                *lines, buf = buf.split(b'\n')
                for ln in lines:
                    if ln.strip():
                        out.write(ln.decode('utf-8', errors='replace') + '\n')
                        count += 1
                if count >= n_lines:
                    break
    print(f"  ✂️  {os.path.basename(dst_path)}：前 {n_lines} 行，"
          f"{os.path.getsize(dst_path)/1e6:.1f} MB")


def data_url(name):
    base = MS_BASE if SOURCE == 'modelscope' else HF_BASE
    return f"{base}/{name}"


def download_tokenizer():
    os.makedirs(TOK_DIR, exist_ok=True)
    for name in TOK_FILES:
        path = os.path.join(TOK_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 100:
            print(f"  ✅ tokenizer/{name} 已存在，跳过")
            continue
        urllib.request.urlretrieve(f"{GH_RAW}/{name}", path)
        print(f"  ✅ tokenizer/{name}（{os.path.getsize(path)/1e3:.0f} KB）")


def check():
    print("═══ 数据体检 ═══")
    for name in DATA_FILES:
        for suffix in ['', '_sample']:
            path = os.path.join(DATA_DIR, name.replace('.jsonl', f'{suffix}.jsonl'))
            if not os.path.exists(path):
                continue
            n = sum(1 for _ in open(path, encoding='utf-8'))
            first = json.loads(open(path, encoding='utf-8').readline())
            keys = list(first.keys())
            preview = json.dumps(first, ensure_ascii=False)[:120]
            print(f"  {os.path.basename(path):36s} {n:>9,} 行  字段={keys}\n    首行: {preview}...")
    tok = os.path.join(TOK_DIR, 'tokenizer.json')
    if os.path.exists(tok):
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(TOK_DIR)
            print(f"  tokenizer: vocab={len(tk)}, bos={tk.bos_token!r}(id={tk.bos_token_id}), "
                  f"eos={tk.eos_token!r}(id={tk.eos_token_id}), pad={tk.pad_token!r}(id={tk.pad_token_id})")
        except Exception as e:
            print(f"  ⚠️ tokenizer 加载失败：{e}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="下载 minimind 官方 mini 数据与分词器")
    ap.add_argument('--source', default='modelscope', choices=['modelscope', 'hf'],
                    help="数据源（hf 走 hf-mirror.com，可用 HF_ENDPOINT 覆盖）")
    ap.add_argument('--sample-only', action='store_true', help="只生成小样本文件")
    ap.add_argument('--no-full', action='store_true', help="配合 --sample-only：边下边切，不落全量文件")
    ap.add_argument('--check', action='store_true', help="只体检已下载的数据")
    args = ap.parse_args()
    SOURCE = args.source

    if args.check:
        check()
        sys.exit(0)

    os.makedirs(DATA_DIR, exist_ok=True)
    print("═══ 1/2 官方分词器（GitHub raw，不建议重训——01 章有教学版对照） ═══")
    download_tokenizer()

    print("═══ 2/2 官方 mini 数据（jsonl） ═══")
    for name in DATA_FILES:
        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(DATA_DIR, name.replace('.jsonl', '_sample.jsonl'))
        if args.sample_only:
            if args.no_full:
                cut_sample(src, dst, SAMPLE_LINES[name])          # 走网络边下边切
            else:
                if not os.path.exists(src):
                    download_full(data_url(name), src)
                cut_sample(src, dst, SAMPLE_LINES[name])
        else:
            download_full(data_url(name), src)

    if not args.sample_only or not args.no_full:
        for name in DATA_FILES:                                   # 全量就位后顺手切样本
            src = os.path.join(DATA_DIR, name)
            if os.path.exists(src):
                cut_sample(src, os.path.join(DATA_DIR, name.replace('.jsonl', '_sample.jsonl')),
                           SAMPLE_LINES[name])
    print("\n下一步：python 01_bpe_tokenizer.py（教学版 BPE 对照）或 python 02_baseline.py（直接开训）")
