#!/usr/bin/env python3
"""
Part 8 - 脚本 12: lm-evaluation-harness 实操 —— Python API + 自定义 task
目标：把 07 章 §2 讲的"预训练评估事实标准"真正跑起来，三步走：
  ① arc_easy，limit=100，num_fewshot=0（Qwen2.5-0.5B-Instruct，hf 后端）
  ② 同任务 num_fewshot=5 对比 —— ⚠️ fewshot 数变了基线就变，两个分数不可互比
  ③ 自定义 task：mytasks/course_quiz（TaskManager(include_path=...) 加载本地 yaml，
     5 道本课程知识题：DPO/GRPO/LoRA/量化/RAG 概念四选一）

优雅降级：
  - lm_eval 未安装 → 打印安装指引并 sys.exit(0)（rc=0，不报错）
  - arc_easy 数据集下载失败 → 打印说明，跳过 ①②，仍尝试 ③

运行：
    MPLBACKEND=Agg python 12_lm_eval_hands_on.py
预期耗时：GPU 上每档 arc_easy（100 题×4 选项 loglikelihood）约 1-3 分钟。

坑位地图（详见教程 07 章 lm-eval-harness 实操小节）：
  - v0.4.10 起 lm_eval 默认不装 HF 栈：必须 pip install "lm_eval[hf]"（本课实测 0.4.13）
  - 自定义模型架构要 trust_remote_code 时，它是 model_args 的参数不是 CLI 顶层参数
  - --batch_size auto 可能 OOM，auto:N（如 auto:8）是带上限的自动探测
  - 随机性由"四元组"控制：python/numpy/torch/fewshot 四个 seed 各自独立
  - v0.4.12 起 vLLM 后端最低版本要求变严（本课环境未装 vLLM，vLLM 命令为预期行为）
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MYTASKS_DIR = os.path.join(SCRIPT_DIR, 'mytasks')
YAML_PATH = os.path.join(MYTASKS_DIR, 'course_quiz.yaml')
JSONL_PATH = os.path.join(MYTASKS_DIR, 'course_quiz.jsonl')

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
BATCH_SIZE = 16          # 0.5B 在 24GB 卡上 16 很稳；紧张就降 8 或用 "auto:8"
LIMIT = 100

# yaml 模板：data_files 用绝对路径写入（相对路径以 cwd 为基准解析的坑，见下）
# ⚠️ 写到【临时目录】而不是仓库里的 mytasks/course_quiz.yaml——仓库版本保持相对路径
#    规范形态（cwd == mytasks/ 时可直接 lm_eval --include_path mytasks/ 用），
#    本脚本运行时生成临时副本，不污染被 git 跟踪的文件。
YAML_TEMPLATE = """\
# lm-evaluation-harness 自定义 task：本课程知识 5 题（DPO/GRPO/LoRA/量化/RAG）
# （运行时生成的临时副本：data_files 为绝对路径，免疫 cwd 坑；规范相对路径版见仓库
#   mytasks/course_quiz.yaml——在 mytasks/ 目录下运行 lm_eval 时可直接使用）
task: course_quiz
dataset_path: json
dataset_kwargs:
  data_files:
    test: {jsonl_abs}
test_split: test          # 不写这行，0.4.13 报 "must have valid or test docs"（实测坑）
output_type: multiple_choice
doc_to_text: "Q: {{{{question}}}}\\nA:"
doc_to_target: "{{{{answer}}}}"
doc_to_choice: "{{{{choices}}}}"
metric_list:
  - {{metric: acc}}
"""


def make_runtime_yaml():
    """在系统临时目录生成一份绝对路径版 course_quiz.yaml，返回 (yaml_path, include_dir)。

    为什么不直接用仓库里的 mytasks/course_quiz.yaml：其相对路径 `test: course_quiz.jsonl`
    只有在 cwd == mytasks/ 时才解析得到（data_files 相对路径以【运行 lm_eval 时的 cwd】
    为基准，不是 yaml 所在目录——这是 lm-eval 的解析坑）；而直接改写仓库文件又会把绝对
    路径提交进 git。折中：仓库版保持规范相对路径，本脚本用临时副本从任何目录跑都成立。
    """
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='lm_eval_mytask_')
    yaml_path = os.path.join(tmp_dir, 'course_quiz.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(YAML_TEMPLATE.format(jsonl_abs=JSONL_PATH))
    print(f"  [OK] 临时 task yaml → {yaml_path}")
    print(f"       data_files.test → {JSONL_PATH}（绝对路径，免疫 cwd 坑；仓库 yaml 未被改动）")
    return yaml_path, tmp_dir


def run_task(task_names, num_fewshot, task_manager=None):
    """调 simple_evaluate 跑一组任务，返回 {task: acc}。"""
    import lm_eval

    model_args = (
        f'pretrained={MODEL},'
        f'dtype=float16,'
        f'batch_size={BATCH_SIZE}'
    )
    print(f"  model_args = {model_args}")
    print(f"  tasks={task_names}, num_fewshot={num_fewshot}, limit={LIMIT}")

    result = lm_eval.simple_evaluate(
        model='hf',
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        limit=LIMIT,
        task_manager=task_manager,
        verbosity='ERROR',          # 压掉 INFO 日志，只留错误
        log_samples=False,          # 不保存逐题样本（省内存）
        bootstrap_iters=0,          # 不做 bootstrap 置信区间（省时间）
    )
    # 结果结构：results[task]['acc,none'] = 准确率（acc 指标的默认聚合值）
    out = {}
    for t in task_names:
        metrics = result['results'][t]
        acc = metrics.get('acc,none')
        out[t] = acc
        print(f"  → {t} acc = {acc:.4f}" if acc is not None else f"  → {t} 指标键: {list(metrics)}")
    return out


def main():
    print("═══ Part 8 脚本 12: lm-evaluation-harness 实操 ═══")

    # ── Step 0: 依赖自检（未装 → 指引 + rc=0 退出）──
    try:
        import lm_eval  # noqa: F401
        from lm_eval.tasks import TaskManager  # noqa: F401
    except ImportError:
        print("  [MISS] 未安装 lm_eval。安装指引（v0.4.10+ 必须带 [hf] extras）：")
        print('         pip install "lm_eval[hf]"   # 本课实测版本 0.4.13')
        print("         注意：不带 extras 的 lm_eval 默认不装 transformers/datasets 等 HF 栈，")
        print("               import 能过但 model='hf' 会在运行时报错——这是 0.4.10 的著名变更。")
        sys.exit(0)

    import lm_eval
    print(f"  lm_eval 版本: {lm_eval.__version__}")
    print(f"  模型: {MODEL}（本地缓存优先）")
    print(f"  种子四元组（simple_evaluate 默认值）: python=0, numpy=1234, torch=1234, fewshot=1234")
    print("  —— fewshot 采样由 fewshot seed 控制：换它 = 换题 = 分数会动，复现实验要四个都钉住")

    # ── Step 1+2: arc_easy，0-shot vs 5-shot ──
    print(f"\n── Step 1: arc_easy limit={LIMIT} num_fewshot=0 ──")
    arc_scores = {}
    try:
        arc_scores[0] = run_task(['arc_easy'], num_fewshot=0)
    except Exception as e:
        print(f"  [SKIP] arc_easy 0-shot 失败：{type(e).__name__}: {str(e)[:200]}")
        print("         常见原因：ai2_arc 数据集下载失败（网络）——检查代理后重跑。")

    if 0 in arc_scores:
        print(f"\n── Step 2: 同任务 num_fewshot=5 对比 ──")
        print("  ⚠️ 坑：fewshot 数变了，任务对模型的'难度'就变了——")
        print(f"     0-shot 刚测得 {arc_scores[0]['arc_easy']:.1%}（本次运行实测值），5-shot 出来后")
        print("     两者是【两条独立基线】，不能说'提升了 X 个点'（fewshot 数也是实验变量）。")
        try:
            arc_scores[5] = run_task(['arc_easy'], num_fewshot=5)
        except Exception as e:
            print(f"  [SKIP] arc_easy 5-shot 失败：{type(e).__name__}: {str(e)[:200]}")

    # ── Step 3: 自定义 task ──
    print("\n── Step 3: 自定义 task（mytasks/course_quiz）──")
    if not os.path.exists(JSONL_PATH):
        print(f"  [MISS] 找不到 {JSONL_PATH}，自定义任务跳过")
    else:
        # 坑：yaml 里 data_files 相对路径以 cwd 为基准 → 生成绝对路径临时副本（不改仓库文件）
        _, tmp_dir = make_runtime_yaml()
        from lm_eval.tasks import TaskManager
        tm = TaskManager(include_path=tmp_dir)  # 只扫临时目录，不与内置任务重名
        try:
            custom = run_task(['course_quiz'], num_fewshot=0, task_manager=tm)
        except Exception as e:
            print(f"  [SKIP] 自定义任务失败：{type(e).__name__}: {str(e)[:200]}")

    # ── 汇总 ──
    print("\n═══ 汇总 ═══")
    if 0 in arc_scores:
        print(f"  arc_easy (limit=100, 0-shot): {arc_scores[0]['arc_easy']:.1%}")
    if 5 in arc_scores:
        print(f"  arc_easy (limit=100, 5-shot): {arc_scores[5]['arc_easy']:.1%}")
    if 0 in arc_scores and 5 in arc_scores:
        print("  ⚠️ 再次提醒：0-shot 与 5-shot 是两条不同的基线，分数不可互比（要比较")
        print("     请固定同一 num_fewshot，只改模型/训练阶段）。")
    try:
        print(f"  course_quiz (自定义 5 题): {custom['course_quiz']:.1%}")
    except (NameError, KeyError):
        pass

    print("""
  vLLM 后端（本课环境未装 vLLM，以下为预期行为，供参考）：
    lm_eval --model vllm --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct, \
            dtype=auto,gpu_memory_utilization=0.8 \
            --tasks arc_easy --limit 100 --batch_size auto
    vLLM 后端吞吐显著高于 hf（连续批处理），大批量评测首选；
    v0.4.12 起 vLLM 最低版本要求变严，版本不匹配会直接拒绝加载。

  方法论参考：Biderman et al., "Lessons from the Trenches on
  Integrating LLMs"（arXiv 2405.14782，lm-eval-harness v0.4 论文）""")


if __name__ == '__main__':
    main()
