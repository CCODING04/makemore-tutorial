# 03 — 实验仪器：没有测量，就没有升级（v2）

> 🧭 上一章的基线"能训"，但你对它一无所知：loss 降得到底好不好？泛化如何？比昨天那次跑好
> 还是差？本章不升级模型，**先升级你自己观察模型的方式**——验证集 ppl、metric 记录（CSV +
> tensorboard + wandb/swanlab）、超参速查表。这套仪器装好后，04/05 章每个组件升级都用它来
> 做"前后对比"——**这就是实验驱动课程和"改一版看一眼"的区别**。

## 🎒 前置回忆包（不翻旧章也能读）

- **交叉熵 loss**：模型对正确下一个 token 的 -log 概率。**ppl = exp(loss)**——"模型等效在多少
  个词里犹豫"，更符合人类直觉（loss 5.6 ↔ ppl 278）。
- **训练/验证切分**：留一小部分数据**不参与训练**，专测"没见过的数据上表现如何"。
- 上一章的基线：28.98M，fp32 训练，loss 从 8.86 起步。

## 📍 你现在的位置

```text
v1 最小可用 ──本章──▶ v2：会做实验的基线（仪器齐全 + 问题清单）
```

## 第 1 件：验证集——训练前切好，永远不许偷看

```python
def split_train_val(data, val_ratio=0.05):
    """训练前先切出验证集：取尾部 5%。"训练开始之前"四个字是纪律，不是细节。"""
    n_val = max(1, int(len(data) * val_ratio))
    return data[:-n_val], data[-n_val:]

@torch.no_grad()
def evaluate(model, data, batch=32):
    model.eval()                      # 评测必须关 dropout（本模型没有，但习惯要从早养）
    ...                               # 整个验证集过一遍，平均交叉熵
    return loss, math.exp(loss)       # ppl = exp(loss)
```

03_experiment_lab.py 实测（20,000 条切出 19,000 训练 / 1,000 验证，每 100 步评一次，fp32）：

```text
    ↳ 验证集: loss 8.4161  ppl 4519.11
    ↳ 验证集: loss 7.0420  ppl 1143.62
    ↳ 验证集: loss 6.4602  ppl 639.21
    ↳ 验证集: loss 6.1575  ppl 472.23
```

- 🔑 **训练 loss 和验证 ppl 要一起看**：训练 loss 一路降、验证 ppl 回升 = 过拟合；两边同步降
  = 还在欠拟合区，可以继续训。300 步的小跑两者都还在降——"训练不够"本身就是第一个发现。

## 第 2 件：metric 记录——CSV 兜底，tensorboard 看曲线，wandb/swanlab 上云端

官方训练脚本用 **SwanLab**（`import swanlab as wandb`）每 100 步记录
`loss / logits_loss / aux_loss / learning_rate / epoch_time`。我们教三条路线，**工程上依次
升级，脚本里三层全支持**：

| 路线 | 依赖 | 什么时候用 |
|---|---|---|
| **CSV** | 无 | 永远可用，Excel 也能打开；脚本兜底 |
| **TensorBoard**（本地） | `pip install tensorboard` | 个人实验首选：零配置、数据在本地 |
| **wandb / swanlab**（云端） | `pip install swanlab` | 团队共享、多实验对比、断点续跑恢复 |

脚本的记录器（`MetricLogger`，正文完整版）：

```python
class MetricLogger:
    FIELDS = ["step", "train_loss", "lr", "val_loss", "val_ppl"]   # 04 章起再加 tok_s/grad_scale

    def __init__(self, run_name):
        self._csv = open(f'.../logs/{run_name}.csv', 'w')       # ① CSV 恒定写
        self._csv.write(','.join(self.FIELDS) + '\n')
        try:
            from torch.utils.tensorboard import SummaryWriter   # ② 检测到才启用
            self._tb = SummaryWriter(f'.../logs/tb/{run_name}')
        except ImportError:
            self._tb = None                                     # 没装也不报错——依赖可选化
        if os.environ.get('P7_SWANLAB') == '1':                 # ③ 官方同款云端，默认关
            import swanlab; swanlab.init(project='my-minimind', experiment_name=run_name)
            self._swan = swanlab

    def log(self, step, **tags):
        row = {'step': step, **tags}
        self._csv.write(','.join(... for k in self.FIELDS) + '\n')
        for k, v in tags.items():
            self._tb.add_scalar(k, v, step)                     # 一条曲线一个标量
        if self._swan is not None:
            self._swan.log(tags, step=step)
```

**怎么看曲线**：

```bash
pip install tensorboard                     # 一次性
python 03_experiment_lab.py             # 训练时自动记录
tensorboard --logdir temp/out/logs --port 6006
# 浏览器开 http://localhost:6006：train_loss 应平滑下降，val_ppl 是你真正关心的那条
```

wandb/swanlab 的用法与官方一致（`swanlab.init(project=..., experiment_name=...)` 后 `log()`
同名字段），好处是多次实验自动汇总成对比表格——04/05 章的消融实验在团队里就是这么比的。

🔑 **记录什么**：不只 loss。`lr`（确认调度真的在走）、`tok_s`（吞吐，04 章 AMP 对比全靠它）、
`grad_scale`（04 章的"保险丝"读数）、`val_ppl`（泛化）。**度量维度决定你能发现哪类问题**。

## 第 3 件：超参速查表——本次实验的全部旋钮

| 超参 | 本课 quick 档 | 官方 full 档（2026-09 核对） | 怎么想 |
|---|---|---|---|
| 学习率 | 5e-4 | 5e-4 | tiny 模型起始值；大了发散小了磨蹭 |
| batch × 累积 | 16（04 章起 ×2） | 32×8=**256** | 有效 batch 压梯度噪声，显存只付小 batch 的钱（04 章） |
| epochs | 定长 300 步 | **2** | Chinchilla"20 token/参数"：26M×20≈5 亿 ≈ mini 语料×2ep |
| seq | 340 | 340 | 官方 argparse 原话"中文 1token≈1.5~1.7 字"，问答对 500~580 字 |
| warmup | 无 | **无**（官方 get_lr 直接满 lr） | 小模型+大 effective batch 成立；大模型加 0.5~2% 步数 |
| grad clip | 1.0 | 1.0 | 全局范数拉回 1 以内（04 章） |
| weight_decay | AdamW 默认 0.01 | 同（官方未暴露） | tiny 模型不敏感，正式实验要显式固定 |
| 精度 | bf16 | bfloat16 | 04 章用实验告诉你为什么 |

🔑 **改超参的唯一正确姿势**：一次只动一个，其余冻结，曲线对比——这正是 04 章之后所有实验
的做法。**"调参玄学"的解药不是经验，是控制变量**。

## ▶️ 运行本章成果

```bash
python 03_experiment_lab.py
tensorboard --logdir ../temp/out/logs --port 6006
```

实测输出已在第 1 件（验证 ppl 曲线）。训练结束后脚本把**问题清单**拍在你面前——后面每章修一个：

```text
── 问题清单（后面每一章修一个）──
① fp32 训练慢、显存高            → 04 章 AMP（fp16+GradScaler / bf16）
② 训练 340 长、推理 512 会怎样？ → 05 章 learned PE 的外推崩溃 → RoPE
③ 同参数还能更 low loss 吗？     → 06 章 ReLU→SwiGLU（论文结论复现）
④ 想要官方 25.83M 的身材         → 07 章 GQA/SwiGLU/RMSNorm 减肥账本
```

## 🏭 工业界

训练循环要配"体检表"：loss 曲线、梯度范数、lr、吞吐、验证指标——大模型训练事故（loss 尖峰、
发散、数据污染）都靠这些曲线在最初几步被发现（[Part 8](../../Part8_post_training/tutorial/README.md)
的预训练体检表是放大版）。实验追踪工具的选型：个人用 tensorboard，团队用 wandb/swanlab/mlflow，
核心不是工具而是**每次实验可追溯**（config + 数据版本 + 曲线一起存）。

## 🪝 引子

- 验证 ppl 是 04/05 章所有组件实验的裁判——下一章先让它跑得更快（AMP）。

## 🎯 面试直通车

<details>
<summary>Q: 为什么必须有验证集？训练 loss 降就不够吗？</summary>
A: 训练 loss 衡量"背了多少"，验证 ppl 衡量"会不会"。语言模型的目标是对新文本的分布建模，
只看训练 loss 无法区分"学到规律"和"背题"。且超参选择（lr、epochs、早停时机）必须依赖一个
不参与训练的信号，否则是在"对训练集调参"——那是过拟合的超参版。
</details>

## ✅ 本章验收

- [ ] `python 03_experiment_lab.py` 跑通，能用 tensorboard 打开曲线并指出 val_ppl 那条
- [ ] 能说出初始 loss ≈ ln(6400) ≈ 8.76 的来历，以及 ppl=exp(loss) 换算
- [ ] 能复述速查表里"官方为什么不加 warmup"和"epochs=2 怎么来的"

---

[← 上一章：基线诞生](02_minimal_model.md) | [课程 README](README.md) | [下一章：稳定与提速 →](04_stability_speed.md)
