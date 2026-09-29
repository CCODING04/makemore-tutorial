# 07 — FFN 进化：ReLU → SwiGLU → MoE（v5）

> 🧭 问题清单第 ③ 条：同参数预算下，loss 还能更低吗？FFN 是参数大头（02 章账本：57.9%），
> 在这里动刀收益最大。本章两个实验：①**同参数预算**下把 ReLU 换成 SwiGLU（SiLU 门控的三投影 FFN，本章主角），看 ppl 变化、
> 对照论文结论；②把 FFN 换成 MoE，理解"容量与算力解耦"。老规矩：先看代码、再跑实验、
> 后对论文。

## 🎒 前置回忆包（不翻旧章也能读）

- **FFN（基线位）**：`Linear(d→4d) → ReLU → Linear(4d→d)`，每层 2·d·4d = 2.10M 参数（d=512），
  对每个 token **独立**变换——注意力的"通信"之后，FFN 是每个 token 的"私有计算"。
- **验证集 ppl**：03 章装的裁判，本章所有对比由它判决。

## 📍 你现在的位置

```text
v4 现代注意力 ──本章──▶ v5：现代 FFN（SwiGLU + MoE 认知）
```

## 实验 ①：ReLU vs SwiGLU——同样的参数预算，换个形状就更强

**假设**（来自论文）：2019 年后的系列实验（Shazeer, *GLU Variants Improve Transformer*,
2020）发现把 ReLU 换成"门控"结构，**同样参数量**语言建模 loss 更低。门控的直觉：

```python
down( silu(gate(x)) * up(x) )
#      └门：每维输出"通过多少"┘└内容：原始特征
```

gate 学会"这条特征这条句子用不用"，乘法门控做**逐维度的信息筛选**——比无差别 ReLU 细。

**手算一遍 silu**（`silu(z) = z·σ(z)`，σ 是 sigmoid），关键差别全在负半轴：

| z | ReLU(z) | silu(z) | 说明 |
|:---:|:---:|:---:|---|
| -3 | 0 | -0.14 | ReLU 直接关死；silu 保留一点"负的微量"，梯度仍不为 0 |
| -1 | 0 | -0.27 | 同上——优化器还能"告诉"这个维度该往哪调 |
| 0 | 0 | 0 | 平滑经过（ReLU 在 0 处是尖角） |
| 1 | 1 | 0.73 | silu 略"收一点" |
| 3 | 3 | 2.86 | 大正值接近线性 |

- 💡 ReLU 对任何负数都输出 0（信息"死"了、梯度为 0）；silu 输出很小的负值、梯度存在。
  ⚠️ 门值**不限幅在 0~1**——`silu(3)=2.86` 就是反例，它的卖点不是"压到 0~1"，而是
  "负区不死、正区线性"。

📊 **交互动图（页内）**：ReLU（硬开关）、SiLU（软门控）、σ（SiLU 里的门函数）三条曲线
对比——**悬停**查看任意点的取值。注意 SiLU 在负半轴"不死"（最小值 ≈ (−1.28, −0.28)，
梯度仍非零）、正半轴近似线性（`silu(3) ≈ 2.86`），而 ReLU 在 0 处是尖角：

```plot
{
  "title": "ReLU vs SiLU vs σ：硬开关 → 软门控",
  "x": [-6, 6], "y": [-1.2, 4.5],
  "curves": [
    {"expr": "Math.max(0,x)", "label": "ReLU(z)"},
    {"expr": "x/(1+Math.exp(-x))", "label": "SiLU(z) = z·σ(z)"},
    {"expr": "1/(1+Math.exp(-x))", "label": "σ(z)（门）"}
  ],
  "hlines": [
    {"y": 0, "label": "y = 0"}, {"y": 1, "label": "y = 1（σ 上界）"}
  ],
  "points": [[-1.278, -0.278, "SiLU 最小值 ≈ (-1.28, -0.28)", "below"]]
}
```

🔑 **参数守恒账**（本章最重要的计算）：SwiGLU 有 3 个投影矩阵，是 ReLU FFN（2 个）的 1.5 倍。
要守住总参数，中间维度得缩水：`3·d·i ≈ 2·d·4d` → **i = 8d/3 ≈ 2.67d**（Llama 论文的"2/3·4d"）：

```python
def intermediate_of(hidden):
    """8/3 守恒 → 再向上对齐 64。512 → 1408（≈2.75d）"""
    v = int(hidden * 8 / 3)
    return 64 * ((v + 63) // 64)      # 64 = tensor core 的 tile 大小——数学差几十无所谓，
                                      # 工程必须对齐；这也是"论文公式≠仓库代码"的典型原因
```

官方仓库两代实现：26M（minimind2-small）用 8/3 守恒版 → 1408；master（minimind-3 起）用
π 版 `ceil(π·d/64)·64` → 768→2432。两代都是守恒启发式，π 版略胖 ~18%；round 版"512→1600"
是常见笔误——round 与 ceil 只在 512 这个点分叉。

**实验**（`--stage 5`，现代底座只换 FFN，同 seed 各训 300 步）：

```text
  FFN       每层参数   末段 loss   验证集 ppl
  relu       2.10M    5.6848      310.53
  swiglu     2.16M    5.5683      278.41
  ↳ 同参数预算（每层 2.10M vs 2.16M，每层口径——log 主表为 8 层总量 16.78M/17.30M）
    换「形状」：ppl 311 → 278（-10%）
```

📜 **论文对照**：方向与 Shazeer 一致（门控赢），且 tiny 规模就见效（-10%）。但注意两个
"实验素养"问题：① 单次运行有波动，正式消融要跑多种子取均值；② 论文里的增益百分比来自
充分训练，300 步小跑的 -10% 不能外推成"SwiGLU 恒赚 10%"——**复现论文数字前先看人家的
规模和训练量**。

## 实验 ②：dense → MoE——容量和算力解耦

SwiGLU 之后还有个天花板：参数越多，每个 token 的计算量同步变贵。**MoE（混合专家）**把 FFN
换成"N 个专家 + 一个路由器"，每个 token 只走其中 1~2 个——**总容量翻 N 倍，单 token 算力
不变**。官方放大版 minimind-3-moe 就是 **198M 总参、每 token 激活约 64M**（口径与
`my_minimind.py --stage 5` 的打印一致）。自己动手写会立刻撞上两个问题（官方代码里各有一行"补丁"）：

```python
class MoEExperiment(nn.Module):
    def forward(self, x):
        scores = F.softmax(self.gate(xf), dim=-1)          # router 给每个专家打分
        top_w, top_i = torch.topk(scores, self.k, dim=-1)  # top-1 选专家
        top_w = top_w - top_w.detach() + 1.0               # ① 直通梯度（见下）
        for e, expert in enumerate(self.experts):
            sel = (top_i == e)
            if sel.any():
                y = y + expert(xf) * w                     # 教学版：全量算再掩
        load = F.one_hot(top_i, self.n).float().mean(0).sum(0)   # f_i：落到专家 i 的 token 占比
        self.aux_loss = (load * scores.mean(0)).sum() * self.n * self.coef   # ② 负载均衡
```

命名澄清：代码里的 `self.gate` 就是正文说的路由器（避免与 SwiGLU 的门投影 gate(x) 混淆）。

1. **top-1 路由的梯度断流**：token 只走 1 个专家时，加权系数就是 1——router 的打分根本没
   出现在前向里，梯度传不回 router。解法一行魔法 `top1 - top1.detach() + 1.0`：前向值仍等于
   1.0（行为不变），反向梯度直通 router（straight-through 技巧，官方 2026-09 修复的 issue #858）。
2. **"没分到 token 的专家"让多卡训练报错**：零命中专家的参数不进计算图，DDP 抱怨"有参数未被
   使用"。官方给输出加 `0 × 参数和`——数值上是 +0，但把参数拉回了计算图。

还有**负载均衡**：自由路由会让所有 token 挤同一个"明星专家"。辅助损失
`aux = N·Σ f_i·P_i·α`（α=5e-4）惩罚"打分高的专家和实际接客的专家错位"。

> 🎛️ **交互演示**：把这条公式跑起来了（top-1 路由、$\alpha{=}1$ 便于观察、4 专家 × 12 token，
> 与上面 `MoEExperiment` 同一套记号）——拖「路由集中度」或点预设（均衡/中等倾斜/完全塌缩），
> 看 token 如何涌向个别专家、每个专家的 $f_i$/$P_i$ 两根柱子和 $L_{aux}$ 读数从 1.00（均衡）
> 滑向 4.00（塌缩）；悬停任意 token 还能看它的完整路由概率分布。

```widget
moe_aux_loss
1210
```

**实验**（`--stage 5` 实验 ②，随机初始化下的最小样本）：

```text
256 个 token 的路由分布: ['21%', '27%', '30%', '21%']（≈均匀：随机初始化的正常现象）
aux_loss = N·Σf_i·P_i·α = 0.0005（≈0：不均衡还没发生，aux 不发力）
```

🔑 路由分布接近均匀是**随机初始化**的正常现象；"一家独大长什么样、aux 怎么拉平"的完整
分化实验，见本章末尾的「⚖️ 负载均衡 α 扫描（实验③）」。

### minimind 的 MoE 是可选项

- 🔑 minimind 的默认 `MiniMindConfig(use_moe=False)` 是 **Dense（稠密）**模型；把
  `use_moe=True` 才切换成 MoE 版（官方放大版 minimind-3-moe：4 专家 / top-1、hidden 768，
  **198M 总参、每 token 激活约 64M**）。
- 💡 对我们 25.83M 的小模型，MoE 属于"锦上添花"：理解概念为主，09-11 章的训练流水线默认
  不开 MoE；等你想真复现 minimind-3-moe（198M）再开（12 章毕业指南）。
- 📝 **口径提醒**：路由演示（实验②）4 专家 top-1、负载均衡扫描（实验③）8 专家 top-1——均为教学口径；官方 minimind-3-moe 为 4 专家 top-1（配置以 12 章复现指南与官方仓库为准）。答题时说清"我教学实现里用的哪个、
  官方是哪个"，就是加分的可辩护版本。

## ⚖️ 负载均衡 α 扫描（实验③）

实验②是随机初始化的最小样本——路由天然均匀，aux 根本没机会发力。实验③把"一家独大"**真的
训出来**再用药：4 簇聚类数据、8 专家 top-1 路由、三档同 seed 同初始化，只扫 α 一个旋钮
（来源：`my_minimind --stage 5`，2026-09，temp/out/logs/exp07_moe_load_balance.csv）：

| α（aux 系数） | gini（基尼系数：0=完全均匀，越大越集中） ↓ | max/mean ↓ | 任务 loss | 专家负载 f_i |
|---|:---:|:---:|:---:|---|
| 0 | **0.533** | 2.53 | **0.0065** | 3 个专家 f_i = 0.00（饿死） |
| 0.01（Switch Transformer 论文推荐） | 0.115 | 1.34 | 0.0234 | 接近均匀 |
| 5e-4（minimind 默认） | **0.087** | 1.28 | 0.0190 | 接近均匀 |

三个读数：

- **α=0 演出"贫富分化"**：gini 0.533、最忙专家接客量是均值的 2.53 倍，3 个专家一个 token
  都没接到（f_i=0.00"饿死"，第 4 个也只剩 0.01）——rich-get-richer：早期稍占优的专家学到
  更多 → 路由器更偏向它 → 循环放大。而任务 loss 反而最低（0.0065）——**塌缩对拟合是
  "高效"的，坏的是容量**：8 个专家实际只养活了 4 个。
- **α=0.01 拉平最狠**（gini 0.115）但任务 loss 最高（0.0234）——均衡要付学费：aux 项与
  任务目标在争同一份梯度。
- **α=5e-4 是官方的落点**：gini 最低（0.087）、max/mean 1.28，任务 loss 只比 α=0 高一点
  （0.019 vs 0.007）——**花很小的任务损失，买回全部专家的利用率**。这正是 minimind
  `router_aux_loss_coef=5e-4` 的取值逻辑：α 不是越大越好，是"刚好够拉平"。

- 🔑 公式里两个量缺一不可：$f_i$（实际接到 token 的频率）**不可微**、负责反映真实负载；
  $P_i$（平均路由概率）**可微**、负责被优化。只留 $P_i$，路由器可以"嘴上说均匀、身体很
  诚实"。运行方式：`python my_minimind.py --stage 5`（与实验①②同一条命令，第 ③ 段输出）。

## ▶️ 运行本章成果

```bash
python my_minimind.py --stage 5      # 三组实验一次跑完（约 3 分钟）
```

## 🏭 工业界

DeepSeek-V3、Qwen3-MoE、混元等旗舰全是 MoE——"万亿参数、千亿激活"的公开报价都靠这个解耦。
专家再拆到多卡（Expert Parallel）是
[Part 10 分布式](../../Part10_distributed/tutorial/README.md)的一环。官方 README 还提醒一个
反直觉事实：MoE 训练往往**更慢**（token 分桶导致 kernel 启停开销），"MoE 推理更快"只在推理
侧成立。

## 🪝 引子

- 你现在能算"26M 参数都花在哪"了。参数量定了，**数据要多少、训多久**？
  [Part 13 的 Scaling Laws](../../Part13_data_engineering/tutorial/00_scaling_laws.md) 用一条
  幂律回答——下一章组装完模型我们就去对账。

## 🎯 面试直通车

<details>
<summary>Q: SwiGLU 的中间维度为什么不是 4×hidden？</summary>
A: SwiGLU 三个投影是 ReLU FFN 的 1.5 倍参数，中间维度缩到 8d/3 守恒总参数（Llama 论文的
"2/3·4d"），再向上对齐 64（tensor core tile）。官方两代口径：26M 用 int(d·8/3) 对齐 64 → 1408；
master（minimind-3）用 ceil(π·d/64)·64 → 768→2432。两代都是守恒启发式；本课实测同预算下
SwiGLU ppl 311→278，方向与 Shazeer 2020 一致。
</details>

## ✅ 本章验收

- [ ] `--stage 5` 跑通，能复算"hidden=512 时 8/3 公式 → 1408"和对齐 64 的理由
- [ ] 能报出同参对比的两个数字（311 vs 278）并说出"论文对照时的两个实验素养问题"
- [ ] 能解释 `top1 - top1.detach() + 1.0` 的前向/反向行为，和 aux 公式每一项的含义

---

[← 上一章：GQA 与 QK-Norm](06_attention_gqa_qknorm.md) | [课程 README](README.md) | [下一章：组装对账 →](08_assemble_model.md)
