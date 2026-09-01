# 01 — 手写 DDPM：扩散模型的三段数学

> 🧭 文生图的"引擎"是扩散模型。本章在 2D 玩具分布上手写它的完整数学
> （跑 [scripts/01_ddpm_from_scratch.py](../scripts/01_ddpm_from_scratch.py)，CPU 30 秒）：
> **前向闭式加噪 → ε 预测训练 → 反向采样循环**。机制与 512×512 图像生成完全同构，
> 只是维度小到能看清每一步。

## 学习目标

完成本章后，你将能够：

- ✅ **手写** DDPM 的完整数学（前向闭式 + ε 预测训练 + 采样循环）
- ✅ **解释** 前向闭式的物理含义（信号保留比例）
- ✅ **画出** 扩散模型的数据流并标注每步 shape
- ✅ **识别** β schedule 选择、方差口径等常见陷阱

## 📖 前置知识

**必须掌握：**
- 高斯分布的加法（两个高斯之和仍是高斯，方差相加）——前向闭式的全部数学基础，
  本章数学全部现推，无扩散基础也能跟上

**建议掌握：**
- [Part 6 02 章 · Attention 从零开始](../../Part6_transformer/tutorial/02_attention_from_scratch.md)——
  Q·K/V"按相似度挑选信息"的机制是 02/03 章 cross-attention 条件注入的地基（本章用不到，读本章前可跳过）

**可选：**
- [Part 8 06 章 · 推理与量化](../../Part8_post_training/tutorial/06_inference_and_serving.md)——
  02 章 SD 实操的 fp16 显存策略与它同源（感兴趣再回看）

## 理论背景

### 问题引入：为什么需要扩散模型？

生成模型虽然强大，但有两种主流方法：

1. **GAN**：训练不稳定，模式崩塌
2. **VAE**：生成质量不高，模糊

扩散模型通过**逐步去噪**来弥补：

```
GAN:    "直接生成，训练不稳定"
VAE:    "压缩再解压，质量不高"
扩散:   "逐步去噪，质量高且稳定"
```

> 💡 **类比**：GAN 像是直接画画，VAE 像是先拍照再画画，扩散像是先涂满颜料再慢慢擦出画。

### 数学推导：前向扩散的闭式解

**问题设定：**
- 原始数据：x_0
- 噪声级别：t ∈ {0, 1, ..., T}
- 噪声 schedule：β_1, β_2, ..., β_T

**推导过程：**

```
Step 1: 单步扩散
  x_t = √(1-β_t) * x_{t-1} + √β_t * ε_t
  其中 ε_t ~ N(0, I)

Step 2: 递推展开
  x_t = √(1-β_t) * x_{t-1} + √β_t * ε_t
      = √(1-β_t) * √(1-β_{t-1}) * x_{t-2} + ...
      = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

  其中 ᾱ_t = ∏_{s=1}^{t} (1-β_s)

Step 3: 闭式解
  x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
  其中 ε ~ N(0, I)
```

**关键洞察：**
- ᾱ_t 是"信号保留比例"：t 小信号多，t→T 信号趋零
- 闭式解让训练时一步到位，不必迭代 t 次
- 这是扩散模型能高效训练的全部秘密

## 代码实现

### 1. 前向：一条"固定的"噪声马尔可夫链

运行 [scripts/01_ddpm_from_scratch.py](../scripts/01_ddpm_from_scratch.py) 验证以下代码。

DDPM（2006.11239）定义前向过程 q：逐步给数据加高斯噪声，β_t 是每步的噪声量：

```
q(x_t | x_{t-1}) = N(x_t; √(1−β_t)·x_{t-1}, β_t·I)
```

关键推导：代入展开后，任意时刻 t 的**边际分布有闭式解**：

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1−ᾱ_t)·I)     # ᾱ_t = ∏ α_s（α=1−β）
```

### 形状追踪：扩散过程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DDPM 扩散过程                                                              │
│                                                                             │
│  前向扩散（训练时）:                                                         │
│  x_0: (B, 2)  # 原始数据                                                    │
│    ↓ 加噪                                                                  │
│  x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε                                       │
│  x_t: (B, 2)  # 噪声数据                                                   │
│                                                                             │
│  反向去噪（采样时）:                                                         │
│  x_T: (B, 2)  # 纯噪声                                                     │
│    ↓ 去噪                                                                  │
│  x_{t-1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε̂) + √β_t · z   # t=0 时 z=0      │
│  x_0: (B, 2)  # 生成数据                                                   │
│                                                                             │
│  训练目标:                                                                   │
│  ε_θ(x_t, t) ≈ ε  # 预测加进去的噪声                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. 训练：预测噪声 ε（VLB 的简化形式）

DDPM 论文 §3.2 把变分下界简化成一个 MSE：**让网络从 (x_t, t) 预测加进去的噪声 ε**：

```python
t = torch.randint(0, T, (B,))               # 随机采 t（闭式让这步 O(1)）
noise = torch.randn_like(x0)
x_t = q_sample(x0, t, noise)                # √ᾱ_t·x0 + √(1−ᾱ_t)·noise
eps_pred = model(x_t, t)
loss = F.mse_loss(eps_pred, noise)          # 训练目标：还原"加进去的噪声"
```

> 📝 **签名说明**：脚本 01 与本节的 `q_sample(x0, t, noise)` 为省参用模块级
> `alphas_cumprod`；章末练习与 Assignment 16 改为显式传参的四参版
> `q_sample(x0, alphas_cumprod, t, noise)`——二者数学完全相同。

**实测（2D 双月环玩具，3000 步；RTX 4090, torch 2.6.0+cu124）**：denoising loss 1.11 → 0.21 后稳定
（中途在 0.19-0.23 间小幅波动，属随机采 t 的正常现象）。

### 3. 采样：反向链（论文式 11）

```
x_T ~ N(0, I)
for t in T−1 … 0:
    ε̂ = model(x_t, t)
    x_{t−1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε̂) + √β_t · z    # t=0 时 z=0
```

直觉：ε̂ 给出"这坨噪声里藏着什么内容"的方向，每步减掉一点、加回一点随机性。
**实测（RTX 4090, torch 2.6.0+cu124）**：采样 2000 点与真实分布对比——均值偏移
[0.069, −0.011]（≈0）、方差比 [1.039, 1.065]（≈1）——分布匹配 ✅。

## 工程实践

### 调试展示：常见错误与修复

#### 错误 1：β schedule 选择不当

**症状：**
```
训练不稳定，或生成质量差
```

**原因：** β schedule 太激进（末端噪声过猛）

**解法：**
```python
# 线性 schedule（简单但末端噪声过猛）
beta = torch.linspace(0.0001, 0.02, T)

# cosine schedule（更平滑，推荐）
def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0, 0.999)
```

#### 错误 2：t 未归一化直接进网络

**症状：**
```
训练 loss 居高不下（甚至 NaN），采样输出仍是一团噪声
```

**原因：** t 是 0~T−1 的整数（本脚本 T=400），量级远大于网络输入 x_t（±1 量级）——
原始 t 直接进 MLP 会淹没 x_t 的信号，时间条件基本没学到，网络分不清噪声级别

**解法：**
```python
# 先转 float 再除以 T，把 t 压到 0~1（脚本 01 的写法）
temb = self.t_mlp(t.view(-1, 1).float() / T)
# 真实实现用 sinusoidal 位置编码或 AdaLN 注入 t——思想相同：先编码、再进网络
```

#### 错误 3：t=0 时加噪

**症状：**
```
生成结果有噪声
```

**原因：** t=0 时不应该加噪

**解法：**
```python
# 采样时 t=0 不加噪
for t in range(T - 1, -1, -1):
    eps_pred = model(x_t, t)
    if t > 0:
        x_t = (x_t - beta[t] / sqrt(1 - alpha_bar[t]) * eps_pred) / sqrt(alpha[t])
        x_t = x_t + sqrt(beta[t]) * torch.randn_like(x_t)
    else:
        x_t = (x_t - beta[t] / sqrt(1 - alpha_bar[t]) * eps_pred) / sqrt(alpha[t])
```

### 性能数据（实测参考）

| 方法 | 训练步数 | 生成质量 | 训练时间 | 说明 |
|------|----------|----------|----------|------|
| DDPM (线性) | 3000 | 良好 | <1min | 2D 玩具 |
| DDPM (cosine) | 3000 | 更好 | <1min | 2D 玩具 |
| DDPM (真实图像) | 100K+ | 高 | 数小时 | 512×512 |

> 📊 数据来源：DDPM 论文 + 本课开发机实测

### 常见陷阱

#### 陷阱 1：β schedule 选择不当

**症状：** 训练不稳定，或生成质量差

**原因：** β schedule 太激进

**解法：** 使用 cosine schedule

#### 陷阱 2：方差口径不一致

**症状：** 训练 loss 不下降

**原因：** 方差用无偏口径

**解法：** 使用有偏口径

#### 陷阱 3：t=0 时加噪

**症状：** 生成结果有噪声

**原因：** t=0 时不应该加噪

**解法：** 采样时 t=0 不加噪

### 最佳实践

#### 配置推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| T | 1000 | 扩散步数 |
| β schedule | cosine | 更平滑 |
| 学习率 | 1e-4 | Adam 优化器 |
| batch_size | 64-256 | 根据显存调整 |

## 学完本部分你能...

- ✅ 写出前向闭式并解释 ᾱ_t 的物理含义（信号保留比例）
- ✅ 解释训练目标为什么是"预测噪声"而不是"预测 x₀"（等价但更稳）
- ✅ 手写完整的采样循环（含 t=0 不加噪的细节）
- ✅ 回答"为什么扩散训练能一步加噪"（固定高斯链的边际闭式解）
- ✅ 识别 β schedule 选择、方差口径等常见陷阱

## 🤔 概念检验

<details>
<summary>Q1: 训练时为什么不迭代 t 次逐步加噪，而是直接采 t 用闭式？</summary>

A: 前向过程没有可学习参数（固定的高斯链），任意 t 的边际有闭式解 → 一步到位。
这让每个训练样本每个 step 都能随机覆盖所有噪声级别，T=1000 也零额外成本。
反向过程才有可学习参数（ε̂ 网络），那才是需要迭代的部分（采样时）。

</details>

<details>
<summary>Q2: β schedule（线性 vs cosine）影响什么？</summary>

A: 决定 ᾱ_t 从 1 降到 0 的速度分布。线性 schedule 在高分辨率下末端噪声过猛
（信息丢失过快），cosine（Nichol & Dhariwal 2021）让 ᾱ 更平滑地衰减，
对训练更稳。本课用线性是为了教学直观。

</details>

<details>
<summary>Q3: 为什么训练目标是"预测噪声"而不是"预测 x₀"？</summary>

A: 两者数学等价，但预测噪声更稳定：
- 预测 x₀：需要预测完整的数据，方差大
- 预测 ε：只需要预测噪声，方差小
- 实验表明：预测 ε 的训练更稳定，生成质量更高

</details>

## 🔧 动手实践

<details>
<summary>练习 1: 实现前向扩散</summary>

**任务：** 实现一个函数，计算前向扩散的闭式解（签名与 Assignment 16 题 1 完全一致）。

**验收标准：**
- [ ] 输入：x0 (B, 2), alphas_cumprod (T,), t (B,) 长整型, noise (B, 2)
- [ ] 输出：x_t (B, 2)，逐行满足闭式 x_t[i] = √ᾱ_{t[i]}·x0[i] + √(1−ᾱ_{t[i]})·noise[i]
- [ ] 使用闭式解一步到位（不迭代加噪）

**步骤提示：**
```python
def q_sample(x0, alphas_cumprod, t, noise):
    """
    Steps:
        1. 取 s = alphas_cumprod[t]，reshape 成 (-1, 1) 以便按行广播
        2. 信号项 s.sqrt() * x0，噪声项 (1 - s).sqrt() * noise
        3. 返回 x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * noise
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 2: 实现采样循环</summary>

**任务：** 实现一个函数，从噪声生成数据。

**验收标准：**
- [ ] 输入：模型、T、alpha、beta、alpha_bar
- [ ] 输出：生成的数据 x_0
- [ ] 正确处理 t=0 时不加噪

**步骤提示：**
```python
def p_sample_loop(model, T, alpha, beta, alpha_bar):
    """
    Steps:
        1. 从纯噪声开始 x_T ~ N(0, I)
        2. 从 T-1 到 0 循环
        3. 预测噪声 eps_pred = model(x_t, t)
        4. 计算 x_{t-1}
        5. t>0 时加噪，t=0 时不加噪
        6. 返回 x_0
    """
    # TODO: Implement
    pass
```

</details>

<details>
<summary>练习 3: 实现 cosine schedule</summary>

**任务：** 实现 cosine β schedule。

**验收标准：**
- [ ] 输入：T（扩散步数）
- [ ] 输出：beta (T,)
- [ ] 使用 cosine 函数

**步骤提示：**
```python
def cosine_schedule(T, s=0.008):
    """
    Steps:
        1. 计算 steps = torch.arange(T + 1)
        2. 计算 f = cos((steps / T + s) / (1 + s) * pi / 2) ** 2
        3. 计算 alphas_cumprod = f / f[0]
        4. 计算 betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        5. 返回 betas
    """
    # TODO: Implement
    pass
```

</details>

## 📝 课后作业

完成本章后，去 Assignment 16 完成练习：

👉 [Assignment 16](../../../assignments/assignment_16/)

## 下一步

会"从噪声生成分布"了。真实文生图 = 把这套数学搬进 VAE 潜空间 + 用 cross-attention
注入文本条件（Latent Diffusion）——以及 img2img 的 strength 数学。

👉 [02 — 文生图与图生图工具链](02_t2i_i2i_pipelines.md)
