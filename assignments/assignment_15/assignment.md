# Assignment 15：多模态理解（VLM）

> 对应 Part 15 教程（[01 手写四件套](../../courses/Part15_vision_language/tutorial/01_handwritten_projection_vlm.md) / [02 三大方案与对齐损失](../../courses/Part15_vision_language/tutorial/02_alignment_losses_and_schemes.md)）。
> 四题纯 CPU 可完成；题 4 为 🌟 stretch（不实现则测试优雅 SKIP ⏭️，不算失败）。

## 题目（实现 `vlm_exercises.py` 后 `python test_vlm_exercises.py`）

### 题 1：patch 数量与形状（25 分）

给定图像高宽与 patch 大小，计算 ViT 的视觉 token 数与输出形状。

**函数签名：**
```python
def patch_tokens(h, w, patch): ...
def vit_out_shape(n_tokens, embed_dim, n_layers=1): ...
```

**步骤提示：**
- `patch_tokens`：token 数 = `ceil(H/p) × ceil(W/p)`（`math.ceil`，非整除时向上取整、边缘补齐）
- `vit_out_shape`：ViT 堆多少层都不改变 token 数与维度，返回**元组** `(n_tokens, embed_dim)`

**验收标准：**
- [ ] `patch_tokens(224, 224, 14) == 256`（LLaVA/CLIP 的 224²，16×16）
- [ ] `patch_tokens(336, 336, 14) == 576`（LLaVA-1.5 的 336²）
- [ ] `patch_tokens(1024, 768, 14) == 4070`（非正方形：74×55）
- [ ] `vit_out_shape(576, 1024) == (576, 1024)`，返回元组而非 list

### 题 2：InfoNCE 对比损失（30 分）

实现 CLIP 的对称双方向 InfoNCE 损失（与脚本 02、教程 02 章练习 1 同名同签名）。

**函数签名：**
```python
def infonce_loss(f_img, f_txt, scale): ...
```

**步骤提示：**
1. 相似度矩阵：`logits = scale * f_img @ f_txt.T`，shape `(N, N)`
2. 标签是对角线：`labels = torch.arange(N)`
3. 对称损失：`loss = 0.5 * (CE(logits, labels) + CE(logits.T, labels))`（图→文 + 文→图 两个方向取平均）

**验收标准：**
- [ ] 输入归一化特征 `f_img (N, d)`、`f_txt (N, d)` 与标量 `scale`，输出正标量 loss
- [ ] 与手动对称实现 `torch.allclose(loss, manual, atol=1e-6)`
- [ ] 完美配对时（两塔输出相同的单位正交阵）`loss < 1e-3`
- [ ] 两个方向都算（只算单向会差一个方向的结果）

### 题 3：投影器参数量（25 分）

算 LLaVA `mlp2x_gelu` 投影器的参数账（对照 Part 8 08 章 LoRA 的 3.4%）。

**函数签名：**
```python
def mlp2x_params(vision_dim, llm_dim): ...
```

**步骤提示：**
- 结构是 `Linear(vision_dim → llm_dim)` + `Linear(llm_dim → llm_dim)`，**两层都含 bias**
- 每个 Linear 参数 = `in×out + out`（权重 + 偏置）

**验收标准：**
- [ ] `mlp2x_params(1024, 4096) == (1024*4096+4096) + (4096*4096+4096) == 21,043,712`
- [ ] 返回 `int`（不是 tensor）
- [ ] 对照结论：脚本 01 的玩具版投影器只有 1,856 参数——LLaVA 7B 约 20M，仍只占全模型的 ~0.3%

### 题 4（🌟 stretch）：动态分辨率 token 估算（20 分）

给定原图与 tile/压缩策略，估算 Qwen 式打包的视觉 token 数（含 token 预算控制）。

**函数签名：**
```python
def dynamic_tokens(h, w, patch=14, compress=4, max_tokens=2560): ...
```

**步骤提示：**
1. `raw = ceil(h/patch) * ceil(w/patch)`（patch 网格）
2. `tokens = raw // compress`（pixel shuffle 类压缩，整除取 floor）
3. 若 `tokens > max_tokens`：`h, w = int(h*0.8), int(w*0.8)` 后回到第 1 步重算
4. 返回 `int`（最终不超过 `max_tokens`）

**验收标准：**
- [ ] `dynamic_tokens(1024, 768) == 1017`（4070//4，未触发预算）
- [ ] `dynamic_tokens(4096, 3072)` 满足 `0 < v <= 2560`（触发预算、逐级缩放）
- [ ] 未实现（返回 `None`）时测试优雅 SKIP ⏭️

## 实验题（观测型）

- 跑脚本 01，把 Stage 1 的 lr 从 3e-3 调到 3e-2，观察"对齐被冲垮"的现象（呼应思考题 Q3）
- 跑脚本 02，把 SigLIP 的 batch 从 32 减到 8，对比 InfoNCE 在同 batch 下的表现

## 🎯 面试直通车

- "三大多模态方案？"——拼接式（主流）/ Flamingo 门控（历史）/ early-fusion（原生，下一代）
- "LLaVA 两阶段为什么这么设计？"——先训翻译器防冲垮、再端到端（1,856 参数→全参的实证）
- "CLIP vs SigLIP？"——softmax 对比 vs 逐对 sigmoid；batch 依赖性；τ 可学习
- "Qwen-VL 为什么 OCR 强？"——原生动态分辨率，token 数随内容自适应（非固定 336²）

## 🤔 思考题

**Q1：LLaVA Stage 1 为什么冻结 ViT 和 LLM，只训投影器？如果一开始就端到端全解冻会怎样？**

<details>
<summary>💡 参考答案</summary>

投影器随机初始化时，输出的"视觉 token"对 LLM 来说就是噪声。若此时同时更新三个模块：

1. **LLM 侧**：大量噪声 token 流入，会把 LLM 推离预训练分布（灾难性遗忘）——
   预训练学到的语言能力反而被破坏；
2. **ViT 侧**：ViT 的 CLIP 对齐（与文本空间的几何结构）也会被随机梯度冲散；
3. **对齐信号本身**：三个模块同时漂移，损失下降不保证是"对齐变好"，可能只是互相迁就。

先冻结两端只训投影器 = 在两个固定的"语言"之间搭一座桥；桥稳了（Stage 2）再端到端微调，
此时梯度是在一个已经对齐的初始点附近做小幅修正。这也是脚本 01 的实测设置：
Stage 1 只有 1,856 个可训练参数。

</details>

**Q2：InfoNCE 的负样本从哪来？为什么说它"batch 依赖强"，而 SigLIP 不受此限制？**

<details>
<summary>💡 参考答案</summary>

InfoNCE 的分母是 `Σ_j exp(sim(i, j))`——**batch 内所有其他文本都充当负样本**。
softmax 是全局归一化：batch=4 时每个正样本只有 3 个负例，对比信号很弱；
CLIP 论文用 32768 的 batch 就是为了负例足够多、足够难。

SigLIP 把 N×N 的 softmax 分解成 N² 个**独立的二分类**（sigmoid），每一对
"+1/-1" 的监督不依赖其他样本的相对大小——因此小 batch 也能形成有效梯度。
论文实测 batch 缩到 1/4 仍能与 InfoNCE 持平。这是 SmolVLM/InternVL/PaliGemma
改用它的直接原因（省显存）。

</details>

**Q3：把脚本 01 Stage 1 的学习率从 3e-3 调到 3e-2，预计会发生什么？这与"只训投影器"的设计矛盾吗？**

<details>
<summary>💡 参考答案</summary>

大概率观察到：早期 loss 剧烈震荡或先降后反弹，最终 loss 高于 3e-3 的版本
（"对齐被冲垮"）。原因：投影器参数极少（1,856），大学习率让它在 LLM 的
固定输入流形上反复"过冲"，跨过最优点来回震荡，无法稳定收敛到对齐解。

不矛盾，反而是同一课的两面：**冻结两端解决了"谁动"的问题，学习率解决"动多快"**。
两端冻结使得大 lr 也不会破坏 LLM/ViT 本身（最坏只是投影器没训好）；
但如果 lr 大到投影器都震荡，Stage 2 端到端阶段的大 lr 会连预训练能力一起毁掉。
这就是为什么 LLaVA Stage 2 用 2e-5 量级的小学习率。

</details>

**Q4：CLIP 为什么把温度实现为 `scale = exp(log_scale)` 的可学习参数，而不是直接学 τ（或直接固定 τ）？**

<details>
<summary>💡 参考答案</summary>

两个原因：

1. **正约束**：softmax 要求缩放因子为正。直接优化 τ 需要投影/裁剪才能保证 τ>0，
   而 `exp(·)` 天然把无约束的实数映射到正数——优化器可以放心用 AdamW。
2. **自适应锐度**：训练早期需要温和的分布（τ 大）避免梯度爆炸，后期需要锐利
   的分布（τ 小）拉开配对与非配对的差距。可学习的 τ 让模型自己调度这个过程。

脚本 02 实测：训练后 CLIP 路径学到 τ≈16、SigLIP≈8.5——都显著偏离初始化，
说明"学出来的锐度"和初始值不同。固定 τ（如 CLIP 论文的 0.07）在数据分布
变化时需要重新调参，可学习版本更鲁棒。

</details>

**Q5：Qwen-VL 的"原生动态分辨率"为什么对 OCR/文档类任务提升特别大？固定 336² 的方案损失了什么？**

<details>
<summary>💡 参考答案</summary>

固定分辨率（LLaVA 的 336²）对一张 4000×3000 的文档照片意味着**下采样 100 倍以上**，
文字笔画直接糊掉——信息在进入模型之前就被 resize 毁了，后端再强也无法恢复。
对小图则相反会被强行放大（引入插值伪影）。

动态分辨率按原始尺寸切 patch 打包，token 数随内容自适应：
- 文字边缘保持原像素级的锐度，OCR 的召回上限大幅提高；
- 代价是 token 数不可预测，所以必须配 **token 预算控制**（题 4 的 `max_tokens`）
  和 **token 压缩**（pixel shuffle ÷4、Q-Former 可学习压缩），否则一张长图
  会吃掉整个上下文窗口。

一句话：固定分辨率是"让图迁就模型"，动态分辨率是"让模型迁就图"——
后者保留了信息，代价是需要一套 token 预算工程。

</details>
