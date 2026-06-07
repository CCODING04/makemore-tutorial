# Part 3: 训练诊断与 BatchNorm — 让训练稳定起来

> 🏥 模型训练不收敛？Loss 下不去？这一课教你给网络做体检！

## 📚 章节导航

| 序号 | 章节 | 内容 |
|------|------|------|
| 01 | [训练诊断：你的模型生病了](01_diagnosis.md) | 初始 Loss 异常、tanh 饱和、Kaiming 初始化 |
| 02 | [BatchNorm：深度学习的维生素](02_batchnorm.md) | BN 原理、从零实现、常见陷阱 |
| 03 | [深层网络与诊断工具](03_deep_network.md) | 多层网络搭建、4 种诊断工具、更新/数据比率 |

## 🗺️ 学习路线图

```
Part 2 (MLP)
    │
    │  "MLP 训练起来了，但总觉得哪里不对劲..."
    ▼
┌─────────────────────────────────┐
│  Part 3: 训练诊断与 BatchNorm   │
│                                 │
│  ① 给模型做体检 — 找到病因     │──→ 01_diagnosis.md
│  ② 开处方 — BatchNorm          │──→ 02_batchnorm.md
│  ③ 深层网络 + 诊断工具箱       │──→ 03_deep_network.md
│                                 │
└──────────────┬──────────────────┘
               │
               │  "网络稳定了，想深入理解反向传播..."
               ▼
          Part 4 (手动反向传播)
```

## 🎯 学完这一部分你能...

- ✅ 诊断**初始 Loss 异常** —— 为什么第一轮 loss 就该 ≈ 3.29
- ✅ 理解 **tanh 饱和**（梯度消失）的原因和修复方法
- ✅ 掌握 **Kaiming 初始化** —— 让每一层的激活值保持合理范围
- ✅ 从零实现 **BatchNorm**，理解训练/评估两种模式
- ✅ 搭建**深层网络**，用 4 种诊断工具监控训练健康度

## 📝 课后作业

完成教程后，去这里做练习：

👉 [Assignment 3](../assignment_3/)

## 🔗 相关资源

- 📺 Andrej Karpathy 原视频：[Building makemore Part 3: Activations & Gradients](https://www.youtube.com/watch?v=P6sfmUTpUmc)
- 📄 Ioffe & Szegedy 2015 论文：[Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
- 📄 He et al. 2015 论文：[Delving Deep into Rectifiers (Kaiming Init)](https://arxiv.org/abs/1502.01852)

---

[← 上一章：Part 2 MLP](../../Part2_mlp/tutorial/README.md) | [下一章：Part 4 Backpropagation →](../../Part4_backprop/tutorial/README.md)
