# 🌱 Makemore 中文教程

> 基于 Andrej Karpathy 的 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列
> 从零构建字符级语言模型，逐步深入神经网络核心概念

---

## 📖 学习路线

```
Part 1: Bigrams          ─── 最简单的语言模型（频率计数 → 概率 → 神经网络）
  ↓
Part 2: MLP              ─── 多层感知机（Embedding + 隐藏层 + 反向传播）
  ↓
Part 3: BatchNorm        ─── 训练诊断与优化（初始化 + BN + 深层网络）
  ↓
Part 4: Backpropagation  ─── 手动反向传播（逐层推导梯度，理解 autograd 原理）
  ↓
Part 5: WaveNet          ─── 层次化架构（PyTorch 化代码 + WaveNet + 卷积预览）
```

| Part | 主题 | 核心概念 | 教程入口 | 原始视频 |
|------|------|----------|----------|----------|
| 1 | Bigrams | 频率矩阵、Softmax、NLL 损失、梯度下降 | [📖 开始学习](courses/Part1_bigrams/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| 2 | MLP | Embedding、多层感知机、Minibatch SGD、Train/Dev/Test | [📖 开始学习](courses/Part2_mlp/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I) |
| 3 | BatchNorm | 激活诊断、Kaiming 初始化、BatchNorm、诊断工具 | [📖 开始学习](courses/Part3_batchnorm/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc) |
| 4 | Backpropagation | 链式法则、手动梯度、CrossEntropy 反传、BN 反传 | [📖 开始学习](courses/Part4_backprop/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=q8SA3rM6ckI) |
| 5 | WaveNet | Sequential 容器、层次融合、FlattenConsecutive、卷积 | [📖 开始学习](courses/Part5_wavenet/tutorial/README.md) | [YouTube](https://www.youtube.com/watch?v=t3YJ5hKiMQ0) |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

### 安装

```bash
# 使用 uv（推荐）
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pip
pip install -r requirements.txt
```

### 数据

数据集 `names.txt`（32,032 个美国人名）已包含在 `data/` 目录。

### 学习方式

每课包含三个部分：

1. **📖 Tutorial** — `courses/PartX/tutorial/` — 中文讲解，概念 + 代码 + 配图
2. **💻 Scripts** — `courses/PartX/scripts/` — 渐进式可运行代码，跟着教程一步步跑
3. **📝 Assignment** — `assignments/assignment_X/` — 练习题，带自动测试

```bash
# 示例：运行 Part 1 的第一个脚本
cd courses/Part1_bigrams/scripts
python 01_explore_data.py
```

---

## 📝 如何做作业

每个 Part 学完后都有配套作业来巩固理解，流程如下：

1. **学完教程** — 完成某个 Part 的 tutorial 阅读和 scripts 运行
2. **进入作业目录** — 对应 `assignments/assignment_X/`（X = Part 编号）
3. **阅读 README.md** — 了解作业要求和背景说明
4. **完成练习** — 在 `xxx_exercises.py` 中找到 `TODO` 标记，填写你的实现
5. **验证答案** — 运行 `test_xxx.py`，确保所有测试通过

```bash
# 示例：完成 Part 1 作业
cd assignments/assignment_1
cat README.md                  # 阅读作业要求
vim bigram_exercises.py        # 完成 TODO 部分
python test_bigram_exercises.py  # 运行测试验证
```

---

## 📁 项目结构

```
makemore-tutorial/
├── README.md                 # 本文件
├── data/
│   └── names.txt             # 训练数据（32K 人名）
├── tools/
│   └── extract_images.py     # 从 notebook 提取图片的工具
├── skills/
│   └── tutorial-creator/     # 用于创建此类教程的 Agent Skill（见下方说明）
├── courses/
│   ├── Part1_bigrams/
│   │   ├── makemore_part1_bigrams.ipynb  # 原始 notebook
│   │   ├── images/                       # 教程配图
│   │   ├── scripts/                      # 可运行脚本
│   │   └── tutorial/                     # 中文教程（从 README.md 开始）
│   ├── Part2_mlp/
│   ├── Part3_batchnorm/
│   ├── Part4_backprop/
│   └── Part5_wavenet/
└── assignments/
    ├── assignment_1/         # Part 1 作业
    │   ├── README.md         # 作业说明
    │   ├── bigram_exercises.py   # TODO 骨架
    │   └── test_bigram_exercises.py  # 自动测试
    ├── assignment_2/
    ├── assignment_3/
    ├── assignment_4/
    └── assignment_5/
```

---

## 🤖 skills/tutorial-creator

`skills/` 目录包含用于**自动生成此类教程**的 Agent Skill：

- **tutorial-creator** — 一个 OpenClaw skill，能从 Jupyter Notebook 自动生成结构化的中文教程（tutorial Markdown）、渐进式脚本（scripts）和带自动测试的作业（assignments）
- 适用于想要复刻此教程模式来覆盖其他课程或项目的场景
- 详见 `skills/tutorial-creator/SKILL.md`

---

## 🙏 致谢

- [Andrej Karpathy](https://karpathy.ai/) — 原始课程和代码
- [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — 原始仓库
- [makemore](https://github.com/karpathy/makemore) — makemore 项目仓库

---

## 📄 License

本教程的中文讲解内容为原创，代码部分遵循原始仓库的 MIT License。
