# Makemore 教程制作流程手册

> 基于 Part 1 (Bigrams) 的制作经验总结，作为 Part 2-5 的标准流程。

---

## 总览

每课制作分为 3 个并行子任务 + 1 个验证阶段：

```
Phase A: 并行制作（3 个子代理）
├── 子任务1: Scripts（渐进式 Python 脚本）
├── 子任务2: Tutorial（中文教程 Markdown）
└── 子任务3: Assignment（作业 + 测试）

Phase B: 验证与修复
├── 跑通所有 scripts
├── 跑通 assignment tests
├── 检查 tutorial 引用路径
└── 检查数据路径
```

---

## Phase A: 子任务 1 — Scripts

### 输入材料
- 原始 notebook: `courses/PartX/xxxx.ipynb`
- 字幕文件: `~/Code/makemore/transcript/partX_xxx.txt`

### 拆分原则
1. 每个脚本 **独立可运行**（包含 import 和数据加载）
2. 后一个在前一个基础上 **演进**（不是从零重写）
3. 命名：`01_描述.py`、`02_描述.py`，按课程推进编号
4. 每课 5-7 个脚本

### 数据路径
scripts 在 `courses/PartX/scripts/` 下，到 data 需要 3 级：
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')
```

### 每个脚本结构模板
```python
#!/usr/bin/env python3
"""
Part X - 脚本 N: 简短描述
目标：一句话说清楚这个脚本做什么
"""

import torch
import torch.nn.functional as F
import os

def main():
    # 数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')
    
    # 加载数据
    words = open(data_path, 'r').read().splitlines()
    
    # === 本脚本核心内容 ===
    ...

if __name__ == '__main__':
    main()
```

### 验证方式
- 每个脚本直接 `python3 <path>` 运行
- matplotlib 脚本需设 `MPLBACKEND=Agg`

---

## Phase A: 子任务 2 — Tutorial

### 输入材料
- 字幕文件（主要参考）
- 原始 notebook（代码参考）
- 网络搜索（交叉验证模糊概念）

### 拆分规则
- 内容 < 3000 行：单文件 `tutorial.md`
- 内容较长：拆成 2-4 个文件 + `README.md` 做导航
- Part 1 经验：拆成 3 个文件比较合适（intro / 统计模型 / 神经网络）

### 引用路径
- 脚本引用：`../scripts/01_xxx.py`
- 图片引用：`../images/xxx.png`
- 作业链接：`../../../assignments/assignment_X/`

### 写作风格
- 中文，口语化但准确
- emoji 标注：💡 重点、⚠️ 坑点、🔑 关键概念
- 代码块用 `python` 语法高亮
- ASCII art 画流程图（优先于 mermaid）
- 课后练习用 `<details>` 折叠

### 教程标准模板
```markdown
# Part X: 标题

## 前置知识
## 课程预告
## 课程主体（分 3-5 个部分）
### 第一部分
...（引用 scripts 代码 + images 图片）
**课后练习（QA 形式）**
## 课后作业（链接到 assignment）
## 下一课预告
```

---

## Phase A: 子任务 3 — Assignment

### 设计原则
- 以 Karpathy 原始练习为主，加入拓展思考题
- 4 基础题 + 1 拓展题
- 固定随机种子确保可测试

### 文件结构
```
assignment_X/
├── assignment.md          # 题目说明
├── xxx_exercises.py       # TODO 骨架
└── test_xxx_exercises.py  # 自动测试
```

### 数据路径
assignment 在 `assignments/assignment_X/` 下，到 data 需要 2 级：
```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')
```

### 测试策略
- 测 **shape、dtype、数学性质（行和=1）、合理范围**，不测精确数值
- 拓展题未实现（返回 None）时优雅跳过
- 每个测试函数独立，一个失败不影响其他

---

## Phase B: 验证清单

### 1. 文件完整性
```bash
ls courses/PartX/scripts/    # 5-7 个 .py
ls courses/PartX/tutorial/   # 3-5 个 .md（含 README）
ls courses/PartX/images/     # notebook 提取的图片
ls assignments/assignment_X/ # 3 个文件
```

### 2. Scripts 运行
```bash
for s in courses/PartX/scripts/*.py; do
  python3 "$s" && echo "✅ $s" || echo "❌ $s"
done
```

### 3. Tests 运行
```bash
python3 assignments/assignment_X/test_xxx.py
```

### 4. 路径检查
- scripts 数据路径：`os.path.join(script_dir, '..', '..', '..', 'data', 'names.txt')`
- test 数据路径：`os.path.join(_THIS_DIR, '..', '..', 'data', 'names.txt')`
- tutorial 脚本引用：`../scripts/xx.py`
- tutorial 图片引用：`../images/xx.png`

---

## 时间参考（基于 Part 1 实测）

| 步骤 | 耗时 |
|------|------|
| 3 个子代理并行 | ~8 分钟 |
| 验证与修复 | ~5 分钟 |
| 写经验文档 | ~5 分钟 |
| **每课总计** | **~20 分钟** |

---

## Part 1 制作中发现的问题与修复

### 问题 1：数据路径
**现象**：exec 执行 python 时 cwd 不在脚本目录，相对路径 `../../data/names.txt` 找不到
**修复**：所有路径用 `os.path.dirname(os.path.abspath(__file__))` 解析

### 问题 2：matplotlib 中文字体
**现象**：matplotlib 默认不支持中文标题
**修复**：图表标题用英文，教程正文用中文

### 问题 3：TODO 函数返回 None
**现象**：test 跑 assignment 骨架时，未实现的函数返回 None
**修复**：这是预期行为，测试脚本需要对 None 做容错处理

### 问题 4：子代理数据路径理解
**现象**：scripts 子代理把数据路径写成了 `../../../data/names.txt`（4级），实际应该是 3 级
**修复**：子代理自行修正了路径并在脚本中验证运行通过
