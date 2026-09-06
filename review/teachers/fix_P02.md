# 修改说明 · Part 2（MLP）整改（T2）

> 整改人：T2 · 日期：2026-09-05
> 输入：plan_P02（T1 预审）+ S1/S2/S3 三份 report_P02 + 必修清单（T0 确认 13 项）
> 原则：REPO 只读；全部修改写镜像；默认档脚本行为与输出逐位不变；无 wontfix，4 项 disputed

## 一、镜像修改文件清单（14 个）

**教程（4）** `REVIEW/courses/Part2_mlp/tutorial/`
- `README.md` — 导航 03 行补"Embedding 可视化"（C1-0）；新增「🛠️ 学习方式」节（STEPS 短程档 + LR_SEARCH 用法）
- `01_introduction.md` — 学习目标补章节指引（C1-1）；新增「字符映射 stoi/itos」小节（C1-2，统一写法）；给出与脚本同签名的 `build_dataset`（C1-3）；X[:5] 配文修正；划分节补"名字数→样本数 228146"衔接（C1-4）
- `02_mlp_architecture.md` — **view(-1)→(192,) 修正**+拍扁顺序说明；**"零拷贝"改"返回新张量"**+one-hot 形状链 (N,3,27)@(27,2)→(N,3,2)；**CE 补 LaTeX 公式 + 3 类 toy 数值例**（手算 0.170 ≡ F.cross_entropy 0.1698）；**新增「初始 loss sanity check」小节**（ln27≈3.2958 / 全零恰为 ln27 / randn≈19.5 归因）；**新增「嵌入梯度稀疏」面试点小节**（5 行实验）；C/W1/b1/W2/b2 补 requires_grad=True + parameters 容器（C1-5/10）；裸代码块加 `text` 语言标注（G2）；keepdim/广播注释
- `03_training_and_eval.md` — **Q3 重写**为三步诊断（绝对值→性质→行动规则），消除"过拟合却增大模型"矛盾；lr 搜索：linspace 注释纠错、补"为什么指数空间"、plt 前置导入、**悬空引用改为 05 的 LR_SEARCH 档**（C1-9）；**loss 曲线图移入本章**（C1-6）+ 训练循环补 stepi/lossi；新增「本章实测数字」框（step0 18.2、train 2.3749/dev 2.3710/test 2.3725）；诊断三情况加"示意值"声明；可视化"元音聚簇"改口（u 例外）；采样种子说明 + **示例输出换实测**（junide/janasar/prafay/adin/koi）+ 注明 mora/kiah/mel 出处；"一个 epoch 228146 次"措辞修正；顶部补学习方式提示

**脚本（7）** `REVIEW/courses/Part2_mlp/scripts/`
- `05_minibatch_training.py` — `MAX_STEPS=int(os.environ.get('STEPS','20000'))` + print flush；新增 LR_SEARCH=1 门控学习率搜索段（默认关闭）；docstring 注明冒烟用法
- `06_visualize_embedding.py` / `07_sampling.py` — 同 STEPS+flush；**07 采样种子 2147483647+10 → 2147483647（与教程统一）**
- `01`~`04` — 字符映射写法统一为 Part1/作业风格（`stoi={s:i+1...}; stoi['.']=0`；01 的字符集打印改 `'.'+...` 保持输出不变）；02 的 build_dataset 去掉未用 itos 参，与 03-07/教程签名统一

**作业（3）** `REVIEW/assignments/assignment_2/`
- `mlp_exercises.py` — 题5 初始化注释改 `(randn*scale).requires_grad_(True)` 并加非叶子张量警示（原写法实测 TypeError）
- `README.md` — 文件结构 self-reference 改 README.md；题5 签名补 seed；阈值统一（完整 200k 步 <2.3 / 测试 1000 步 <2.5）；提示代码补 build_dataset、初始化缩放警告、mini-batch ix 定义
- `test_mlp_exercises.py` — test_tuning 补 `p.grad is not None` 回归断言（G9），阈值注释与 README 对齐

**台账** `REVIEW/ledger/ledger_P02.md`（29 条 fixed + 4 条 disputed）

## 二、验证记录（全部实跑，scratch/t2_P2/）

1. **脚本 01-04**：exit=0；01 映射 `z->26, '.'->0`、02 划分 25626/3203/3204→182625/22655/22866、03 `'.'-> [1.5674, -0.2373]`×3、**04 初始 loss=19.5116**——与教程引用逐项一致，映射统一未改变任何输出。
2. **默认档不变性**：修改版 05（STEPS=2000）与原版 05（sed 2000）输出逐位一致：final 2.5491 / train 2.5999 / dev 2.5985 / test 2.6021。
3. **短程档**：05 STEPS=2000（1.2s）、06 STEPS=500（存 png）、07 STEPS=2000、05 LR_SEARCH=1 STEPS=200（搜索段运行并出 lr_search.png）全部通过。
4. **全量档**：05/07 默认 20000 步完整跑通（run05_full.log、run07_full.log）：train 2.3749 / dev 2.3710 / test 2.3725；07 采样 junide/janasar/prafay/adin/koi/...——即教程新示例输出。
5. **作业 pytest**：修复后 test + 参考答案实现 → **5 passed**；旧 bug 写法（`randn(...,requires_grad=True)*0.1`）实证 5 步内 TypeError（附 PyTorch 非叶子告警），新 grad 断言可捕获该回归。
6. **check_latex.py**：4 个教程文件 + 作业 README 全部 0 问题（`$$` 单行闭合、无公式内中文）。

## 三、Disputed 清单（详见台账）

- **D1 映射统一方向**：已按"Part1/作业风格"统一 Part2 侧，Part1 零改动即三方一致；若 Part1 agent 倾向反向，请 T0 批1收尾裁决。
- **D2 roadmap 节点 2**："作业 2 含采样"不实（无采样题）；pytest 说法与独立脚本形式小出入。docs 侧微修，不阻塞本 Part。
- **D3 可视化双图**：已核实同源（坐标逐点一致），教程注明同源；是否合并单一来源留 G4-4 低优先。
- **D4 ±0.002 数值差**：本机全量实测 2.3749 vs S3 报告 2.3766，多线程归约非确定性，非文档错误。

## 四、遗留（不在本 Part 范围）

- 03-07 五脚本各复制一份 build_dataset/训练循环，无共享模块（S3-08c）：重构影响面大、超整改授权，记入通用问题候选。
- 03 章论文进阶（block_size 3→8 重训）教程未提（roadmap 验证项 ❌）：属 Part2 增量内容，建议 T0 决定是否立项。
