# Part 9（CUDA 内核编程）T2 整改报告

- 整改日期：2026-09-04
- 依据：plan_P09.md（T1 预审）+ S1/S2/S3 三份 report_P09 + T0 必修清单 8 项
- REPO 只读；全部修改写入 REVIEW 镜像；G16 遵守（脚本先拷 scratch 再编译运行）

## 一、镜像改动清单

### courses/Part9_cuda_kernels/tutorial/（6 md 全改 + 新增 1 图）
- `02_matmul_optimization.md`：**必修 1**——"FLOP:byte = 2N:8" 笔误改 **2N:12** 并补两笔完整推导
  （理想强度 $\frac{2N^3}{12N^2}=N/6\approx85.33$、naive 有效强度 $\frac{2N}{8N}=0.25$、
  roofline 82 = 82.6 TFLOPS ÷ 1000 GB/s 出处，各配 ⚠️/正文说明）；G2 算术强度推导 LaTeX 化；
  G14 实测口径回填（RTX 4090 / CUDA 12.4 / torch 2.6.0+cu124 / 2026-09-02 共享 GPU + 全 Part 总声明）；
  **G4** 阶梯表后插入 `../images/roofline_ladder_4090.png`（英文图内标注 + 英中双语图注）
- `05_flash_attention.md`：**必修 2**——§2.2 补数字例 x=[1,2,3,5]（l_new=1.2034 与整行直算逐位一致、
  直接相加 2.5032 错约 2 倍）+ `o_new = o₁·α + o₂` 两行正式推导（内核第 ⑤ 行依据）；
  G2×3（softmax 公式 / m₁l₁o₁ 定义 / l_new 推导）全部 LaTeX 化（$$ 单行、无中文、underbrace 标注）；
  G1×2（§4.3 反超三原因、Q3 答案①②③拆行）；**必修 4** 新增"📝 课后作业"节（Assignment 9 链接）
- `03_profiling_and_cuda_apis.md`：**必修 3**——列主序恒等式拆两步（2×2 内存布局例：
  行主序 [1,2,3,4] 按列读=[[1,3],[2,4]]=Aᵀ；恒等式 LaTeX 化，cublasSgemm 调用块保留代码）；
  G1 Q2 答案①②③拆行；ncu "Memory=Compute=92.01 同值"补读表解释（真实现象，复跑 91.60=91.60，
  同一 L1/TEX 管线顶满——T1 待核项就此裁定，数字未改）
- `04_triton_and_extensions.md`：G14×2（两处实测口径回填）；编译坑段补 ninja PATH 精确形态
  （全路径调 python 时不在 PATH）+ TORCH_CUDA_ARCH_LIST warning 属正常两条；末尾补
  "下一章：05 Flash Attention"链接（T1 导航链待核项）；vecadd "~1.70 TB/s 超 DRAM 峰值"补
  L2 命中解释一句（S3 ⚠️ 项）
- `01_gpu_and_first_kernel.md`：G14 实测口径回填（L166）
- `README.md`：**必修 6**——脚本 09 耗时"约 2-4 分钟"改实测口径（**13.5s/13.6s@4090 共享 GPU
  两次独立实测**，15-60 秒预期，冷启动更久；纠正"需独占 GPU"旧说法）；编译坑清单补
  ninja PATH 强化、TORCH_CUDA_ARCH_LIST warning、复制 scripts 连带 bin/ 先 make clean 三条；
  G14 实测参考节标题回填全口径
- `images/roofline_ladder_4090.png`（新建）：六级阶梯 GFLOPS 柱状图 + fp32 峰值 82.6 TFLOPS 虚线，
  数据取教程主表（2026-09-02 实测口径），图内全英文

### assignments/assignment_9/
- `test_cuda_exercises.py`：**必修 5（G18）**——照 P8 修法改 SKIP 语义：`_skip()` 在 pytest 下改抛
  `pytest.skip(reason)`；新增 `_skip_exceptions()` 显式列出两种异常（pytest 的 Skipped 继承
  BaseException，except Exception 捕不到）；main except 链改 `except _skip_exceptions()`；
  docstring 同步
- `cuda_exercises.py` / `assignment.md`：未改（题面与"⏭️ 跳过"承诺本就正确，失效的是 test 兑现方）

### 其它产出
- `ledger/ledger_P09.md`：17 条（P0×2 / P1×11 / P2×4，见下）
- `outline_review/outline_suggestions.md` 追加 3 条 roadmap 项（105% 验收线与教程 50/85 口径冲突、
  节点 9 未同步 5 章/9 脚本、SMEM 48KB 与 decode memory-bound 两目标正文 0 覆盖）——未改 docs

## 二、验证结果（全部通过）

1. **脚本实跑**（scratch/t2_P9/scripts 副本，`make` 自动选 /usr/local/cuda-12.4/bin/nvcc，
   4090 共享 GPU，零编译错误）：
   - 脚本 03：naive **4665.4 GFLOPS**（教程 4697.8，<1%）；打印 "Ideal arithmetic intensity
     85.33 FLOP/byte" 与 "effective intensity 0.25 FLOP/byte" 与修后正文逐字一致
   - 脚本 04：阶梯 L1-L5 = 613.7 / 4868.9 / 5965.1 / 7079.2 / 8686.9 GFLOPS（教程
     553.6/4844.9/5905.4/6967.5/8795.2，共享 GPU 波动内量级/排序一致）
   - 09 耗时口径直接引用 S2/S3 两次独立实测（13.5s/13.6s），未重跑（三方一致已足证）
2. **作业四象限 + S2 场景**（参考答案 = assignment_reference/assignment_09，修复后 test）：
   - pytest + 参考答案 = **5 passed**；独立运行 + 参考答案 = 5/5 通过 🎉
   - pytest + 未实现骨架 = **4 failed, 1 skipped**（修复前 1 failed, 4 passed——SKIP 语义修复生效）
   - 独立 + 未实现骨架 = 4 失败 1 跳过，rc=0
   - S2 场景（题 1-4 已实现 + 题 5 未实现）pytest = **4 passed, 1 skipped**（修复前 4 passed, 1 failed）
3. **check_latex.py**：6 个改动 md 全部"✅ 未发现问题"（含新增 7 处 $$ 公式；两处 `$PATH`
   字面量误配对已改写规避）。

## 三、台账条数

**共 16 条：P0×2、P1×11、P2×3；disputed 0 条**（T1 四项待核全部关闭：3 项处理、1 项经 S3
实测裁定为非笔误；roadmap 3 条按分工移入 outline_suggestions.md，不改 docs）。

## 四、通用问题候选（≤3，供全课程规范）

1. **实测口径"就近书写、无全 Part 统一"**：同文件内 README FA 段有全口径而 matmul 表无、
   05 章标准而 01/02/04 章缺——建议全课程统一"每 Part 实测默认口径写在 README 一处 +
   各章首处引用"的单一出处模式（本次以 05 章 §四为标准回填）。
2. **"文档承诺的测试语义"与 test 实现脱节**：assignment.md 承诺 ⏭️ 而 test 抛自定义异常，
   pytest 下呈现为 FAILED（P8/P9 连续两 Part 同款问题，P9 靠 P8 修法闭环）——建议把
   P8 版 `_skip()/_skip_exceptions()` 固化为全课程作业模板（含"pytest 的 Skipped 继承
   BaseException"这条坑注释），新增作业直接套用。
3. **数字自洽依赖读者手算**：2N:8 与脚本打印 85.33 自相矛盾这种笔误，正是"正文比例式
   与打印值无交叉校验"漏出来的——建议关键算术（强度/倍数/百分比）在教程定稿时跑一遍
   5 行 Python 对账（S1 的 self-check 清单可直接变脚本）。

## 五、好写法候选（≤3）

1. **05 章 §4.3"教学版为什么能反超（诚实解读）" + 4.4 陷阱 4**（三学生共同点赞）：主动交代
   LSE 不写/autotune 选优/小形状开销三原因，"跨卡数字不可直接比"写进正文——反吹牛段落是
   面试防守现成弹药，可作全部实测章节的模板。
2. **Makefile 的"executable 版编译坑清单"**（S2 满分体验）：多版本 CUDA 自动选择 + rpath +
   gcc-11 降级每个 ifeq 都注释"为什么"，README 坑点表的每条防御都在构建脚本里真实生效——
   "文档坑表 = 构建脚本注释"的双写模式值得推广。
3. **05 章 3.1 数据流 ASCII 图**（S1：靠它读懂 5 行核心）：图先于代码、每个箭头标形状、
   每个状态标精度——毕业章"推导→图→逐行→实测"的四遍读法，可作高难内核章节的标准叙事序。
