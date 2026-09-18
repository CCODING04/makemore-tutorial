# Archify 重制课程架构图 · 经验总结

> 2026-09 · temp/archify_demo 演示项目 · 9 张图全部 showcase 校验 + 4 视口浏览器核验 + judge 视觉验收通过

## 最重要的三条

1. **先读源内容的"语义"，再选图结构**。第一版 Block 图把 LN→注意力→⊕ 画成串联主路，
   残差只是旁边一条虚线——形状对了，语义反了。正确做法：先从章节代码
   （`x = x + self.sa(self.ln1(x))`）提炼不变量（x 沿主干只经历加法、LN 在支路上、
   子层只贡献增量），再让图结构直接表达这个不变量。
2. **汇合点要用"多入一节点"表达，不能用线性链冒充**。Self-Attention 的打分需要 Q、K
   两路汇合，加权求和需要权重、V 两路汇合；画成 Q→K→softmax→V 线性链等于教错数学。
   同理：RAG 的查询注入、GRPO 的训练闭环、LoRA 的 ⊕ 融合，都要让"谁和谁汇合、
   谁回流"在图上真实可见。
3. **布局约束要前置，不要事后补救**。dataflow 渲染器的列间距固定 215px（首列节点
   ≤152px、相邻同行节点宽度和 ≤~360px、viewBox ≤~1085 才能保证 1440 视口下小字
   ≥6px）；workflow 节点宽度直接影响画布宽度和投影字号。先按约束设计节点文案宽度，
   比生成后反复修 labelDx/labelDy 便宜得多。

## 站点嵌入八条（第二、三轮反馈沉淀，嵌入变体必须逐条过）

1. **主题跟随**：iframe 内默认配色必须同步父页 `data-theme`（构建期注入 `inject_widget_theme`，
   `THEME_SYNC_WIDGETS` 登记），并隐藏 iframe 内的主题切换按钮；独立打开不受影响、可自由切换。
2. **播放故事（story）只属于诚实的顺序流**：存在汇合（join）或闭环（cycle）的图一律不写
   `meta.views`——顺序巡游动画会暗示"先走完一条再走另一条"，而 ⊕ 这类同步点是
   "两路到齐才放行"。同步语义改由 ⊕ 节点 tag（如"两路到齐才输出"）+ 结论卡表达。
3. **图例禁止出现渲染器的通用类别词**（消息总线/外部系统……读者看不懂）：`meta.legend`
   用 mode=auto + entries 按图的真实语义起名（如 messagebus→"⊕ 残差汇合点（同步）"）；
   起不出真实名字的类别 `visible:false`。
4. **嵌入版 = 定制模板页**（第三轮定稿）：archify 通用查看器只用于 showcase/temp 画廊；
   站内 widget 一律由 `temp/archify_demo/custom_template/build_custom.py` 从已校验规格生成
   零依赖单文件 HTML（设计语言见「定制模板做对的事」）。SVG viewBox 宽 ~1000-1060，
   在 798px iframe 里投影 ≥0.75，12px 标题→9px、10px 副标→7.5px，可读。
5. **showcase 版与嵌入版是两份规格**（`specs/X.json` + `specs/X.embed.architecture.json`）：
   同一语义、两套几何；语义修改要同步改两份，几何各自独立。showcase 版可保留 views（诚实顺序流）。
6. **注册两件事才算接完**：`WIDGET_HOME`（独立页"↩ 返回章节"按钮）+ `THEME_SYNC_WIDGETS`
   （主题跟随）；章节侧用站点原生 ```widget 围栏（组件名 + 高度px）。定制模板页用
   `[data-theme=dark]` CSS 变量接管配色——站点注入脚本的兜底分支正好直设该属性，
   与 archify 查看器共用同一套注入，无需改构建脚本。
7. **替换要结合上下文**：保留有教学价值的手绘/ASCII 版本时，新旧之间加一句导语衔接；
   被替换的 Mermaid/ASCII/图引用删干净，导语里不要引用已删除的交互特性（如"试试视图切换"）。
8. **高度要实测**：iframe 高度 ≈ 头部(90) + SVG×投影 + 卡片(230) + 余量；交付后用浏览器
   量 iframe 内 scrollHeight 验证无内部滚动条。

## 定制模板做对的事（第三轮定稿，站内 9 张图全部采用）

参照物：一张外部 "RSI 闭环流程图" 页（产品落地页质感）与自研 transformer_block_custom 试点。
**架构：archify 规格（validate 管线）仍是语义唯一源；定制模板只是表现层**——
"Archify IR → 定制模板"，两份产物各司其职。

1. **设计 token 化**：--bg/--surface/--border/--brand… 一套变量同时服务亮色与暗色
   （`:root[data-theme=dark]` + `prefers-color-scheme` 兜底），换主题零重复样式。
2. **页面叙事三段式**：徽章（脉冲点+Part 归属）→ 大标题（`clamp()` 流体字号 + 品牌色
   公式 em）→ 导语（内联 code）→ **图前三张数据卡**（label/value/sub，先给数字锚点）。
3. **节点三级层次 + 语义色标题**：标题按角色着色（蓝=通信/参数、绿=计算/输出、
   紫=冻结/内容、琥珀=可训练、蓝底=⊕ 汇合主角），副标灰、第三行更弱；描边用
   88/55 透明度克制表达，底色只在"主角节点"用 soft 填充。
4. **动效分寸（最重要的一条）**：
   - 顺序管线（MLP/GPT-2/MinHash）→ 允许 d1..dN 入场分级 + 主路常速虚线；
   - 汇合/闭环图（Block/attention/LoRA/RAG/verl/agent）→ **禁入场排序**，全部
     通路同速常速流动——"并行、到齐才放行"由同时流动本身表达；
   - `prefers-reduced-motion` 一律停动画；图例声明"常速流动=数据流（不表示先后）"。
5. **悬停即教学**：悬停 ⊕/汇合点 → 淡出无关、同时高亮它的**两路输入**；悬停支路 →
   高亮"读 x → 子层 → 写回"全路径。data-rel 多标签 + 每组一条 CSS 规则，JS 仅 10 行。
6. **工程纪律**：SVG 内嵌坐标校验表注释；生成后跑程序化断言——节点文本 getBBox
   不溢出、边标签与节点零相交、1280/1440 无横向溢出、悬停透明度 0.18、
   reduced-motion 动画归零。
7. **零依赖单文件**：无 CDN、无字体请求，iframe 秒开；`[data-theme]` 挂钩与站点
   主题注入天然兼容。

## 渲染器约束速查（踩过的坑）

| 约束 | 数值/规则 | 后果 |
|---|---|---|
| dataflow 列间距 | 固定 215px，与 viewBox 无关 | 节点过宽会侵入相邻列，产生 2.5px 碎线段 |
| 节点宽度 | 文本估宽 = 字符单位 ×6.2（CJK≈2 单位） | 标签超宽直接 FAIL，需改文案或加 width |
| 同行相邻流 | 间距 ≥34px，微段 ≥8px | 7px 抖动会被拒；同排水平流用 `route:"straight"` |
| 直线流标签 | 27px 双行块（带 classification）放不进窄间隙 | 直线流标签用单词级短词，语义挪进节点/卡片 |
| 桌面可读性 | 1440 视口投影字号 ≥6px | viewBox 宽度上限 ≈1085（dataflow） |
| 首屏完整 | scrollHeight ≤ 视口高 | 竖排长链必须改横排；卡片约占 300px 高 |
| 架构图标签 | 默认锚在源端附近 | 竖排链上标签会压节点，需 labelDy +24 |
| 标签防碰撞 | 每条对角线标签可独立锚定 labelDx/Dy | 交叉对角线的标签锚向不同端点错开 |
| dataflow/workflow 小字 | sublabel 源字号仅 6.7~7px | 嵌入投影后 <6px；嵌入版改用 architecture 重排 |
| Archify 主题 | `?theme=` / localStorage / `data-theme` 属性 | 注入脚本直设属性即可生效（按钮隐藏后无需调 API） |
| 自研渲染器斜体 × 数学 | `^*` 在同一 li/段落里出现两次会被配成 `<em>`，撕开 `$...$` | 数学里写 `J^{\ast}` 等价形式，避免裸 `*` 成对出现 |

## 工作流（可复用的循环）

```
读源（章节代码/ASCII/Mermaid 提取事实） → 选类型（architecture/dataflow/workflow）
→ 写规格（先自动路由，不预调几何） → validate（showcase，9 项检查 0 错 0 警）
→ 按诊断修（一次一个 subject；两轮无改善才读渲染器源码） → deliver（产出 SHA 回执）
→ visual-check（4 视口） → 截图 → judge 视觉验收（重点验收语义表达而非美观）
```

## 画 Transformer Block 的业界参照（调研结论）

- **经典派（Illustrated Transformer / 原论文）**：竖排堆叠，"Add & Norm" 盒子内联在
  主路上，残差箭头绕过子层汇入。直观但残差语义被装进盒子里。
- **Pre-LN 平行支路派（Xiong et al. 2020 图 1）**：恒等路径与 LN→子层支路并行，显式 +。
- **Anthropic 残差流总线派（Transformer Circuits, Elhage et al. 2021）**：残差流是
  通信总线，子层"读"（经 LN）和"写"（加法）流。可解释性论文的事实标准。
- 本项目最终采用 2+3 的横排混合：主干 = 残差流（region 显式命名），支路 = 读 x →
  LN → 子层 → 写回 ⊕。judge 验收确认 `x = x + f(x)` 一眼可读。

## 遗留事项

- p02/p08 的 `requires_grad`、"残差主干" 标签被渲染进卡片/子标题而非节点 tag 芯片，
  judge 判定不误导、已通过；若想强制上芯片需再收文案宽度。
- 全图集为 temp 演示，未动课程源文件；正式替换进章节/站点需另行评审。
