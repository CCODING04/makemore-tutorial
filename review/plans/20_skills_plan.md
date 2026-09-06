# Skills 选型与执行计划 v3（网页 + Markdown 双轨优化）

> **版本**：v3.0（2026-09-05）｜ **前置**：备份基线已提交（git: backup 形态优化 v2 完成态）
> **依据**：imported-skills.md（46 个已导入 skills）× 当前站点（116 页自研渲染）× 课程 md（19 Part，G/W 规范已固化）

---

## 一、选型结论（46 选 10）

### A 组 · 网页版面（site_build 渲染层 + 视觉）

| Skill | 用途 | 阶段 |
|---|---|---|
| **awesome-design-md** | 对标 54 个真实站点（Linear/Stripe/Vercel）的排版规范，重做站点 CSS：字号阶梯/行宽/间距系统/代码块/目录当前章滚动高亮/TOC 悬浮窗 | **P-A1（立即）** |
| **frontend-dev** | 实现规范时的工程约束参考（响应式/无障碍/性能） | P-A1 辅助 |
| **excalidraw-diagram** | 关键概念的手绘风讲解图（区别于 Mermaid 的正式感），试点 3 张 | P-A2 |
| **infographic-maker** | 章首"全景信息卡"（一图看懂本章结构），试点 2 张 | P-A2 可选 |

### B 组 · 课程 Markdown 内容（发现问题 + 增补形态）

| Skill | 用途 | 阶段 |
|---|---|---|
| **knowledge-framework-builder** | 19 个 Part 逐一生成结构化知识脉络图（Markmap，可嵌入站点）——**交叉发现章节结构缺失/依赖倒置** | **P-B1（立即）** |
| **education** | 每 Part 生成测验题 + 闪卡（与既有 Anki 卡互补）——**反向检验"学习目标↔正文覆盖"**，G17 的内容侧延伸 | **P-B1（立即）** |
| **academic-tutor** | 对 3 个试点 Part 做苏格拉底式提问审查——发现"自以为讲清了其实没讲清"的点 | P-B2 可选 |
| **tutor-skills** | 全课程转 Obsidian 学习库（StudyVault）——学员侧增值产物 | P-B3 可选 |
| **md-to-pdf-cjk** | 课程 md → PDF 离线教材（按 Part 分册） | P-B2（依赖装不上则阻塞） |

### C 组 · 分发与演示（需网络/账号，**先问用户再动**）

| Skill | 用途 | 阻塞点 |
|---|---|---|
| web-deploy-github / github-pages-auto-deploy | 站点发布 GitHub Pages | 需用户的 GitHub 仓库与凭据 |
| static-app / html-deploy / netlify-deploy / edgeone-pages-deploy | 第三方托管 | 需网络与账号 |
| guizang-ppt-skill / pptx-generator | 每章讲义幻灯 | 网络依赖项需确认 |
| gsap-animation-assistant | 英雄动画升级（GSAP 缓动/时间线） | 需 CDN（可内联裁剪） |

**不采用**：小说/公众号/电商类写作、shader、canvas-design（与课程场景无关）；deck-generator（Gemini 代理当前不可用）。

## 二、执行顺序（✅ v3 全部完成，2026-09-06）

> 完成实绩：P-A1 站点 CSS v3、P-B1 知识图×20、P-B1' 测验闪卡×19（quiz_index 汇总+19 页上线）、P-A2 概念图×3（SVG，excalidraw 渲染链网络阻改零依赖）、P-B2 PDF 分册 19/19（3 份净化版）。P-C 分发阻塞于 STATIC_APP_API_KEY 缺失+网络，待用户解除。

```
P-A1  awesome-design-md 站点 CSS v3 重做        ← 当前
P-B1  knowledge-framework-builder 知识图 ×19
P-B1' education 测验/闪卡 ×19
P-A2  excalidraw 概念图试点 ×3
P-B2  md-to-pdf-cjk PDF 分册（网络允许时）
P-C   分发（问用户：目标平台/仓库）
每步完成：build_site_lite 重建 + check_latex/check_links + git commit
```

## 三、"发现新问题"的机制设计（本轮核心价值）

skills 不只做美化，更做**交叉验证**：
- 知识图生成时若发现某章概念在图中无位置/依赖倒置 → 记入 ledger（结构问题）
- 测验生成时若某学习目标出不了题 → 说明正文覆盖缺失（G17 的内容侧）
- 设计规范对照时若发现可读性违规（行宽/对比度） → 记入 site TODO

所有新发现统一进 `ledger/skills_findings.md`，严重的回写主计划 G 规则。
