# Part7 02_modern_components.md 阅读检查记录

## 问题记录

| 时间 | 文件 | 问题 | 是否已修改 |
|------|------|------|-----------|
| 2026-09-02 | `courses/Part7_minimind/tutorial/02_modern_components.md` | RoPE 注入代码 `cos[:T].unsqueeze(0).unsqueeze(0)` 得到 `(1,1,T,hd)`，与 `q(B,T,n_heads,hd)` 广播时 dim2 上 `T≠n_heads` 报错。应为 `unsqueeze(0).unsqueeze(2)` 得到 `(1,T,1,hd)` | 已修改 |
| 2026-09-02 | `courses/Part7_minimind/tutorial/02_modern_components.md` | 权重绑定代码变量名和方向与脚本不一致：文档写 `self.model.embed_tokens.weight = self.lm_head.weight`，实际脚本为 `self.lm_head.weight = self.tok_embeddings.weight`（embedding 层名也不同） | 已修改 |
