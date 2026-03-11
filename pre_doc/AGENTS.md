# AGENTS.md

本文件是 WSOVVIS 仓库在 **文档驱动自动化开发模式** 下的最高优先级项目入口。

## 1. 当前唯一主线目标
当前自动化主线不是“把所有模块都做出来”，而是优先验证以下研究命题：

> 在 incomplete positive label set 的 WS-OVVIS 设定下，closed-world 弱监督会把 hidden positives 系统性压成背景；open-world set-to-track attribution 能显著缓解这一失败模式。

任何不直接服务于该命题的复杂增强，默认不开启。

## 2. 读取顺序（强制）
每次新会话或新任务，按以下顺序读取：

1. `AGENTS.md`（本文件）
2. `docs/DOC_DRIVEN_AUTOMATION_INDEX.md`
3. `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
4. `docs/contracts/mainline_doc_driven/00_mainline_objective_scope.md`
5. `docs/contracts/mainline_doc_driven/01_mainline_implementation_spec.md`
6. `docs/contracts/mainline_doc_driven/02_mainline_metrics_acceptance.md`
7. `docs/contracts/mainline_doc_driven/03_mainline_failure_playbook.md`
8. `docs/runbooks/mainline_phase_gate_runbook.md`
9. 当前 task pack 文档

## 3. 主线默认范围（ON）
主线默认只保留：
- WS-OVVIS 协议（Uniform Missing / Long-tail Missing）
- VideoCutLER 伪 tube 与 Stage B export / bridge / loader
- SeqFormer class-agnostic basis
- mixed track representation
- 简化的 open-world attribution（bg / unk / coverage）
- 相邻窗口 linking
- 单轮 quality-aware refinement（仅在 G4 通过后允许进入主线）

## 4. 主线默认关闭项（OFF）
以下项默认关闭，除非 phase gate 文档显式允许：
- Scenario/Domain Missing 主协议
- label-set expansion
- 第二轮 refinement
- 强 memory-based global classification
- LLM 重排 / 全局词表扩张
- 大范围 joint fine-tune / 全网弱语义重训练

## 5. 阶段门控规则（强制）
- 若 G3 未通过（HPR/UAR/SCR 未相对 closed-world baseline 稳定提升），不得默认进入 refinement。
- 若 G4 未通过（linking 不稳），不得默认启用 memory aggregation。
- 若 refinement 收益不稳定或引入负迁移，必须降级为附录增强。

## 6. 历史任务与当前主线的关系
`codex/2026*/` 下的历史任务记录是实现证据与失败模式证据，不再自动等同于当前主线规范。
当前主线规范以本文件和 `docs/contracts/mainline_doc_driven/*` 为准。

## 7. 修改原则
- 优先小步增量，不擅自扩大 scope。
- 优先通过当前 gate 的最小验收，不为了“看起来更完整”而新增模块。
- 遇到高风险模块失败时，先按 failure playbook 退化，不得默认继续加损失或加组件。
