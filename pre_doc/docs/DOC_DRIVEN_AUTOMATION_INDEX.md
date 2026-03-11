# WSOVVIS 文档驱动自动化统一入口

本文件是当前仓库从“Codex 辅助开发”转向“文档驱动自动化开发”的统一入口索引。

## 1. 目的
当前仓库已经积累了大量：
- 历史任务 pack
- prompt / output 记录
- 阶段性实现与测试
- workflow 级远端验证规则

但这些内容天然分散，且更偏向“人工带着 Codex 工作”。

本索引把当前主线开发需要遵守的项目级规则集中起来，使后续任务优先服从：
- 主线目标
- 主线默认范围
- 阶段门控
- 失败回退
- 统一验收口径

## 2. 当前主线
当前主线目标：
1. 稳定复现 WS-OVVIS 协议与基础基线；
2. 基于现有 Stage B / Stage C 数据平面，验证 open-world attribution 的必要性；
3. 在核心门控通过后，再进入整视频 linking 与单轮 refinement；
4. 所有增强项均从属于主线命题，不抢占主文叙事。

## 3. 文档层级与职责
### A. workflow 事实层
- `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
- `codex/START_HERE.md`

作用：定义本地/远端环境、canonical validation、Git / remote verify 等事实。

### B. 项目主线合同层（当前新增）
- `docs/contracts/mainline_doc_driven/00_mainline_objective_scope.md`
- `docs/contracts/mainline_doc_driven/01_mainline_implementation_spec.md`
- `docs/contracts/mainline_doc_driven/02_mainline_metrics_acceptance.md`
- `docs/contracts/mainline_doc_driven/03_mainline_failure_playbook.md`

作用：定义当前主线的目标、实现版本、验收门槛与失败退化路径。

### C. 阶段执行层
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `codex/templates/WS_OVVIS_DOC_DRIVEN_TASKPACK_TEMPLATE.md`

作用：把主线合同转成每一阶段可执行的任务包与推进顺序。

### D. 历史实现证据层
- `docs/PROJECT_PROGRESS.md`
- `docs/STAGE*_*.md`
- `codex/2026*/`

作用：作为已实现状态与失败模式证据，不再替代主线规范。

## 4. 与现有仓库进度的对应关系
从当前仓库状态看，以下模块已有明显基础：
- P1 protocol tooling
- Stage B export / bridge / loader
- Stage C0--C4 offline attribution scorer / plumbing
- Stage D wiring / helper / quick-check / CI mirror

因此，当前文档驱动自动化的重点不是重写这些模块，而是：
- 统一主线规范
- 压缩默认范围
- 把未来任务组织成 phase/gate 推进

## 5. 冲突优先级
当前任务若发生文档冲突，优先级从高到低为：
1. `AGENTS.md`
2. 本文件
3. `docs/contracts/mainline_doc_driven/*`
4. `docs/runbooks/mainline_phase_gate_runbook.md`
5. 当前 task pack
6. `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
7. 历史 task 输出与旧 handoff 文档

## 6. 统一入口的使用方式
未来任何新任务，都应在 task pack 冒头显式引用：
- `AGENTS.md`
- 本文件
- 四份 mainline contracts
- 对应 phase gate 的运行手册

若 task pack 未引用这些文档，则视为仍沿用旧的“长 prompt 辅助开发模式”。
