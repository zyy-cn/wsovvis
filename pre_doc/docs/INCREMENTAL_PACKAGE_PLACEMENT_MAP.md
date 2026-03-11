# 文档驱动自动化增量包放置位置

本增量包按 **仓库相对路径** 组织，可直接覆盖到 `wsovvis.zip` 解压后的仓库根目录。

## 1. 放置方式
将本增量包中的文件按相对路径复制到仓库中：

- `AGENTS.md` → 仓库根目录
- `docs/DOC_DRIVEN_AUTOMATION_INDEX.md` → `docs/`
- `docs/contracts/mainline_doc_driven/*.md` → `docs/contracts/mainline_doc_driven/`
- `docs/runbooks/mainline_phase_gate_runbook.md` → `docs/runbooks/`
- `codex/templates/WS_OVVIS_DOC_DRIVEN_TASKPACK_TEMPLATE.md` → `codex/templates/`
- `docs/INCREMENTAL_PACKAGE_PLACEMENT_MAP.md` → `docs/`

## 2. 不需要替换的现有内容
以下现有内容保留，不建议删除：
- `codex/START_HERE.md`
- `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
- `docs/contracts/stageb_track_feature_export_v1/*`
- 历史 `codex/2026*/` 任务目录

## 3. 统一入口
自动化开发的统一入口为：
1. 仓库根目录 `AGENTS.md`
2. `docs/DOC_DRIVEN_AUTOMATION_INDEX.md`

其中：
- `AGENTS.md` 负责给 Codex 提供最高优先级项目规则；
- `docs/DOC_DRIVEN_AUTOMATION_INDEX.md` 负责把 workflow 层、合同层、runbook 层和历史证据层串起来。

## 4. 覆盖策略
本增量包是“新增优先”的覆盖包：
- 不要求推翻现有 workflow；
- 只是在现有 workflow 之上增加一层主线合同与阶段门控；
- 后续新任务优先引用这些新文档，从而逐步把工作流从“prompt 驱动”转成“文档驱动”。
