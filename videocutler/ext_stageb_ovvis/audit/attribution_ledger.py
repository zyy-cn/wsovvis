from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .trajectory_gt_audit import (
    build_attribution_rows,
    load_gt_sidecar_lookup,
    summarize_attribution_rows,
    _write_jsonl,
)


Record = Dict[str, Any]


@dataclass
class _Snapshot:
    stage_id: str
    snapshot_id: str
    phase: str
    rows: List[Record] = field(default_factory=list)


class AttributionLedgerBuffer:
    def __init__(
        self,
        *,
        output_root: Path,
        dataset_name: str,
        trajectory_source_branch: str,
        topk: int = 5,
        gt_sidecar_dir: str = "audit",
    ) -> None:
        self.output_root = Path(output_root)
        self.dataset_name = str(dataset_name)
        self.trajectory_source_branch = str(trajectory_source_branch)
        self.topk = int(topk) if int(topk) > 0 else 5
        self.gt_sidecar_dir = str(gt_sidecar_dir)
        self.gt_lookup = load_gt_sidecar_lookup(
            self.output_root,
            dataset_name=self.dataset_name,
            trajectory_source_branch=self.trajectory_source_branch,
            gt_sidecar_dir=self.gt_sidecar_dir,
        )
        self._snapshots_by_stage: dict[str, List[_Snapshot]] = defaultdict(list)
        self._previous_by_trajectory: dict[str, Record] = {}
        self._stage_written: set[str] = set()
        self._summary_payload: Optional[Dict[str, Any]] = None

    def __call__(self, context: Mapping[str, Any]) -> None:
        self.record_snapshot(dict(context))

    def record_snapshot(self, context: Mapping[str, Any]) -> List[Record]:
        stage_id = str(context.get("stage_id", "")).strip()
        snapshot_id = str(context.get("snapshot_id", "")).strip() or str(context.get("phase", "snapshot"))
        phase = str(context.get("phase", "")).strip() or "snapshot"
        materialized_samples = list(context.get("materialized_samples", []))
        projector = context.get("projector")
        if not stage_id:
            raise ValueError("audit snapshot missing stage_id")
        if projector is None:
            raise ValueError("audit snapshot missing projector")
        temperature = float(context.get("temperature", 0.07))
        rows = build_attribution_rows(
            output_root=self.output_root,
            dataset_name=self.dataset_name,
            trajectory_source_branch=self.trajectory_source_branch,
            stage_id=stage_id,
            snapshot_id=snapshot_id,
            materialized_samples=materialized_samples,
            projector=projector,
            topk=self.topk,
            gt_sidecar_lookup=self.gt_lookup,
            temperature=temperature,
            previous_by_trajectory=self._previous_by_trajectory,
        )
        for row in rows:
            trajectory_id = str(row.get("trajectory_id", ""))
            if trajectory_id:
                self._previous_by_trajectory[trajectory_id] = dict(row)
        self._snapshots_by_stage[stage_id].append(_Snapshot(stage_id=stage_id, snapshot_id=snapshot_id, phase=phase, rows=rows))
        if phase == "stage_end":
            self.flush_stage(stage_id)
        return rows

    def flush_stage(self, stage_id: str) -> Path:
        stage_id = str(stage_id)
        stage_dir = self.output_root / "train" / stage_id
        stage_path = stage_dir / "attribution_ledger.jsonl"
        snapshots = list(self._snapshots_by_stage.get(stage_id, []))
        rows: List[Record] = []
        for snap in snapshots:
            rows.extend(sorted(snap.rows, key=lambda row: str(row.get("trajectory_id", ""))))
        _write_jsonl(stage_path, rows)
        self._stage_written.add(stage_id)
        return stage_path

    def summarize(self) -> Dict[str, Any]:
        stage_rows: Dict[str, List[Record]] = {}
        for stage_id, snapshots in self._snapshots_by_stage.items():
            stage_rows[stage_id] = []
            for snap in snapshots:
                stage_rows[stage_id].extend(snap.rows)
        summary = summarize_attribution_rows(stage_rows)
        summary.update(
            {
                "dataset_name": self.dataset_name,
                "trajectory_source_branch": self.trajectory_source_branch,
                "topk": self.topk,
                "gt_sidecar_dir": self.gt_sidecar_dir,
                "stage_ids": sorted(stage_rows.keys(), key=lambda item: (0 if item == "prealign" else 1 if item == "softem_base" else 2 if item == "softem_aug" else 99, str(item))),
                "snapshots_emitted": int(sum(len(snapshots) for snapshots in self._snapshots_by_stage.values())),
                "stages_emitted": int(len(self._snapshots_by_stage)),
                "stage_row_counts": {stage_id: int(sum(len(snap.rows) for snap in snapshots)) for stage_id, snapshots in self._snapshots_by_stage.items()},
            }
        )
        self._summary_payload = summary
        return summary

    def finalize(self) -> Dict[str, Any]:
        for stage_id in list(self._snapshots_by_stage.keys()):
            if stage_id not in self._stage_written:
                self.flush_stage(stage_id)
        summary = self.summarize()
        summary_path = self.output_root / "train" / "audit" / "attribution_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
