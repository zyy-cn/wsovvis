#!/usr/bin/env python3
"""Patch GT-fullY clean trainer with optional trajectory-level confidence EMA.

This patch is intentionally surgical: it preserves existing soft_routing / nohub /
auto-absorber code and only adds optional EMA gating behind default-off CLI flags.
"""
from __future__ import annotations

import argparse
from pathlib import Path


HELPER_BLOCK = r'''


def _trajectory_ema_key(ex: Mapping[str, Any]) -> str:
    """Stable per-trajectory key for confidence EMA state."""
    dataset_name = str(ex.get("dataset_name", ""))
    clip_id = str(ex.get("clip_id", ""))
    video_id = str(ex.get("video_id", clip_id))
    trajectory_id = str(ex.get("trajectory_id", ex.get("join_key", "")))
    if not trajectory_id:
        trajectory_id = str(ex.get("gt_instance_id", ex.get("ann_id", "")))
    return "|".join([dataset_name, video_id, clip_id, trajectory_id])


class ConfidenceEMAState:
    """GPU-friendly delayed-use per-trajectory EMA for soft-routing confidence.

    The current step uses the EMA value from previous observations. The current
    local_explained observation is detached and written back only after the gate
    value is produced, so the row gate represents historical stability rather
    than the current batch's instantaneous confidence.
    """

    def __init__(self, *, num_items: int, device: torch.device, beta: float) -> None:
        self.num_items = int(num_items)
        self.beta = float(beta)
        self.device = device
        self.explained_ema = torch.zeros((self.num_items,), device=device, dtype=torch.float32)
        self.update_count = torch.zeros((self.num_items,), device=device, dtype=torch.long)

    def gate_and_update(
        self,
        *,
        ids: torch.Tensor,
        local_explained: torch.Tensor,
        epoch_one_based: int,
        warmup_epochs: int,
        min_updates: int,
        delayed_use: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if ids.numel() == 0:
            return local_explained.detach(), {
                "confidence_ema_enabled": 1.0,
                "confidence_ema_ready_rate": 0.0,
                "confidence_ema_fallback_rate": 1.0,
                "confidence_ema_update_count_mean": 0.0,
                "confidence_ema_local_explained_mean": 0.0,
                "confidence_ema_gate_explained_mean": 0.0,
                "confidence_ema_abs_local_minus_ema_mean": 0.0,
            }
        ids = ids.to(device=self.device, dtype=torch.long)
        local = local_explained.detach().to(device=self.device, dtype=torch.float32)
        prev = self.explained_ema[ids]
        counts = self.update_count[ids]
        if int(epoch_one_based) > int(warmup_epochs):
            ready = counts >= int(min_updates)
        else:
            ready = torch.zeros_like(counts, dtype=torch.bool)
        ready = ready.to(device=self.device, dtype=torch.bool)

        # Delayed-use path: gate uses the pre-update EMA.  A non-delayed option is
        # kept only for controlled ablations; the recommended setting is delayed.
        if bool(delayed_use):
            gate = torch.where(ready, prev, local)
        else:
            first = counts <= 0
            candidate_new = torch.where(first, local, self.beta * prev + (1.0 - self.beta) * local)
            gate = torch.where(ready, candidate_new, local)

        first = counts <= 0
        new = torch.where(first, local, self.beta * prev + (1.0 - self.beta) * local)
        self.explained_ema[ids] = new
        self.update_count[ids] = counts + 1

        ready_f = ready.float()
        stats = {
            "confidence_ema_enabled": 1.0,
            "confidence_ema_ready_rate": float(ready_f.mean().detach().cpu().item()),
            "confidence_ema_fallback_rate": float((1.0 - ready_f).mean().detach().cpu().item()),
            "confidence_ema_update_count_mean": float(counts.float().mean().detach().cpu().item()),
            "confidence_ema_local_explained_mean": float(local.mean().detach().cpu().item()),
            "confidence_ema_gate_explained_mean": float(gate.mean().detach().cpu().item()),
            "confidence_ema_abs_local_minus_ema_mean": float(torch.abs(local - prev).mean().detach().cpu().item()),
        }
        return gate.detach(), stats

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_items": int(self.num_items),
            "beta": float(self.beta),
            "explained_ema": self.explained_ema.detach().cpu(),
            "update_count": self.update_count.detach().cpu(),
        }

    def final_rows(self, *, key_by_index: Sequence[str]) -> List[Dict[str, Any]]:
        explained = self.explained_ema.detach().cpu().numpy().astype(float).tolist()
        counts = self.update_count.detach().cpu().numpy().astype(int).tolist()
        rows: List[Dict[str, Any]] = []
        for idx in range(int(self.num_items)):
            rows.append({
                "ema_index": int(idx),
                "trajectory_key": str(key_by_index[idx]) if idx < len(key_by_index) else str(idx),
                "explained_ema": float(explained[idx]),
                "update_count": int(counts[idx]),
            })
        return rows
'''


CLI_BLOCK = r'''
    p.add_argument("--enable_confidence_ema", action="store_true", help="Enable delayed-use per-trajectory EMA for soft-routing row gating.")
    p.add_argument("--confidence_ema_beta", type=float, default=0.9)
    p.add_argument("--confidence_ema_warmup_epochs", type=int, default=3)
    p.add_argument("--confidence_ema_min_updates", type=int, default=3)
    p.add_argument("--confidence_ema_signal", default="local_explained", choices=("local_explained",), help="EMA signal. First version intentionally supports only the calibrated nohub local_explained signal.")
    p.add_argument("--confidence_ema_delayed_use", action="store_true", default=True, help="Use pre-update EMA for current step and update EMA after gate creation.")
    p.add_argument("--disable_confidence_ema_delayed_use", dest="confidence_ema_delayed_use", action="store_false")
    p.add_argument("--enable_confidence_ema_logging", action="store_true")
'''


def replace_once(s: str, old: str, new: str, label: str) -> str:
    if new.strip() in s:
        return s
    if old not in s:
        raise RuntimeError(f"patch marker not found: {label}")
    return s.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--target", default="videocutler/run_stageb_train_gt_full_y_clean.py")
    args = ap.parse_args()
    target = Path(args.repo_root) / args.target
    if not target.is_file():
        raise SystemExit(f"missing target file: {target}")
    s = target.read_text(encoding="utf-8")
    original = s

    # 1) Helpers after _iter_microbatches.
    helper_marker = '''def _iter_microbatches(groups: Sequence[Sequence[Mapping[str, Any]]], *, max_groups_per_batch: int) -> List[List[int]]:\n    n = max(1, int(max_groups_per_batch))\n    return [list(range(i, min(i + n, len(groups)))) for i in range(0, len(groups), n)]\n'''
    s = replace_once(s, helper_marker, helper_marker + HELPER_BLOCK, "helper block after _iter_microbatches")

    # 2) Validation.
    validation_marker = '''    if bool(args.enable_absorber_logging) and str(args.protocol) != "soft_routing":\n        raise ValueError("--enable_absorber_logging is only supported for --protocol soft_routing")\n'''
    validation_new = validation_marker + '''\n    if bool(getattr(args, "enable_confidence_ema", False)) and str(args.protocol) != "soft_routing":\n        raise ValueError("--enable_confidence_ema is only supported for --protocol soft_routing")\n    if str(getattr(args, "confidence_ema_signal", "local_explained")) != "local_explained":\n        raise ValueError("Only --confidence_ema_signal local_explained is supported in the first EMA version")\n'''
    s = replace_once(s, validation_marker, validation_new, "confidence EMA validation")

    # 3) Initialization after groups.
    groups_marker = '''    groups = _group_by_clip(examples)\n'''
    groups_new = groups_marker + '''    confidence_ema_enabled = bool(getattr(args, "enable_confidence_ema", False))\n    confidence_ema_state: Optional[ConfidenceEMAState] = None\n    confidence_ema_key_to_index: Dict[str, int] = {}\n    confidence_ema_key_by_index: List[str] = []\n    confidence_ema_epoch_rows: List[Dict[str, Any]] = []\n    if confidence_ema_enabled:\n        for ex in examples:\n            key = _trajectory_ema_key(ex)\n            if key not in confidence_ema_key_to_index:\n                confidence_ema_key_to_index[key] = len(confidence_ema_key_by_index)\n                confidence_ema_key_by_index.append(key)\n        confidence_ema_state = ConfidenceEMAState(\n            num_items=len(confidence_ema_key_by_index),\n            device=device,\n            beta=float(getattr(args, "confidence_ema_beta", 0.9)),\n        )\n'''
    s = replace_once(s, groups_marker, groups_new, "confidence EMA init after groups")

    # 4) Replace row_weight creation after local explained.
    row_weight_marker = '''                    explained = torch.sigmoid(float(args.soft_gamma) * (conf - float(args.soft_tau)))\n                    row_weight = 1.0 - explained\n'''
    row_weight_new = '''                    explained = torch.sigmoid(float(args.soft_gamma) * (conf - float(args.soft_tau)))\n                    confidence_ema_row: Dict[str, float] = {}\n                    if confidence_ema_state is not None:\n                        ema_ids = torch.tensor(\n                            [int(confidence_ema_key_to_index[_trajectory_ema_key(ex)]) for ex in group],\n                            device=device,\n                            dtype=torch.long,\n                        )\n                        gate_explained, confidence_ema_row = confidence_ema_state.gate_and_update(\n                            ids=ema_ids,\n                            local_explained=explained.detach(),\n                            epoch_one_based=int(epoch_idx) + 1,\n                            warmup_epochs=int(getattr(args, "confidence_ema_warmup_epochs", 3)),\n                            min_updates=int(getattr(args, "confidence_ema_min_updates", 3)),\n                            delayed_use=bool(getattr(args, "confidence_ema_delayed_use", True)),\n                        )\n                        row_weight = 1.0 - gate_explained\n                    else:\n                        row_weight = 1.0 - explained\n'''
    s = replace_once(s, row_weight_marker, row_weight_new, "confidence EMA gate replacement")

    # 5) Add EMA row stats to soft_row.
    softrow_marker = '''                        "explained_mass_mean": float((1.0 - row_weight_np).mean()) if row_weight_np.size else 0.0,\n                        "explicit_hub_top1_share": explicit_hub_share,\n                    }\n'''
    softrow_new = '''                        "explained_mass_mean": float((1.0 - row_weight_np).mean()) if row_weight_np.size else 0.0,\n                        "explicit_hub_top1_share": explicit_hub_share,\n                    }\n                    if confidence_ema_row:\n                        soft_row.update(confidence_ema_row)\n'''
    s = replace_once(s, softrow_marker, softrow_new, "confidence EMA soft_row stats")

    # 6) Add microbatch summary fields.
    micro_marker = '''                "explicit_hub_top1_share": _mean(batch_float_stats.get("explicit_hub_top1_share", [])),\n            }\n'''
    micro_new = '''                "explicit_hub_top1_share": _mean(batch_float_stats.get("explicit_hub_top1_share", [])),\n                "confidence_ema_ready_rate": _mean(batch_float_stats.get("confidence_ema_ready_rate", [])),\n                "confidence_ema_fallback_rate": _mean(batch_float_stats.get("confidence_ema_fallback_rate", [])),\n                "confidence_ema_update_count_mean": _mean(batch_float_stats.get("confidence_ema_update_count_mean", [])),\n                "confidence_ema_local_explained_mean": _mean(batch_float_stats.get("confidence_ema_local_explained_mean", [])),\n                "confidence_ema_gate_explained_mean": _mean(batch_float_stats.get("confidence_ema_gate_explained_mean", [])),\n                "confidence_ema_abs_local_minus_ema_mean": _mean(batch_float_stats.get("confidence_ema_abs_local_minus_ema_mean", [])),\n            }\n'''
    s = replace_once(s, micro_marker, micro_new, "confidence EMA microbatch stats")

    # 7) Add epoch summary fields.
    epoch_marker = '''            "explicit_hub_top1_share_epoch": _mean(epoch_float_stats.get("explicit_hub_top1_share", [])),\n            "absorber_logging_enabled": bool(absorber_logging_enabled),\n'''
    epoch_new = '''            "explicit_hub_top1_share_epoch": _mean(epoch_float_stats.get("explicit_hub_top1_share", [])),\n            "confidence_ema_enabled": bool(confidence_ema_enabled),\n            "confidence_ema_ready_rate_epoch": _mean(epoch_float_stats.get("confidence_ema_ready_rate", [])),\n            "confidence_ema_fallback_rate_epoch": _mean(epoch_float_stats.get("confidence_ema_fallback_rate", [])),\n            "confidence_ema_update_count_mean_epoch": _mean(epoch_float_stats.get("confidence_ema_update_count_mean", [])),\n            "confidence_ema_local_explained_mean_epoch": _mean(epoch_float_stats.get("confidence_ema_local_explained_mean", [])),\n            "confidence_ema_gate_explained_mean_epoch": _mean(epoch_float_stats.get("confidence_ema_gate_explained_mean", [])),\n            "confidence_ema_abs_local_minus_ema_mean_epoch": _mean(epoch_float_stats.get("confidence_ema_abs_local_minus_ema_mean", [])),\n            "absorber_logging_enabled": bool(absorber_logging_enabled),\n'''
    s = replace_once(s, epoch_marker, epoch_new, "confidence EMA epoch stats")

    # 8) Append epoch rows after jsonl append.
    epoch_append_marker = '''        _append_jsonl(runtime_metrics_path, epoch_summary)\n        _append_jsonl(protocol_metrics_path, epoch_summary)\n\n        if absorber_logging_enabled:\n'''
    epoch_append_new = '''        _append_jsonl(runtime_metrics_path, epoch_summary)\n        _append_jsonl(protocol_metrics_path, epoch_summary)\n        if confidence_ema_enabled:\n            confidence_ema_epoch_rows.append({\n                "epoch": int(epoch_idx) + 1,\n                "confidence_ema_ready_rate_epoch": epoch_summary.get("confidence_ema_ready_rate_epoch", 0.0),\n                "confidence_ema_fallback_rate_epoch": epoch_summary.get("confidence_ema_fallback_rate_epoch", 0.0),\n                "confidence_ema_update_count_mean_epoch": epoch_summary.get("confidence_ema_update_count_mean_epoch", 0.0),\n                "confidence_ema_local_explained_mean_epoch": epoch_summary.get("confidence_ema_local_explained_mean_epoch", 0.0),\n                "confidence_ema_gate_explained_mean_epoch": epoch_summary.get("confidence_ema_gate_explained_mean_epoch", 0.0),\n                "confidence_ema_abs_local_minus_ema_mean_epoch": epoch_summary.get("confidence_ema_abs_local_minus_ema_mean_epoch", 0.0),\n                "residual_weight_mean_epoch": epoch_summary.get("residual_weight_mean_epoch", 0.0),\n                "residual_weight_p10_mean_epoch": epoch_summary.get("residual_weight_p10_mean_epoch", 0.0),\n                "residual_weight_p50_mean_epoch": epoch_summary.get("residual_weight_p50_mean_epoch", 0.0),\n                "residual_weight_p90_mean_epoch": epoch_summary.get("residual_weight_p90_mean_epoch", 0.0),\n            })\n\n        if absorber_logging_enabled:\n'''
    s = replace_once(s, epoch_append_marker, epoch_append_new, "confidence EMA epoch row collection")

    # 9) Save EMA state in checkpoint.
    ckpt_marker = '''        "global_step": int(global_step),\n    }, ckpt_path)\n'''
    ckpt_new = '''        "global_step": int(global_step),\n        "confidence_ema_state": confidence_ema_state.state_dict() if confidence_ema_state is not None else None,\n    }, ckpt_path)\n'''
    s = replace_once(s, ckpt_marker, ckpt_new, "confidence EMA checkpoint state")

    # 10) Write CSV outputs after absorber_outputs declaration.
    outputs_marker = '''    absorber_outputs: Dict[str, str] = {}\n    if absorber_logging_enabled:\n'''
    outputs_new = '''    confidence_ema_outputs: Dict[str, str] = {}\n    if confidence_ema_state is not None:\n        ema_epoch_fields = [\n            "epoch",\n            "confidence_ema_ready_rate_epoch",\n            "confidence_ema_fallback_rate_epoch",\n            "confidence_ema_update_count_mean_epoch",\n            "confidence_ema_local_explained_mean_epoch",\n            "confidence_ema_gate_explained_mean_epoch",\n            "confidence_ema_abs_local_minus_ema_mean_epoch",\n            "residual_weight_mean_epoch",\n            "residual_weight_p10_mean_epoch",\n            "residual_weight_p50_mean_epoch",\n            "residual_weight_p90_mean_epoch",\n        ]\n        _write_csv_rows(train_dir / "confidence_ema_stats_by_epoch.csv", confidence_ema_epoch_rows, ema_epoch_fields)\n        _write_csv_rows(\n            train_dir / "confidence_ema_final_snapshot.csv",\n            confidence_ema_state.final_rows(key_by_index=confidence_ema_key_by_index),\n            ["ema_index", "trajectory_key", "explained_ema", "update_count"],\n        )\n        confidence_ema_outputs = {\n            "confidence_ema_stats_by_epoch": str((Path("train") / "prealign" / "confidence_ema_stats_by_epoch.csv").as_posix()),\n            "confidence_ema_final_snapshot": str((Path("train") / "prealign" / "confidence_ema_final_snapshot.csv").as_posix()),\n        }\n\n    absorber_outputs: Dict[str, str] = {}\n    if absorber_logging_enabled:\n'''
    s = replace_once(s, outputs_marker, outputs_new, "confidence EMA CSV outputs")

    # 11) Stage summary config fields.
    config_marker = '''            "absorber_outputs": absorber_outputs,\n        } if str(args.protocol) == "soft_routing" else None,\n'''
    config_new = '''            "absorber_outputs": absorber_outputs,\n            "enable_confidence_ema": bool(confidence_ema_enabled),\n            "confidence_ema_beta": float(getattr(args, "confidence_ema_beta", 0.9)),\n            "confidence_ema_warmup_epochs": int(getattr(args, "confidence_ema_warmup_epochs", 3)),\n            "confidence_ema_min_updates": int(getattr(args, "confidence_ema_min_updates", 3)),\n            "confidence_ema_signal": str(getattr(args, "confidence_ema_signal", "local_explained")),\n            "confidence_ema_delayed_use": bool(getattr(args, "confidence_ema_delayed_use", True)),\n            "confidence_ema_outputs": confidence_ema_outputs,\n        } if str(args.protocol) == "soft_routing" else None,\n'''
    s = replace_once(s, config_marker, config_new, "confidence EMA stage summary config")

    # 12) CLI args.
    cli_marker = '''    p.add_argument("--top_absorbers_k", type=int, default=50)\n'''
    s = replace_once(s, cli_marker, cli_marker + CLI_BLOCK, "confidence EMA CLI args")

    if s == original:
        print("NOOP: target already appeared patched")
    else:
        backup = target.with_suffix(target.suffix + ".confidence_ema_patch.bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        target.write_text(s, encoding="utf-8")
        print(f"PATCHED {target}")
        print(f"BACKUP {backup}")


if __name__ == "__main__":
    main()
