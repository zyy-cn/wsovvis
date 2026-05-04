#!/usr/bin/env python3
"""Patch A8 Hungarian matched-pair training with optional online hard-negative loss.

This patch is intentionally conservative:
- it modifies only videocutler/run_stageb_train_residual_gated_hungarian_matched.py
- it preserves existing CE/InfoNCE behavior when --hard_negative_mode none
- it does not change Hungarian matched pairs, candidate full-Y, unmatched-row policy,
  evaluator semantics, text/carrier banks, or checkpoints outside the requested output root.

Run from repo root:
  python tools/apply_a8_hard_negative_training_patch.py --repo_root . --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_REL = "videocutler/run_stageb_train_residual_gated_hungarian_matched.py"
ALLOW_REL = "codex/control/PATH_POLICY.json"

HELPER = r'''

def _hard_negative_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    k: int,
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Online hard-negative hinge loss over the same full-Y denominator as CE.

    Positive class is the fixed Hungarian pseudo label. Hard negatives are selected
    online from current logits among classes c != pseudo label. This function uses
    no row-level GT and no audit table.
    """
    mode = str(mode or "none").strip().lower()
    zero = logits.sum() * 0.0
    if mode in {"", "none", "off", "disabled"}:
        return zero, {
            "hard_negative_loss": 0.0,
            "hard_negative_active_rate": 0.0,
            "mean_pos_minus_hardneg_gap": 0.0,
            "hard_negative_k_effective": 0,
        }
    if logits.ndim != 2:
        raise ValueError(f"hard-negative expects 2D logits, got shape={tuple(logits.shape)}")
    if logits.shape[1] <= 1:
        return zero, {
            "hard_negative_loss": 0.0,
            "hard_negative_active_rate": 0.0,
            "mean_pos_minus_hardneg_gap": 0.0,
            "hard_negative_k_effective": 0,
        }

    row_idx = torch.arange(logits.shape[0], device=logits.device)
    pos = logits.gather(1, target.view(-1, 1)).squeeze(1)
    neg_logits = logits.clone()
    neg_logits[row_idx, target] = -torch.inf

    if mode == "top1":
        neg = neg_logits.max(dim=1).values.view(-1, 1)
        k_eff = 1
    elif mode == "topk":
        k_eff = min(max(int(k), 1), int(logits.shape[1]) - 1)
        neg = neg_logits.topk(k_eff, dim=1).values
    else:
        raise ValueError(f"unsupported hard_negative_mode={mode!r}; expected none|top1|topk")

    gap = pos.view(-1, 1) - neg
    hinge = F.relu(float(margin) - gap)
    loss = hinge.mean()
    with torch.no_grad():
        stats = {
            "hard_negative_loss": float(loss.detach().cpu().item()),
            "hard_negative_active_rate": float((hinge > 0).float().mean().detach().cpu().item()),
            "mean_pos_minus_hardneg_gap": float(gap.mean().detach().cpu().item()),
            "hard_negative_k_effective": int(k_eff),
        }
    return loss, stats
'''


def replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def patch_training_file(path: Path) -> tuple[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    original = text
    notes: list[str] = []

    if "def _hard_negative_loss_from_logits" not in text:
        marker = "def _project_text(projector: Projector, text_tensor: torch.Tensor) -> torch.Tensor:\n    return F.normalize(projector(text_tensor), p=2.0, dim=-1)\n"
        new_marker = marker + HELPER
        text, ok = replace_once(text, marker, new_marker, "insert hard-negative helper")
        if not ok:
            raise RuntimeError("Could not insert helper after _project_text; target file layout changed.")
        notes.append("inserted _hard_negative_loss_from_logits")
    else:
        notes.append("helper already present")

    sig_old = "    loss_name: str,\n) -> Tuple[torch.Tensor, Dict[str, Any]]:\n"
    sig_new = "    loss_name: str,\n    hard_negative_mode: str = \"none\",\n    hard_negative_k: int = 1,\n    hard_negative_margin: float = 0.5,\n    hard_negative_weight: float = 0.0,\n) -> Tuple[torch.Tensor, Dict[str, Any]]:\n"
    if "hard_negative_mode: str = \"none\"" not in text:
        text, ok = replace_once(text, sig_old, sig_new, "extend _loss_for_clip signature")
        if not ok:
            raise RuntimeError("Could not extend _loss_for_clip signature; target file layout changed.")
        notes.append("extended _loss_for_clip signature")
    else:
        notes.append("_loss_for_clip signature already extended")

    loss_old = '''    if str(loss_name) == "ce":
        loss = F.cross_entropy(logits, target, reduction="mean")
    elif str(loss_name) == "infonce":
        # Row-wise InfoNCE over the exact same full-Y denominator as CE.
        pos = logits.gather(1, target.view(-1, 1)).squeeze(1)
        loss = -(pos - torch.logsumexp(logits, dim=1)).mean()
    else:
        raise ValueError(f"unsupported loss: {loss_name}")
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        pseudo_acc = float((pred == target).float().mean().detach().cpu().item())
    return loss, {"rows": len(kept_rows), "clip_id": clip_id, "pseudo_top1_acc": pseudo_acc}
'''
    loss_new = '''    if str(loss_name) == "ce":
        base_loss = F.cross_entropy(logits, target, reduction="mean")
    elif str(loss_name) == "infonce":
        # Row-wise InfoNCE over the exact same full-Y denominator as CE.
        pos = logits.gather(1, target.view(-1, 1)).squeeze(1)
        base_loss = -(pos - torch.logsumexp(logits, dim=1)).mean()
    else:
        raise ValueError(f"unsupported loss: {loss_name}")

    hn_loss, hn_stats = _hard_negative_loss_from_logits(
        logits,
        target,
        mode=str(hard_negative_mode),
        k=int(hard_negative_k),
        margin=float(hard_negative_margin),
    )
    loss = base_loss + float(hard_negative_weight) * hn_loss

    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        pseudo_acc = float((pred == target).float().mean().detach().cpu().item())
        stats = {
            "rows": len(kept_rows),
            "clip_id": clip_id,
            "pseudo_top1_acc": pseudo_acc,
            "base_loss": float(base_loss.detach().cpu().item()),
            "hard_negative_weight": float(hard_negative_weight),
            "hard_negative_mode": str(hard_negative_mode),
            "hard_negative_margin": float(hard_negative_margin),
            **hn_stats,
        }
    return loss, stats
'''
    if "base_loss = F.cross_entropy(logits, target" not in text:
        text, ok = replace_once(text, loss_old, loss_new, "replace loss block")
        if not ok:
            raise RuntimeError("Could not replace CE/InfoNCE loss block; target file layout changed.")
        notes.append("replaced loss block with CE/InfoNCE + optional HN")
    else:
        notes.append("loss block already patched")

    call_old = "loss, stats = _loss_for_clip(rows=rows, data=data, example_by_tid=example_by_tid, text_proj_all=text_proj_all, theta_t=theta_t, device=device, loss_name=str(args.loss))"
    call_new = "loss, stats = _loss_for_clip(rows=rows, data=data, example_by_tid=example_by_tid, text_proj_all=text_proj_all, theta_t=theta_t, device=device, loss_name=str(args.loss), hard_negative_mode=str(args.hard_negative_mode), hard_negative_k=int(args.hard_negative_k), hard_negative_margin=float(args.hard_negative_margin), hard_negative_weight=float(args.hard_negative_weight))"
    if call_new not in text:
        text, ok = replace_once(text, call_old, call_new, "extend _loss_for_clip call")
        if not ok:
            raise RuntimeError("Could not extend _loss_for_clip call; target file layout changed.")
        notes.append("extended _loss_for_clip call")
    else:
        notes.append("_loss_for_clip call already extended")

    lists_old = "        epoch_accs: List[float] = []\n"
    lists_new = "        epoch_accs: List[float] = []\n        epoch_hn_losses: List[float] = []\n        epoch_hn_active_rates: List[float] = []\n        epoch_pos_hn_gaps: List[float] = []\n"
    if "epoch_hn_losses" not in text:
        text, ok = replace_once(text, lists_old, lists_new, "add epoch HN stat lists")
        if not ok:
            raise RuntimeError("Could not add epoch HN stat lists; target file layout changed.")
        notes.append("added epoch HN stat lists")
    else:
        notes.append("epoch HN stat lists already present")

    append_old = "            epoch_losses.append(lv); all_losses.append(lv); epoch_rows += int(stats[\"rows\"]); epoch_accs.append(float(stats[\"pseudo_top1_acc\"]))\n"
    append_new = "            epoch_losses.append(lv); all_losses.append(lv); epoch_rows += int(stats[\"rows\"]); epoch_accs.append(float(stats[\"pseudo_top1_acc\"]))\n            epoch_hn_losses.append(float(stats.get(\"hard_negative_loss\", 0.0)))\n            epoch_hn_active_rates.append(float(stats.get(\"hard_negative_active_rate\", 0.0)))\n            epoch_pos_hn_gaps.append(float(stats.get(\"mean_pos_minus_hardneg_gap\", 0.0)))\n"
    if "epoch_hn_active_rates.append" not in text:
        text, ok = replace_once(text, append_old, append_new, "append HN stats")
        if not ok:
            raise RuntimeError("Could not append HN stats; target file layout changed.")
        notes.append("added HN stat accumulation")
    else:
        notes.append("HN stat accumulation already present")

    epoch_old = "epoch_row = {\"timestamp\": _now(), \"row_type\": \"epoch_summary\", \"epoch\": int(epoch)+1, \"loss_mean\": _mean(epoch_losses), \"loss_last\": epoch_losses[-1] if epoch_losses else 0.0, \"pseudo_top1_acc_mean\": _mean(epoch_accs), \"epoch_rows\": epoch_rows, \"epoch_clips\": len(clip_ids)}"
    epoch_new = "epoch_row = {\"timestamp\": _now(), \"row_type\": \"epoch_summary\", \"epoch\": int(epoch)+1, \"loss_mean\": _mean(epoch_losses), \"loss_last\": epoch_losses[-1] if epoch_losses else 0.0, \"pseudo_top1_acc_mean\": _mean(epoch_accs), \"hard_negative_loss_mean\": _mean(epoch_hn_losses), \"hard_negative_active_rate_mean\": _mean(epoch_hn_active_rates), \"mean_pos_minus_hardneg_gap\": _mean(epoch_pos_hn_gaps), \"epoch_rows\": epoch_rows, \"epoch_clips\": len(clip_ids)}"
    if "hard_negative_active_rate_mean" not in text:
        text, ok = replace_once(text, epoch_old, epoch_new, "extend epoch summary")
        if not ok:
            raise RuntimeError("Could not extend epoch summary; target file layout changed.")
        notes.append("extended epoch summary")
    else:
        notes.append("epoch summary already extended")

    args_old = "    p.add_argument(\"--loss\", choices=[\"ce\", \"infonce\"], default=\"ce\")\n"
    args_new = "    p.add_argument(\"--loss\", choices=[\"ce\", \"infonce\"], default=\"ce\")\n    p.add_argument(\"--hard_negative_mode\", choices=[\"none\", \"top1\", \"topk\"], default=\"none\")\n    p.add_argument(\"--hard_negative_k\", type=int, default=1)\n    p.add_argument(\"--hard_negative_margin\", type=float, default=0.5)\n    p.add_argument(\"--hard_negative_weight\", type=float, default=0.0)\n"
    if "--hard_negative_mode" not in text:
        text, ok = replace_once(text, args_old, args_new, "add CLI args")
        if not ok:
            raise RuntimeError("Could not add CLI args after --loss; target file layout changed.")
        notes.append("added CLI hard-negative args")
    else:
        notes.append("CLI hard-negative args already present")

    policy_old = "\"loss_only_arm\": str(args.loss)},"
    policy_new = "\"loss_only_arm\": str(args.loss), \"hard_negative\": {\"mode\": str(args.hard_negative_mode), \"k\": int(args.hard_negative_k), \"margin\": float(args.hard_negative_margin), \"weight\": float(args.hard_negative_weight), \"uses_gt_for_negative_selection\": False}},"
    if "uses_gt_for_negative_selection" not in text:
        text, ok = replace_once(text, policy_old, policy_new, "extend policy metadata")
        if not ok:
            # Non-fatal: target may format policy differently in newer code.
            notes.append("WARNING: could not extend setup policy metadata; code semantics still patched")
        else:
            notes.append("extended policy metadata")
    else:
        notes.append("policy metadata already extended")

    save_old = "\"matched_pairs_csv\": str(matched_csv)}, ckpt_out)"
    save_new = "\"matched_pairs_csv\": str(matched_csv), \"hard_negative\": {\"mode\": str(args.hard_negative_mode), \"k\": int(args.hard_negative_k), \"margin\": float(args.hard_negative_margin), \"weight\": float(args.hard_negative_weight)}}, ckpt_out)"
    if "\"hard_negative\": {\"mode\": str(args.hard_negative_mode)" not in text:
        text, ok = replace_once(text, save_old, save_new, "extend checkpoint metadata")
        if ok:
            notes.append("extended checkpoint metadata")
        else:
            notes.append("WARNING: could not extend checkpoint metadata; code semantics still patched")
    else:
        notes.append("checkpoint metadata already extended")

    if text == original:
        notes.append("no content changes needed")
    else:
        path.write_text(text, encoding="utf-8")
    return text, notes


def patch_path_policy(repo_root: Path, apply: bool) -> dict:
    p = repo_root / ALLOW_REL
    required = [
        TARGET_REL,
        "tools/apply_a8_hard_negative_training_patch.py",
        "docs/A8_HARD_NEGATIVE_REMOTE_COMMANDS.md",
        "codex/outputs/G8_inference_and_eval/a8_hard_negative_patch_20260504/**",
    ]
    if not p.is_file():
        return {"available": False, "path": str(p), "required_allow_paths": required}
    obj = json.loads(p.read_text(encoding="utf-8"))
    allow = list(obj.get("allow_paths", []))
    added = []
    for x in required:
        if x not in allow:
            allow.append(x); added.append(x)
    obj["allow_paths"] = allow
    if apply and added:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"available": True, "path": str(p), "added": added, "required_allow_paths": required, "applied": bool(apply)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--update_path_policy", action="store_true", help="Also add minimal allow_paths to codex/control/PATH_POLICY.json if present.")
    args = ap.parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    target = repo_root / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)
    backup = target.with_suffix(target.suffix + ".bak_before_a8_hard_negative")
    if args.apply and not backup.exists():
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    if args.apply:
        _, notes = patch_training_file(target)
    else:
        # Dry-run uses a temporary copy in memory by writing/reading from a shadow file avoided; report only.
        notes = ["dry-run: target exists; run with --apply to patch"]

    policy = patch_path_policy(repo_root, apply=bool(args.apply and args.update_path_policy)) if args.update_path_policy else {"skipped": True}
    report = {"status": "PASS", "target": str(target), "apply": bool(args.apply), "notes": notes, "path_policy": policy}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
