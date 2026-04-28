from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Record = Dict[str, Any]


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_json(path: Path) -> Optional[Record]:
    try:
        if path.is_file():
            obj = json.loads(path.read_text(encoding='utf-8'))
            return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def _slope(points: Sequence[Tuple[float, Optional[float]]]) -> Optional[float]:
    vals = [(float(x), float(y)) for x, y in points if y is not None and math.isfinite(float(y))]
    if len(vals) < 2:
        return None
    xs = [p[0] for p in vals]
    ys = [p[1] for p in vals]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom <= 1e-12:
        return None
    return sum((x - xm) * (y - ym) for x, y in vals) / denom


def _read_epoch_rows(runtime_path: Path) -> List[Record]:
    rows: List[Record] = []
    for obj in _iter_jsonl(runtime_path):
        if str(obj.get('row_type')) != 'epoch_summary':
            continue
        epoch = _safe_int(obj.get('epoch'))
        if epoch is None:
            continue
        row: Record = {'epoch': epoch}
        # Core optimization curves.
        for src, dst in [
            ('loss_mean', 'loss_mean'),
            ('loss_last', 'loss_last'),
            ('optimization_loss_mean', 'optimization_loss_mean'),
            ('support_null_active_epoch', 'support_null_active'),
            ('support_epoch_index_epoch', 'support_epoch_index'),
            ('null_mass_mean_epoch', 'null_mass_mean'),
            ('nonnull_mass_mean_epoch', 'nonnull_mass_mean'),
            ('null_demand_mean_epoch', 'null_demand_mean'),
            ('null_residual_uncapped_mean_epoch', 'null_residual_uncapped_mean'),
            ('null_cap_mean_epoch', 'null_cap_mean'),
            ('yprime_demand_mean_epoch', 'yprime_demand_mean'),
            ('yprime_low_demand_rate_epoch', 'yprime_low_demand_rate'),
            ('null_collapse_guard_triggered_epoch', 'null_collapse_guard_triggered'),
            ('support_demand_guard_triggered_epoch', 'support_demand_guard_triggered'),
            ('sinkhorn_load_gini_epoch', 'sinkhorn_load_gini'),
            ('sinkhorn_column_coverage_mean_epoch', 'sinkhorn_column_coverage_mean'),
            ('sinkhorn_column_coverage_min_epoch', 'sinkhorn_column_coverage_min'),
            ('sinkhorn_max_row_load_mean_epoch', 'sinkhorn_max_row_load_mean'),
            ('sinkhorn_mean_row_load_epoch', 'sinkhorn_mean_row_load'),
            ('sinkhorn_effective_num_trajectories_epoch', 'sinkhorn_effective_num_trajectories'),
        ]:
            row[dst] = _safe_float(obj.get(src))
        rows.append(row)
    rows.sort(key=lambda r: int(r['epoch']))
    return rows


def _window_mean(rows: Sequence[Record], key: str, n: int, *, head: bool) -> Optional[float]:
    if not rows:
        return None
    chosen = list(rows[:n] if head else rows[-n:])
    return _mean([_safe_float(r.get(key)) for r in chosen])


def _trend_for(rows: Sequence[Record], key: str, window: int) -> Record:
    active_rows = [r for r in rows if (_safe_float(r.get('support_null_active'), 0.0) or 0.0) >= 0.5]
    use_rows = active_rows if active_rows else list(rows)
    first = _window_mean(use_rows, key, window, head=True)
    last = _window_mean(use_rows, key, window, head=False)
    delta = None if first is None or last is None else float(last - first)
    slope_all = _slope([(float(r['epoch']), _safe_float(r.get(key))) for r in use_rows])
    return {
        'metric': key,
        'active_epoch_count': len(active_rows),
        'first_window_mean': first,
        'last_window_mean': last,
        'last_minus_first': delta,
        'slope_per_epoch': slope_all,
    }


def _read_text_hubness(run_root: Path, dataset: str, stage: str) -> Record:
    p = run_root / 'analysis' / 'text_projector_hubness' / dataset / stage / 'stage_comparison_summary.csv'
    if not p.is_file():
        return {'exists': False, 'path': str(p)}
    with p.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    row = rows[0] if rows else {}
    out: Record = {'exists': True, 'path': str(p)}
    for k in [
        'row_count',
        'positive_text_margin_gt_vs_nearest_hub_rate',
        'positive_text_margin_gt_vs_person_rate',
        'mean_text_margin_gt_vs_person',
        'mean_text_gt_rank_full_vocab',
        'median_text_gt_rank_full_vocab',
        'text_top1_is_hub_rate',
    ]:
        out[k] = _safe_float(row.get(k)) if k != 'row_count' else _safe_int(row.get(k))
    return out


def _read_responsibility_summary(run_root: Path, dataset: str, stage: str) -> Record:
    p = run_root / 'analysis' / 'support_null_responsibility' / dataset / stage / 'summary.json'
    obj = _load_json(p)
    if not obj:
        return {'exists': False, 'path': str(p)}
    out: Record = {'exists': True, 'path': str(p)}
    for k in [
        'overall_null_mass_mean',
        'overall_top1_null_rate',
        'sinkhorn_yprime_true_support_mass_mean',
        'sinkhorn_yprime_true_support_top1_rate',
        'sinkhorn_yprime_hub_hijack_rate',
        'supported_yprime_demand_mean',
        'unsupported_yprime_demand_mean',
        'support_demand_gap_supported_minus_unsupported',
        'supported_yprime_low_demand_rate',
        'unsupported_yprime_low_demand_rate',
    ]:
        out[k] = _safe_float(obj.get(k))
    bucket = {}
    for row in obj.get('bucket_summary', []) if isinstance(obj.get('bucket_summary'), list) else []:
        if not isinstance(row, dict):
            continue
        b = str(row.get('bucket'))
        bucket[b] = {
            'mean_null_mass': _safe_float(row.get('mean_null_mass')),
            'median_null_mass': _safe_float(row.get('median_null_mass')),
            'top1_null_rate': _safe_float(row.get('top1_null_rate')),
            'row_count': _safe_int(row.get('row_count')),
        }
    out['bucket_summary'] = bucket
    return out


def _read_yprime_summary(run_root: Path, dataset: str, stage: str) -> Record:
    p = run_root / 'analysis' / 'yprime_support_coverage' / dataset / stage / 'summary.json'
    obj = _load_json(p)
    if not obj:
        return {'exists': False, 'path': str(p)}
    out: Record = {'exists': True, 'path': str(p)}
    for k in [
        'clip_yprime_pair_count',
        'yprime_trajectory_support_rate',
        'clip_all_yprime_supported_rate',
        'support_exists_but_person_higher_rate',
        'sinkhorn_yprime_true_support_mass_mean',
        'sinkhorn_yprime_true_support_top1_rate',
        'sinkhorn_yprime_hub_hijack_rate',
    ]:
        out[k] = _safe_float(obj.get(k)) if k not in {'clip_yprime_pair_count'} else _safe_int(obj.get(k))
    out['failure_bucket_counts'] = obj.get('failure_bucket_counts')
    return out


def _derive_verdict(epoch_rows: Sequence[Record], trends: Mapping[str, Record], resp: Mapping[str, Any], text: Mapping[str, Any]) -> Record:
    loss_slope = _safe_float(trends.get('loss_mean', {}).get('slope_per_epoch'))
    null_slope = _safe_float(trends.get('null_mass_mean', {}).get('slope_per_epoch'))
    yprime_matched_null = None
    bucket = resp.get('bucket_summary') if isinstance(resp.get('bucket_summary'), dict) else {}
    if isinstance(bucket, dict) and isinstance(bucket.get('yprime_matched_gt'), dict):
        yprime_matched_null = _safe_float(bucket['yprime_matched_gt'].get('mean_null_mass'))
    true_mass = _safe_float(resp.get('sinkhorn_yprime_true_support_mass_mean'))
    hub_hijack = _safe_float(resp.get('sinkhorn_yprime_hub_hijack_rate'))
    text_hub = _safe_float(text.get('text_top1_is_hub_rate'))

    signs: List[str] = []
    if loss_slope is not None and loss_slope < 0:
        signs.append('loss_decreasing')
    if null_slope is not None and abs(null_slope) < 0.005:
        signs.append('null_mass_plateau')
    elif null_slope is not None and null_slope > 0:
        signs.append('null_mass_increasing')
    if yprime_matched_null is not None and yprime_matched_null >= 0.55:
        signs.append('yprime_matched_gt_overabsorbed_by_null')
    if true_mass is not None and true_mass < 0.55:
        signs.append('true_support_mass_low')
    if hub_hijack is not None and hub_hijack >= 0.20:
        signs.append('hub_hijack_high')
    if text_hub is not None and text_hub >= 0.20:
        signs.append('text_top1_hub_high')

    if 'loss_decreasing' in signs and ('yprime_matched_gt_overabsorbed_by_null' in signs or 'hub_hijack_high' in signs):
        verdict = 'loss_decreases_but_oracle_assignment_not_good_enough_extra_epochs_unproven'
    elif 'null_mass_increasing' in signs:
        verdict = 'training_trend_risks_stronger_null_absorption'
    elif true_mass is not None and true_mass >= 0.60 and (hub_hijack is None or hub_hijack < 0.15):
        verdict = 'assignment_quality_looks_promising_extra_epochs_may_help'
    else:
        verdict = 'inconclusive_needs_snapshot_oracle_trend'
    return {
        'verdict': verdict,
        'signals': signs,
        'limitation': 'This audit can read epoch-wise optimization/null/demand curves from runtime_metrics. Oracle true-support responsibility is final-snapshot only unless training writes per-epoch responsibility snapshots or checkpoints.',
    }


def run(args: argparse.Namespace) -> Record:
    run_root = Path(args.run_root).expanduser().resolve()
    dataset = str(args.dataset_name)
    stage = str(args.stage)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_root / 'analysis' / 'support_null_training_dynamics' / dataset / stage
    runtime_path = run_root / 'train' / stage / 'runtime_metrics.jsonl'
    epoch_rows = _read_epoch_rows(runtime_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'epoch','loss_mean','loss_last','optimization_loss_mean','support_null_active','support_epoch_index',
        'null_mass_mean','nonnull_mass_mean','null_demand_mean','null_residual_uncapped_mean','null_cap_mean',
        'yprime_demand_mean','yprime_low_demand_rate','null_collapse_guard_triggered','support_demand_guard_triggered',
        'sinkhorn_load_gini','sinkhorn_column_coverage_mean','sinkhorn_column_coverage_min','sinkhorn_max_row_load_mean',
        'sinkhorn_mean_row_load','sinkhorn_effective_num_trajectories',
    ]
    _write_csv(output_dir / 'epoch_dynamics.csv', epoch_rows, fieldnames)

    trend_metrics = ['loss_mean','null_mass_mean','nonnull_mass_mean','yprime_demand_mean','yprime_low_demand_rate','sinkhorn_load_gini','sinkhorn_column_coverage_mean']
    trend_rows = [_trend_for(epoch_rows, key, int(args.window)) for key in trend_metrics]
    _write_csv(output_dir / 'trend_summary.csv', trend_rows, ['metric','active_epoch_count','first_window_mean','last_window_mean','last_minus_first','slope_per_epoch'])

    resp = _read_responsibility_summary(run_root, dataset, stage)
    yprime = _read_yprime_summary(run_root, dataset, stage)
    text = _read_text_hubness(run_root, dataset, stage)
    verdict = _derive_verdict(epoch_rows, {r['metric']: r for r in trend_rows}, resp, text)

    active_rows = [r for r in epoch_rows if (_safe_float(r.get('support_null_active'), 0.0) or 0.0) >= 0.5]
    summary: Record = {
        'status': 'PASS' if epoch_rows else 'MISSING_RUNTIME_METRICS',
        'audit_name': 'support_null_training_dynamics_audit',
        'run_root': str(run_root),
        'dataset_name': dataset,
        'stage': stage,
        'runtime_metrics_path': str(runtime_path),
        'epoch_count': len(epoch_rows),
        'support_null_active_epoch_count': len(active_rows),
        'first_support_null_active_epoch': int(active_rows[0]['epoch']) if active_rows else None,
        'last_epoch': int(epoch_rows[-1]['epoch']) if epoch_rows else None,
        'trend_window': int(args.window),
        'trends': {r['metric']: r for r in trend_rows},
        'final_responsibility_summary': resp,
        'final_yprime_support_summary': yprime,
        'final_text_hubness_summary': text,
        'verdict': verdict,
        'outputs': {
            'summary': str(output_dir / 'summary.json'),
            'epoch_dynamics': str(output_dir / 'epoch_dynamics.csv'),
            'trend_summary': str(output_dir / 'trend_summary.csv'),
            'takeover': str(output_dir / 'SUPPORT_NULL_TRAINING_DYNAMICS_TAKEOVER.md'),
        },
    }
    _write_json(output_dir / 'summary.json', summary)

    def fmt(v: Any) -> str:
        return 'None' if v is None else str(v)

    lines = [
        '# Support-Null Training Dynamics Audit',
        '',
        f'- status: `{summary["status"]}`',
        f'- run_root: `{run_root}`',
        f'- dataset: `{dataset}`',
        f'- stage: `{stage}`',
        f'- epoch_count: `{summary["epoch_count"]}`',
        f'- first_support_null_active_epoch: `{summary["first_support_null_active_epoch"]}`',
        f'- verdict: `{verdict["verdict"]}`',
        f'- signals: `{", ".join(verdict["signals"])}`',
        '',
        '## Key epoch trends',
    ]
    for row in trend_rows:
        lines.append(f'- {row["metric"]}: first_window=`{fmt(row["first_window_mean"])}`, last_window=`{fmt(row["last_window_mean"])}`, delta=`{fmt(row["last_minus_first"])}`, slope=`{fmt(row["slope_per_epoch"])}`')
    lines.extend([
        '',
        '## Final oracle snapshot reminders',
        f'- yprime_matched_gt_mean_null_mass: `{fmt((resp.get("bucket_summary", {}).get("yprime_matched_gt", {}) if isinstance(resp.get("bucket_summary"), dict) else {}).get("mean_null_mass"))}`',
        f'- unmatched_mean_null_mass: `{fmt((resp.get("bucket_summary", {}).get("unmatched_or_no_auditable_gt", {}) if isinstance(resp.get("bucket_summary"), dict) else {}).get("mean_null_mass"))}`',
        f'- true_support_mass_mean: `{fmt(resp.get("sinkhorn_yprime_true_support_mass_mean"))}`',
        f'- true_support_top1_rate: `{fmt(resp.get("sinkhorn_yprime_true_support_top1_rate"))}`',
        f'- hub_hijack_rate: `{fmt(resp.get("sinkhorn_yprime_hub_hijack_rate"))}`',
        f'- text_top1_is_hub_rate: `{fmt(text.get("text_top1_is_hub_rate"))}`',
        '',
        '## Limitation',
        f'- {verdict["limitation"]}',
        '',
        '## Outputs',
        f'- summary: `{output_dir / "summary.json"}`',
        f'- epoch_dynamics: `{output_dir / "epoch_dynamics.csv"}`',
        f'- trend_summary: `{output_dir / "trend_summary.csv"}`',
    ])
    (output_dir / 'SUPPORT_NULL_TRAINING_DYNAMICS_TAKEOVER.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': summary['status'],
        'output_dir': str(output_dir),
        'epoch_count': summary['epoch_count'],
        'first_support_null_active_epoch': summary['first_support_null_active_epoch'],
        'verdict': verdict['verdict'],
        'signals': verdict['signals'],
    }, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Audit support-null epoch dynamics and final oracle responsibility snapshot.')
    p.add_argument('--run_root', required=True, type=Path)
    p.add_argument('--dataset_name', default='lvvis_train_base')
    p.add_argument('--stage', default='prealign')
    p.add_argument('--output_dir', default=None, type=Path)
    p.add_argument('--window', default=3, type=int)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
