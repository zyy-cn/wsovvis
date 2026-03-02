#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.track_feature_export import (
    StageC1MilConfig,
    run_stagec1_mil_baseline_offline,
)


def _parse_supported_statuses(raw: str) -> tuple[str, ...]:
    statuses = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not statuses:
        raise argparse.ArgumentTypeError("--supported-video-statuses must provide at least one status")
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage C1 MIL-first offline attribution baseline scoring")
    parser.add_argument("--split-root", type=Path, required=True, help="Stage B export split root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for Stage C1 artifacts")
    parser.add_argument(
        "--scorer-backend",
        choices=("mil_v1", "labelset_proto_v1", "em_v1", "sinkhorn_v1"),
        default="mil_v1",
        help="Scoring backend (default preserves StageC1 MIL behavior)",
    )
    parser.add_argument(
        "--decoder-backend",
        choices=("independent", "coverage_greedy_v1", "otlite_v1"),
        default="independent",
        help="Decoder backend applied after scorer output (default preserves independent behavior)",
    )
    parser.add_argument("--decoder-fg-score-min", type=float, default=-1.0, help="Minimum foreground score for decoder assignment")
    parser.add_argument(
        "--decoder-bg-score-threshold",
        type=float,
        default=None,
        help="If set, assign decoder background when top foreground score is below this threshold",
    )
    parser.add_argument(
        "--decoder-bg-min-margin",
        type=float,
        default=None,
        help="If set, assign decoder background when top1-top2 margin is below this threshold",
    )
    parser.add_argument(
        "--decoder-otlite-temperature",
        type=float,
        default=0.10,
        help="OT-lite temperature (>0); lower values make assignments sharper",
    )
    parser.add_argument(
        "--decoder-otlite-iters",
        type=int,
        default=8,
        help="OT-lite fixed row/column normalization iteration count (>=1)",
    )
    parser.add_argument(
        "--decoder-otlite-eps",
        type=float,
        default=1e-12,
        help="OT-lite numerical epsilon (>0)",
    )
    parser.add_argument(
        "--decoder-otlite-ot-prob-min",
        type=float,
        default=None,
        help="If set, OT-lite assigns BG when selected OT probability is below this threshold",
    )
    parser.add_argument("--embedding-abs-mean-weight", type=float, default=1.0)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--length-log-weight", type=float, default=0.25)
    parser.add_argument("--top-k-per-video", type=int, default=3)
    parser.add_argument(
        "--supported-video-statuses",
        type=_parse_supported_statuses,
        default=("processed_with_tracks", "processed_zero_tracks"),
        help="Comma-separated statuses accepted for processed videos",
    )
    parser.add_argument("--labelset-json", type=Path, default=None, help="Per-video labelset JSON input")
    parser.add_argument("--prototype-manifest-json", type=Path, default=None, help="Prototype manifest JSON input")
    parser.add_argument("--labelset-key", type=str, default="label_set_observed_ids", help="Key used to read per-video labelset IDs")
    parser.add_argument(
        "--empty-labelset-policy",
        choices=("use_all_prototypes", "error"),
        default="use_all_prototypes",
        help="Behavior when labelset/prototype intersection is empty",
    )
    parser.add_argument("--em-temperature", type=float, default=0.10, help="EM scorer temperature (>0)")
    parser.add_argument("--em-iterations", type=int, default=5, help="EM scorer iteration count (>=1)")
    parser.add_argument("--em-prior-alpha", type=float, default=1.0, help="EM scorer Dirichlet prior alpha (>=0)")
    parser.add_argument("--em-eps", type=float, default=1e-12, help="EM scorer numerical epsilon (>0)")
    parser.add_argument("--sinkhorn-temperature", type=float, default=0.10, help="Sinkhorn scorer temperature (>0)")
    parser.add_argument("--sinkhorn-iterations", type=int, default=12, help="Sinkhorn scorer iteration count (>=1)")
    parser.add_argument("--sinkhorn-tolerance", type=float, default=1e-6, help="Sinkhorn convergence tolerance (>=0)")
    parser.add_argument("--sinkhorn-eps", type=float, default=1e-12, help="Sinkhorn numerical epsilon (>0)")
    parser.add_argument(
        "--sinkhorn-c43-enable",
        action="store_true",
        help="Enable sinkhorn C4.3 additive schema (C4.3-A supports bg-only path)",
    )
    parser.add_argument(
        "--sinkhorn-c43-enable-bg",
        action="store_true",
        help="Enable sinkhorn C4.3 bg special column (requires --sinkhorn-c43-enable)",
    )
    parser.add_argument(
        "--sinkhorn-c43-enable-unk-fg",
        action="store_true",
        help="Reserved for C4.3-B; C4.3-A rejects this when set",
    )
    parser.add_argument(
        "--sinkhorn-c43-bg-prior-weight",
        type=float,
        default=0.0,
        help="C4.3 bg column prior weight (must be >0 when bg column is enabled)",
    )
    parser.add_argument(
        "--sinkhorn-c43-unk-fg-prior-weight",
        type=float,
        default=0.0,
        help="Reserved for C4.3-B; C4.3-A expects 0",
    )
    parser.add_argument(
        "--sinkhorn-c43-unk-fg-min-top-obs-score",
        type=float,
        default=None,
        help="Reserved for C4.3-B; C4.3-A expects null",
    )
    parser.add_argument(
        "--sinkhorn-c43-unk-fg-max-top-obs-score",
        type=float,
        default=None,
        help="Reserved for C4.3-B; C4.3-A expects null",
    )
    parser.add_argument(
        "--sinkhorn-c43-bg-score",
        type=float,
        default=0.0,
        help="Raw score used for sinkhorn C4.3 bg special column",
    )
    parser.add_argument("--no-eager-validate", action="store_true", help="Disable eager Stage C0 shard validation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = StageC1MilConfig(
        embedding_abs_mean_weight=args.embedding_abs_mean_weight,
        objectness_weight=args.objectness_weight,
        length_log_weight=args.length_log_weight,
        top_k_per_video=args.top_k_per_video,
        supported_video_statuses=args.supported_video_statuses,
    )

    report = run_stagec1_mil_baseline_offline(
        split_root=args.split_root,
        output_dir=args.output_dir,
        config=config,
        eager_validate=not args.no_eager_validate,
        scorer_backend=args.scorer_backend,
        decoder_backend=args.decoder_backend,
        decoder_fg_score_min=args.decoder_fg_score_min,
        decoder_bg_score_threshold=args.decoder_bg_score_threshold,
        decoder_bg_min_margin=args.decoder_bg_min_margin,
        decoder_otlite_temperature=args.decoder_otlite_temperature,
        decoder_otlite_iters=args.decoder_otlite_iters,
        decoder_otlite_eps=args.decoder_otlite_eps,
        decoder_otlite_ot_prob_min=args.decoder_otlite_ot_prob_min,
        labelset_json=args.labelset_json,
        prototype_manifest_json=args.prototype_manifest_json,
        labelset_key=args.labelset_key,
        empty_labelset_policy=args.empty_labelset_policy,
        em_temperature=args.em_temperature,
        em_iterations=args.em_iterations,
        em_prior_alpha=args.em_prior_alpha,
        em_eps=args.em_eps,
        sinkhorn_temperature=args.sinkhorn_temperature,
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_tolerance=args.sinkhorn_tolerance,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_c43_enable=args.sinkhorn_c43_enable,
        sinkhorn_c43_enable_bg=args.sinkhorn_c43_enable_bg,
        sinkhorn_c43_enable_unk_fg=args.sinkhorn_c43_enable_unk_fg,
        sinkhorn_c43_bg_prior_weight=args.sinkhorn_c43_bg_prior_weight,
        sinkhorn_c43_unk_fg_prior_weight=args.sinkhorn_c43_unk_fg_prior_weight,
        sinkhorn_c43_unk_fg_min_top_obs_score=args.sinkhorn_c43_unk_fg_min_top_obs_score,
        sinkhorn_c43_unk_fg_max_top_obs_score=args.sinkhorn_c43_unk_fg_max_top_obs_score,
        sinkhorn_c43_bg_score=args.sinkhorn_c43_bg_score,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
