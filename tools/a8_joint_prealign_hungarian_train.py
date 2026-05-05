#!/usr/bin/env python3
"""Retired invalid A8 joint Hungarian entrypoint.

The previous implementation exposed an invalid dynamic-Hungarian interface and
could be confused with the corrected train-time dynamic objective. This wrapper
is intentionally left in place to fail loudly and direct callers to the canonical
replacement.
"""
from __future__ import annotations

import sys
from pathlib import Path


REPLACEMENT = "tools/a8_joint_prealign_train_time_dynamic_hungarian.py"
ERROR_CODE = "RETIRED_INVALID_DYNAMIC_INTERFACE"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    message = f"""
{ERROR_CODE}

This entrypoint has been retired and must not be used for A8 joint training.
It previously represented an invalid/stale dynamic-Hungarian interface.

Use instead:
  python {repo_root / REPLACEMENT} [args]

Required policy:
  - train-time Hungarian assignments must be recomputed from current logits;
  - matched_pairs_csv.matched_raw_id must not be used as a training target;
  - canonical visible525 rank@K is the headline metric;
  - retired row_gap / clip-local micro_top1 must not be used as the headline.
""".strip()
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
