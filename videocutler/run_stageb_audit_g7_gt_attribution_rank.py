from __future__ import annotations

import sys

_RETIRED_MSG = "RETIRED_USE_NEW_CHAIN: videocutler/run_stageb_audit_g7_gt_attribution_rank.py is retired for current GT-sidecar generation / audit preparation. Use videocutler/run_stageb_audit_g8_gt_sidecar.py instead."


def main() -> int:
    raise SystemExit(_RETIRED_MSG)


if __name__ == "__main__":
    raise SystemExit(main())
