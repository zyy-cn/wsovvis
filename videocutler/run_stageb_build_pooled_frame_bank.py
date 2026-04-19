from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "run_stageb_build_pooled_frame_bank.py is deprecated: pooled_frame_records.jsonl is retired and no longer part of the canonical runtime chain.\n"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
