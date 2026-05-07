#!/usr/bin/env python3
"""Audit whether generated Llama3 text-bank responses appear truncated.

This is a lightweight text-level audit. It does not load Llama3 or CLIP.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_root", required=True)
    ap.add_argument("--expected_max_gen_len", type=int, default=None, help="If provided and >0, report feature_count hits against this cap.")
    ap.add_argument("--print_examples", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.bank_root)
    resp = root / "llama3_responses.jsonl"
    if not resp.is_file():
        raise FileNotFoundError(resp)

    n = 0
    end_punc = 0
    hit_cap = 0
    fc_counter: Counter[int] = Counter()
    suspicious: List[Dict[str, Any]] = []
    cap = int(args.expected_max_gen_len) if args.expected_max_gen_len is not None else None

    for line in resp.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        o = json.loads(line)
        n += 1
        txt = str(o.get("generated_text", "")).strip()
        fc = int(o.get("feature_count", 0))
        fc_counter[fc] += 1
        if txt.endswith((".", "!", "?")):
            end_punc += 1
        if cap and cap > 0 and fc >= cap:
            hit_cap += 1
        # Heuristic: no terminal punctuation, ends with short function-like tail, or max cap hit.
        is_suspicious = (not txt.endswith((".", "!", "?"))) or (cap and cap > 0 and fc >= cap)
        if is_suspicious and len(suspicious) < int(args.print_examples):
            suspicious.append({
                "raw_id": o.get("raw_id"),
                "class_name": o.get("class_name"),
                "feature_count": fc,
                "tail": txt[-160:],
            })

    out = {
        "status": "PASS",
        "bank_root": str(root),
        "response_count": n,
        "feature_count_distribution_top20": dict(fc_counter.most_common(20)),
        "end_with_terminal_punctuation_count": end_punc,
        "end_with_terminal_punctuation_ratio": end_punc / max(n, 1),
        "expected_max_gen_len": cap,
        "hit_expected_max_gen_len_count": hit_cap,
        "hit_expected_max_gen_len_ratio": hit_cap / max(n, 1),
        "suspicious_examples": suspicious,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
