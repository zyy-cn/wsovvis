#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

CANDIDATE_PATTERNS = [
    "**/per_class_context_identifiability.csv",
    "**/*context*identifiability*per_class*.csv",
    "**/*gt*context*identifiability*.csv",
]

ap = argparse.ArgumentParser()
ap.add_argument("--run_root", required=True)
ap.add_argument("--repo_root", default=".")
args = ap.parse_args()
roots = [Path(args.run_root), Path(args.repo_root)]
seen = set()
for root in roots:
    if not root.exists():
        continue
    for pat in CANDIDATE_PATTERNS:
        for p in root.glob(pat):
            s = str(p.resolve())
            if s in seen:
                continue
            seen.add(s)
            print(s)
