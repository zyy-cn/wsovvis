#!/usr/bin/env python3
from __future__ import annotations
import runpy, sys
if "--only" not in sys.argv:
    sys.argv.extend(["--only", "projector_distortion"])
runpy.run_path(__file__.replace("a8_projector_distortion_audit.py", "a8_manifold_alignment_diagnosis.py"), run_name="__main__")
