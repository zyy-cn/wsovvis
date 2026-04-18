from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Mapping


def memory_audit_enabled() -> bool:
    value = str(os.environ.get("WSOVVIS_MEMORY_AUDIT", "")).strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def current_rss_kb() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        pass
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return -1


def shallow_size_bytes(obj: Any) -> int:
    try:
        return int(sys.getsizeof(obj))
    except Exception:
        return -1


def memory_checkpoint(label: str, **stats: Any) -> None:
    if not memory_audit_enabled():
        return
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": str(label),
        "rss_kb": int(current_rss_kb()),
    }
    for key, value in stats.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            payload[str(key)] = value
        elif isinstance(value, Mapping):
            payload[str(key)] = {str(k): v for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            payload[str(key)] = {"len": len(value), "shallow_size": shallow_size_bytes(value)}
        else:
            payload[str(key)] = {"type": type(value).__name__, "shallow_size": shallow_size_bytes(value)}
    print(f"[memory-audit] {payload}", file=sys.stderr, flush=True)


def timing_checkpoint(label: str, *, started_at: float, **stats: Any) -> None:
    if not memory_audit_enabled():
        return
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": str(label),
        "elapsed_s": float(max(0.0, time.perf_counter() - float(started_at))),
        "rss_kb": int(current_rss_kb()),
    }
    for key, value in stats.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            payload[str(key)] = value
        elif isinstance(value, Mapping):
            payload[str(key)] = {str(k): v for k, v in value.items()}
        elif isinstance(value, (list, tuple, set)):
            payload[str(key)] = {"len": len(value), "shallow_size": shallow_size_bytes(value)}
        else:
            payload[str(key)] = {"type": type(value).__name__, "shallow_size": shallow_size_bytes(value)}
    print(f"[time-audit] {payload}", file=sys.stderr, flush=True)
