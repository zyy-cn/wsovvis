# Session Handoff: Stage C4 Closure -> Next Step Continuation

## Actual completed milestone state (reconciled)
- Stage C1 complete (MIL-first offline baseline).
- Stage C1.r1 complete (real-artifact smoke + diagnostics hardening + determinism).
- Stage C2 complete (labelset_proto_v1 scorer baseline + canonical remote pytest + real-artifact smoke + determinism).
- Stage C3 complete (global decoder comparison closure; default decoder remains `independent`).
- Stage C4 complete (offline attribution expansion line):
  - C4.1 `em_v1` offline scorer backend.
  - C4.2 `sinkhorn_v1` minimal offline backend.
  - C4.3 spec-lock + C4.3-A (`__bg__`) + C4.3-B (`__unk_fg__` gating).

## Canonical remote validation discipline (must preserve)
- Canonical remote host alias: `gpu4090d`
- Canonical remote runner repo: `/home/zyy/code/wsovvis_runner`
- PASS claims require explicit branch/commit match:
  - intended local/pushed commit hash must equal remote checked-out `HEAD` hash.
- `PYTHONPATH` wiring requirement:
  - set in both `--env-cmd` and `--cmd` for remote verify commands.
- Real-artifact smoke discipline:
  - use `tools/remote_verify_wsovvis.sh --keep-untracked` when untracked runner symlinks are needed.
  - run symlink preflight/relink guards for `runs`/`outputs` when applicable.

## C4 closure decisions that must remain stable
- Additive-only evolution:
  - no removal/rename of existing Stage C artifact fields.
  - C4 diagnostics/fields remain additive.
- Default-OFF compatibility:
  - C4.3 flags default OFF and must not perturb C4.2 behavior.
- Row-level gating semantics (C4.3-B):
  - `sinkhorn_active_special_columns` can be row-dependent under gating.
  - valid rows may expose `["__bg__"]` or `["__bg__", "__unk_fg__"]` depending on effective activation.
- Parity hard gate role:
  - `test_sinkhorn_c42_parity_hard_gate_snapshot` is a required regression sentinel for disabled-C4.3 path.
- Determinism expectation:
  - `test_sinkhorn_backend_deterministic_double_run` must pass in both base and C4.3-enabled configurations.

## Closure evidence pointers
- C4 planning/spec lock: `codex/2026030302_stagec4-c43-coverage-unkfg/01_output.txt`
- C4.3-A parity repair + canonical pass: `codex/2026030302_stagec4-c43-coverage-unkfg/05_output.txt`
- C4.3-B repair + targeted canonical pass + determinism pass: `codex/2026030302_stagec4-c43-coverage-unkfg/07_output.txt`

## Next-step continuation (single concrete target)
- Stage D planning/spec-lock task:
  - define training-loop integration boundary for Stage C outputs,
  - lock acceptance gates (parity/determinism/schema compatibility) before implementation,
  - keep C4 scorer behavior/defaults unchanged unless a new gated milestone explicitly authorizes changes.
