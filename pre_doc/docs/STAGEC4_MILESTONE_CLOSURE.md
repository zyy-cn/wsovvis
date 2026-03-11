# Stage C4 Milestone Closure Memo

Date: 2026-03-03
Branch line: `stagec4-c43-coverage-unkfg`
Milestone: Stage C4 offline attribution expansion

## Scope closed
- C4.1: `em_v1` offline scorer backend.
- C4.2: `sinkhorn_v1` minimal offline backend.
- C4.3: spec-lock + C4.3-A (`__bg__`) + C4.3-B (`__unk_fg__` gating).

## Key design decisions (locked)
- Additive-only changes:
  - existing artifact fields/paths were preserved;
  - new fields were added without renaming/removing prior schema elements.
- Default-OFF compatibility for C4.3:
  - C4.3 config remains opt-in; disabled path is protected by parity gate.
- Row-level gating semantics:
  - under C4.3-B, `sinkhorn_active_special_columns` is allowed to vary by row;
  - `__unk_fg__` can be configured globally but inactive for rows failing gating, while `__bg__` remains present when C4.3 path is active.

## Parity hard gate role
- `test_sinkhorn_c42_parity_hard_gate_snapshot` is treated as a release-blocking regression sentinel.
- Purpose:
  - guarantee disabled-C4.3 behavior remains aligned with established C4.2 baseline.

## Determinism expectations
- Deterministic double-run equality is required for closure evidence:
  - base mode (no C4.3 flags),
  - C4.3-enabled mode including unk-fg configuration.
- Gate test:
  - `test_sinkhorn_backend_deterministic_double_run`.

## Completion vs future work
- Complete in C4:
  - offline scorer backend expansion (`em_v1`, `sinkhorn_v1`),
  - C4.3 special-column support within bounded scope (`bg`, `unk_fg` gating),
  - targeted canonical parity/bg/unk-fg and determinism pass evidence.
- Not part of C4 closure (future work):
  - C4.3-C or broader coverage/slack redesign,
  - training-loop / Stage D integration implementation,
  - any default-policy switch requiring new evidence/milestone approval.

## Canonical evidence references
- Planning/spec lock: `codex/2026030302_stagec4-c43-coverage-unkfg/01_output.txt`
- C4.3-A parity repair + canonical pass: `codex/2026030302_stagec4-c43-coverage-unkfg/05_output.txt`
- C4.3-B repair + canonical targeted pass + determinism pass: `codex/2026030302_stagec4-c43-coverage-unkfg/07_output.txt`
