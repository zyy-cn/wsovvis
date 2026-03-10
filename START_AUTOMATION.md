# Start WSOVVIS Automation

Deploy this kit at the **repository root** of a clean WSOVVIS code checkout.

Required paths after deployment:
- `AGENTS.md`
- `.codex/config.toml`
- `docs/mainline/*`
- `.agents/skills/wsovvis-phase-gate-check/SKILL.md`
- `.agents/skills/wsovvis-eval-acceptance/SKILL.md`
- optional but recommended: `.agents/skills/wsovvis-mainline-supervisor/SKILL.md`

## First run
From the repo root:

```bash
codex
```

Then paste:

```text
Read AGENTS.md first, then read docs/mainline/INDEX.md and all referenced mainline documents.
Do not code yet. First identify the active phase/gate, blocking acceptance conditions, out-of-scope modules, and the smallest next valid coding step.
Treat the first run as G0 environment inheritance verification.
```

Or explicitly invoke the first skill:

```text
$wsovvis-phase-gate-check Determine the active phase/gate, blocking acceptance conditions, and the smallest next valid step.
```

After a code or experiment step, evaluate:

```text
$wsovvis-eval-acceptance Evaluate PASS / FAIL / INCONCLUSIVE for the current gate using docs/mainline/METRICS_ACCEPTANCE.md.
```

## Important
Do not assume the environment is inherited correctly until G0 records:
- canonical runner facts
- wrapper availability
- bootstrap preflight evidence
- remote HEAD consistency evidence
