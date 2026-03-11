# WSOVVIS Environment and Validation

This file is the **authoritative environment contract** for the clean automation kit.
It fully replaces the old workflow docs for environment facts, canonical validation semantics,
and remote execution discipline.

## 1. Split execution model
### Local workstation / WSL
Role:
- control plane
- edit code
- inspect diffs
- run best-effort local checks
- prepare commits and prompts

Local results are **informative only** unless the active gate explicitly allows a doc-only or local-only acceptance path.

### `gpu4090d` remote runner
Role:
- canonical validation plane
- canonical pytest/smoke results
- canonical replay for environment-sensitive checks

Rule:
- canonical PASS requires remote validation on `gpu4090d` unless the active gate explicitly states otherwise.
- local PASS never overrides missing or stale canonical remote evidence.

## 2. Canonical remote target
Defaults unless the active gate explicitly overrides:
- remote host alias: `gpu4090d`
- canonical remote repo dir: `/home/zyy/code/wsovvis_runner`

Hard rules:
- do not create side runner directories for canonical validation
- do not use `git worktree` as a canonical runner substitute
- historical side-runner paths are archival only and are **not authoritative** for the current mainline

## 3. Canonical validation wrapper and semantics
Preferred canonical wrapper:
- `tools/remote_verify_wsovvis.sh`

This wrapper is the default canonical remote entrypoint.
Do **not** silently replace it with an ad hoc ssh flow unless the active gate explicitly permits a temporary exception and the result is recorded as non-canonical.

### Wrapper semantics that must be preserved
The canonical wrapper is expected to enforce or preserve all of the following:
- remote execution happens inside the canonical runner repo dir
- remote checkout/reset targets the intended branch/commit
- the validation command is recorded verbatim
- commit consistency is checkable after the run

If the wrapper appears broken:
1. verify the wrapper path exists
2. inspect wrapper arguments and repo-dir resolution
3. repair the wrapper path/usage first
4. do **not** bypass straight into arbitrary remote shell commands and still call the result canonical

## 4. Commit-consistency rule (critical)
A remote validation PASS counts only if:
- the local intended commit hash is recorded, and
- the remote checked-out `HEAD` equals that intended commit.

If remote validation ran on a stale or different commit, the result is **informational only**.
It may help debugging, but it does not satisfy a mainline acceptance gate.

Minimum evidence that must be recorded for every canonical validation:
- local branch
- intended local commit hash
- remote host alias
- remote repo dir used
- remote `HEAD`
- whether `remote HEAD == intended local commit`
- exact remote validation command
- PASS / FAIL / BLOCKED / INCONCLUSIVE outcome

## 5. Canonical remote execution recipe
Use this recipe unless a gate-specific runbook explicitly overrides part of it.

### 5.1 Canonical environment activation
The remote shell should be able to execute the following sequence or an equivalent project-approved form:

```bash
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}
```

### 5.2 Optional path additions when a task explicitly needs them
Some tasks may additionally require repo-root or third-party paths already provided by the live runner bootstrap.
Do **not** add extra paths by default unless the gate or task explicitly requires them.

### 5.3 Canonical wrapper pattern
Use the wrapper in this general shape:

```bash
bash tools/remote_verify_wsovvis.sh \
  --remote gpu4090d \
  --repo-dir /home/zyy/code/wsovvis_runner \
  --branch <task-branch> \
  --clone-url git@github.com:zyy-cn/wsovvis.git \
  --env-cmd 'source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}' \
  --cmd '<validation command>'
```

### 5.4 When `--keep-untracked` should be used
Use `--keep-untracked` when the canonical runner depends on untracked bootstrap links or live directories that must survive the run, especially when validating:
- real-artifact replay
- Stage B / Stage C sidecar-based tests
- any check that depends on live `runs`, `outputs`, `weights`, `data`, or `third_party/*` links

If unsure, prefer preserving untracked links and explicitly record that choice.

## 6. Runner cleanliness and untracked symlink pitfalls
The canonical runner may rely on untracked symlinks such as:
- `runs`
- `outputs`
- `weights`
- `data`
- `third_party/CutLER`
- `third_party/dinov2`
- `third_party/VNext`

Default destructive cleanup can remove them.
A canonical remote run is not considered trustworthy if it silently destroyed required live links first.

## 7. Canonical runner preflight
Before any canonical remote smoke or replay that depends on artifacts or third-party links, do this preflight.

### 7.1 Required checks
Verify that the canonical runner contains the expected live links/targets for:
- `runs`
- `outputs`
- `weights`
- `data`
- required `third_party/*` links

### 7.2 Preferred commands
Use the project preflight helper when available:

```bash
python tools/check_canonical_runner_bootstrap_links.py --check
```

If a fix is required and the helper supports it:

```bash
python tools/check_canonical_runner_bootstrap_links.py --fix
```

If the helper does not exist in the clean repo yet, record this as an environment bootstrap gap and classify the gate as `BLOCKED` or `INCONCLUSIVE` rather than faking a canonical PASS.

### 7.3 Which gates require preflight by default
At minimum, run preflight before:
- G0 environment verification
- any Stage B real-run or replay validation
- any Stage C loader/bridge replay using sidecar artifacts
- any validation command using `--keep-untracked`

## 8. Git / push route constraints
This repo may operate under network-restricted conditions.
If direct GitHub push is unreliable, use the project-approved SSH routing pattern, such as `github-via-gpu`, while preserving the same repository identity.

### Minimum operational rule
Before treating a remote validation as canonical, make sure the intended branch/commit is actually reachable by the canonical runner.
If push routing fails, classify the run as `BLOCKED` instead of pretending the validation executed on the intended code.

### Minimal connectivity check
A minimal SSH alias check may look like:

```bash
ssh -T git@github-via-gpu
```

If the alias or route is missing, record the issue as infra/network configuration debt.

## 9. Output-path discipline
Task artifacts belong under:
- `codex/<task_dir>/...`
- `docs/mainline/reports/...` for mainline gate and acceptance records

Writing loose `*_output.txt` or similar artifacts at repo root is not allowed.

## 10. Classification rules
Use these outcome classes consistently:
- `PASS`: canonical evidence exists and satisfies the gate
- `FAIL`: the code or contract is violated under canonical evaluation
- `BLOCKED`: environment, routing, missing wrapper, or missing bootstrap preflight prevents canonical evaluation
- `INCONCLUSIVE`: partial evidence exists but the gate cannot yet be authoritatively judged

Environment mismatch, remote mismatch, stale remote HEAD, or missing bootstrap links are **not** algorithmic failures.
