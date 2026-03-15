# WS-OVVIS Environment and Validation

This file is the authoritative environment contract for the project-private adaptation.

## 1. Split execution model
### Local workstation / WSL
Role:
- control plane
- edit code
- inspect diffs
- run best-effort local checks
- prepare commits and prompts

### Canonical remote runner
Role:
- canonical validation plane
- canonical pytest/smoke results
- canonical replay for environment-sensitive checks

## 2. Canonical remote target
Defaults for this profile:
- remote host alias: `gpu4090d`
- canonical remote repo dir: `/home/zyy/code/wsovvis_runner`

Hard rules:
- do not create side runner directories for canonical validation
- do not use `git worktree` as a canonical runner substitute unless the project-private docs explicitly authorize it

## 3. Canonical validation wrapper
Preferred canonical wrapper:
- `tools/remote_verify_wsovvis.sh`

This wrapper is the default canonical remote entrypoint.
Do not silently replace it with ad hoc ssh execution and still call the result canonical.

## 4. Commit-consistency rule
A remote validation PASS counts only if:
- the local intended commit hash is recorded, and
- the remote checked-out `HEAD` equals that intended commit.

If remote validation ran on a stale or different commit, the result is informational only.

## 5. Canonical environment activation recipe
The canonical remote shell should be able to execute the following sequence or an equivalent project-approved form:

```bash
source $HOME/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}
```

## 6. Push route constraints
If direct Git push is unreliable, use the project-approved SSH route, such as:
- `github-via-gpu`

## 7. Bootstrap preflight
Check the live links or directories listed in the project-private bootstrap manifest before canonical replay when required.
