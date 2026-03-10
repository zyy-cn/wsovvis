# WSOVVIS V9 Environment and Validation

This file preserves the same validation semantics as the current automation kit.
Only the mainline target changes; the canonical validation model does not.

## 1. Local vs canonical validation
- Local checks are informative.
- Canonical PASS requires the remote validation model when the gate contract says canonical evidence is required.
- A remote PASS counts only if remote `HEAD == intended local commit`.

## 2. Canonical remote model
Use the project-approved canonical path:
- remote host alias: `gpu4090d`
- canonical runner repo dir: `/home/zyy/code/wsovvis_runner`
- wrapper: `bash tools/remote_verify_wsovvis.sh`

## 3. Bootstrap preflight
Before canonical remote validation when live links or artifacts matter:
- confirm wrapper availability
- confirm canonical runner repo dir
- run `python tools/check_canonical_runner_bootstrap_links.py --check` when required
- confirm intended local commit is reachable remotely

## 4. Environment activation
The canonical remote environment should use the project-approved recipe, including:
- `source ~/software/miniconda3/etc/profile.d/conda.sh`
- `conda activate wsovvis`
- minimum `PYTHONPATH` for `third_party/VNext`

## 5. Result classification
Classify results as:
- `canonical`
- `informational`
- `blocked`
- `inconclusive`

Do not classify wrapper, environment, push-route, or remote-HEAD mismatches as algorithmic failure.

## 6. Gate-to-validation expectations
- G0/G1 may often rely on local checks plus environment verification.
- G2-G7 should use canonical validation when a PASS claim depends on repository-integrated evidence.
- terminal PASS at G7 requires canonical evidence.

## 7. Evidence-pack interaction
When canonical evidence is required, the evidence pack must record:
- intended local commit
- observed remote `HEAD`
- exact remote command or wrapper invocation
- canonical output artifact/report locations
- whether the quantitative and visual evidence was produced from the canonical run or from an informative local run
