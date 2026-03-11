# WS-OVVIS Environment and Validation

This file defines the canonical validation model for the repository.

## 1. Canonical validation
- Local checks are informative.
- Canonical PASS requires the project-approved remote validation model when the gate contract says canonical evidence is required.
- A remote PASS counts only if remote `HEAD == intended local commit`.

## 2. Canonical remote model
Use the project-approved canonical path:
- remote host alias: `gpu4090d`
- canonical runner repo dir: `/home/zyy/code/wsovvis_runner`
- wrapper: `bash tools/remote_verify_wsovvis.sh`
- origin clone URL: `git@github.com:zyy-cn/wsovvis.git`
- push-route alias: `github-via-gpu`
- push-route URL (from local git config): `git@github-via-gpu:zyy-cn/wsovvis.git`

## 3. Bootstrap preflight
Before canonical remote validation when live links or artifacts matter:
- confirm wrapper availability
- confirm canonical runner repo dir
- run `python tools/check_canonical_runner_bootstrap_links.py --check` when required
- confirm intended local commit is reachable remotely

The current bootstrap-link checker expects these managed links under the runner:
- `third_party/CutLER -> ../../wsovvis_live/third_party/CutLER`
- `third_party/dinov2 -> ../../wsovvis_live/third_party/dinov2`
- `runs -> ../wsovvis_live/runs`
- `weights -> ../wsovvis_live/weights`
- `data -> ../wsovvis_live/data`

## 4. Environment activation
The canonical remote environment should use the project-approved recipe:
```bash
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}
```

## 5. Canonical invocation pattern
A canonical run should record the exact wrapper invocation, for example:
```bash
bash tools/remote_verify_wsovvis.sh \
  --remote gpu4090d \
  --repo-dir /home/zyy/code/wsovvis_runner \
  --branch <current-branch> \
  --env-cmd "source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis && export PYTHONPATH=/home/zyy/code/wsovvis_runner/third_party/VNext:${PYTHONPATH:-}" \
  --cmd "<gate-specific verification command>" \
  --clone-url git@github.com:zyy-cn/wsovvis.git \
  --allow-suspicious-repo-dir
```

## 6. Result classification
Classify results as:
- `canonical`
- `informational`
- `blocked`
- `inconclusive`

Do not classify wrapper, environment, push-route, bootstrap-link, or remote-HEAD mismatches as algorithmic failure.

## 7. Gate-to-validation expectations
- `G0` and some `G1` work may rely on local checks plus environment verification.
- `G2`-`G7` should use canonical validation when a PASS claim depends on repository-integrated evidence.
- terminal PASS at `G7` requires canonical evidence.

## 8. Evidence-pack interaction
When canonical evidence is required, the evidence pack must record:
- intended local commit
- observed remote `HEAD`
- exact remote command or wrapper invocation
- canonical output artifact or report locations
- whether the quantitative and visual evidence was produced from the canonical run or from an informative local run
