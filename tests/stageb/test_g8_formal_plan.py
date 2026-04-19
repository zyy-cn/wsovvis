from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videocutler.ext_stageb_ovvis.eval.formal_plan import (
    build_formal_package_manifest,
    package_phase3_arms,
)


def test_paper_main_4arm_manifest_excludes_prealignonly(tmp_path: Path) -> None:
    manifest = build_formal_package_manifest(
        package_kind='paper_main_4arm',
        run_root=tmp_path / 'run_root',
        device='cuda:0',
        seed=0,
        logit_chunk_size=256,
        smoke=False,
    )
    arm_names = [arm['arm_name'] for arm in manifest['arms']]
    assert arm_names == ['Full', 'NoCoverage', 'TrajOnly', 'BaseOnly']
    specs = {arm['arm_name']: arm for arm in manifest['arms']}
    full_train = specs['Full']['commands'][1]
    assert full_train['stage'] == 'G7_softem'
    assert '--mode base_then_aug' in full_train['shell']
    assert '--lambda_frame 0.25' in full_train['shell']
    assert '--lambda_cov 1.0' in full_train['shell']
    nocov_train = specs['NoCoverage']['commands'][1]
    assert '--lambda_cov 0.0' in nocov_train['shell']
    traj_train = specs['TrajOnly']['commands'][1]
    assert '--lambda_frame 0.0' in traj_train['shell']
    base_train = specs['BaseOnly']['commands'][1]
    assert '--mode base_only' in base_train['shell']
    assert specs['BaseOnly']['selected_for_infer_expected'] == 'base_only'


def test_package_phase3_manifest_matches_package_authorized_arms(tmp_path: Path) -> None:
    manifest = build_formal_package_manifest(
        package_kind='package_phase3_full',
        run_root=tmp_path / 'run_root',
        device='cpu',
        seed=0,
        logit_chunk_size=64,
        smoke=True,
    )
    arm_names = [arm['arm_name'] for arm in manifest['arms']]
    assert arm_names == package_phase3_arms()
    prealign_arm = manifest['arms'][-1]
    assert prealign_arm['arm_name'] == 'PrealignOnly'
    assert len(prealign_arm['commands']) == 4
    assert prealign_arm['commands'][0]['stage'] == 'G7_prealign'
    assert prealign_arm['commands'][1]['stage'] == 'G8_infer'
    assert '--smoke' in prealign_arm['commands'][0]['shell']
    assert prealign_arm['selected_for_infer_expected'] == 'prealign_only'


def test_prepare_formal_package_tool_writes_bundle(tmp_path: Path) -> None:
    run_root = tmp_path / 'package_run'
    cmd = [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'g8_prepare_formal_package.py'),
        '--package_kind', 'full_single_arm_sanity',
        '--run_root', str(run_root),
        '--device', 'cuda:1',
        '--seed', '3',
        '--logit_chunk_size', '128',
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    manifest_path = run_root / 'g8_formal_package_manifest.json'
    summary_path = run_root / 'g8_formal_package_summary.md'
    shell_path = run_root / 'g8_formal_package_commands.sh'
    assert manifest_path.is_file()
    assert summary_path.is_file()
    assert shell_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['package_kind'] == 'full_single_arm_sanity'
    assert manifest['arms'][0]['arm_name'] == 'Full'
    assert 'run_stageb_eval_lvvis.py' in shell_path.read_text(encoding='utf-8')
