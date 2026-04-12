from __future__ import annotations

import sys
from pathlib import Path

from common import (
    append_ops_log,
    control_file,
    default_cache_dir,
    default_runtime_dir,
    load_profile_from_repo,
    parse_args,
    read_json,
    remote_host,
    remote_repo_dir,
    require_local_remote_profile,
    require_task_truth,
    run_cmd,
    write_runtime_state,
)


def _g2_export_full_commands(task: dict) -> list[str]:
    primary_artifacts = [
        str(item).strip()
        for item in task.get("primary_artifact_paths", [])
        if isinstance(item, str) and str(item).strip()
    ]
    dataset_names: list[str] = []
    for rel in primary_artifacts:
        parts = Path(rel).parts
        if len(parts) == 3 and parts[0] == "exports" and parts[2] == "trajectory_records.jsonl":
            dataset_names.append(parts[1])

    if not dataset_names or task.get("active_gate") != "G2_exports":
        return []

    export_root = "codex/outputs/g2_exports"
    exp_name = "g2_exports_full"
    generator_ckpt = "train/softem_aug/checkpoints/softem_aug_last.pth"
    device = "cuda:0"
    seed = "3407"
    contract_check = f"{export_root}/export_contract_check.json"
    commands: list[str] = []
    for dataset_name in dataset_names:
        commands.append(
            "python videocutler/run_stageb_export.py "
            f"--exp_name {exp_name} "
            f"--dataset_name {dataset_name} "
            f"--generator_ckpt {generator_ckpt} "
            f"--output_root {export_root} "
            f"--device {device} "
            f"--seed {seed} "
            f"--contract_check_json {contract_check}"
        )
    return commands


def main() -> int:
    parser = parse_args('Run authoritative remote formal execution after human approval')
    parser.add_argument('--repo-root', required=True)
    parser.add_argument('--profile')
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    kit_root = Path(__file__).resolve().parents[1]
    cache = default_cache_dir(repo_root)
    runtime = default_runtime_dir(repo_root)
    runtime.mkdir(parents=True, exist_ok=True)
    try:
        task_truth = require_task_truth(repo_root)
        task = read_json(control_file(repo_root, 'CURRENT_TASK.json'))
        smoke_state = read_json(runtime / 'smoke_loop_state.json') if (runtime / 'smoke_loop_state.json').exists() else {}
        if task.get('status') != 'READY':
            print('CURRENT_TASK is not READY', file=sys.stderr)
            return 2
        if smoke_state.get('status') != 'PASS':
            print('implementation smoke has not PASSed', file=sys.stderr)
            return 2
        profile, profile_path, profile_source = load_profile_from_repo(repo_root, args.profile)
        require_local_remote_profile(profile)
    except Exception as exc:
        write_runtime_state(repo_root, 'exec_state', {
            'task_id': '',
            'active_gate': '',
            'task_truth_stamp': '',
            'phase': 'formal',
            'status': 'FAIL',
            'returncode': 2,
            'stderr_tail': str(exc),
        })
        run_cmd([sys.executable, str(kit_root / 'tools' / 'refresh_takeover.py'), '--repo-root', str(repo_root)])
        print(str(exc), file=sys.stderr)
        return 2
    remote_root = remote_repo_dir(profile)
    host = remote_host(profile)
    task_id = task.get('task_id', '')
    active_gate = task.get('active_gate', '')
    append_ops_log(repo_root, 'formal_run_start', {
        'task_id': task_id,
        'task_truth_stamp': task_truth['task_truth_stamp'],
        'profile_source': profile_source,
        'profile_path': str(profile_path) if profile_path else '',
        'remote_host': host,
        'remote_repo_dir': remote_root,
    })
    write_runtime_state(repo_root, 'exec_state', {
        'task_id': task_id,
        'active_gate': active_gate,
        'task_truth_stamp': task_truth['task_truth_stamp'],
        'phase': 'formal',
        'status': 'RUNNING',
        'returncode': None,
    })
    if run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'preflight_git_clean.py'),
        '--repo-root', str(repo_root),
        '--remote-host', host,
        '--remote-repo-root', remote_root,
        '--output', str(cache / 'git_clean_result.json'),
        '--task-id', task_id,
        '--active-gate', active_gate,
        '--profile', str(profile_path) if profile_path else '',
    ]).returncode != 0:
        write_runtime_state(repo_root, 'exec_state', {'task_id': task_id, 'active_gate': active_gate, 'task_truth_stamp': task_truth['task_truth_stamp'], 'phase': 'formal', 'status': 'FAIL', 'returncode': 2})
        run_cmd([sys.executable, str(kit_root / 'tools' / 'refresh_takeover.py'), '--repo-root', str(repo_root)])
        return 2
    if run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'preflight_sync_code.py'),
        '--repo-root', str(repo_root),
        '--remote-host', host,
        '--remote-repo-root', remote_root,
        '--output', str(cache / 'sync_manifest.json'),
        '--task-id', task_id,
    ]).returncode != 0:
        write_runtime_state(repo_root, 'exec_state', {'task_id': task_id, 'active_gate': active_gate, 'task_truth_stamp': task_truth['task_truth_stamp'], 'phase': 'formal', 'status': 'FAIL', 'returncode': 2})
        run_cmd([sys.executable, str(kit_root / 'tools' / 'refresh_takeover.py'), '--repo-root', str(repo_root)])
        return 2
    if run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'preflight_verify_remote_tree.py'),
        '--repo-root', str(repo_root),
        '--remote-host', host,
        '--remote-repo-root', remote_root,
        '--sync-manifest', str(cache / 'sync_manifest.json'),
        '--output', str(cache / 'preflight_result.json'),
        '--task-id', task_id,
        '--active-gate', active_gate,
    ]).returncode != 0:
        write_runtime_state(repo_root, 'exec_state', {'task_id': task_id, 'active_gate': active_gate, 'task_truth_stamp': task_truth['task_truth_stamp'], 'phase': 'formal', 'status': 'FAIL', 'returncode': 2})
        run_cmd([sys.executable, str(kit_root / 'tools' / 'refresh_takeover.py'), '--repo-root', str(repo_root)])
        return 2
    plan = read_json(control_file(repo_root, 'VALIDATION_PLAN.json'))
    formal_remote_exists = any(c.get('phase') == 'formal' and c.get('runner') == 'remote' for c in plan.get('checks', []))
    phase_to_run = 'formal' if formal_remote_exists else 'implementation'
    export_full_cmds = _g2_export_full_commands(task)
    validation_cmd = (
        'python kit-src/tools/check_kit_runtime.py --scope remote --output codex/state/private-cache/kit_runtime_check_remote.json '
        f'&& python kit-src/tools/run_validation.py --repo-root . --plan codex/control/VALIDATION_PLAN.json '
        f'--output codex/state/private-cache/validation_result.json --phase {phase_to_run} --runner-scope remote '
        '&& python kit-src/tools/validate_private_layer.py --repo-root . --output codex/state/private-cache/private_layer_validation.json '
        '&& python kit-src/tools/build_context_resume.py --repo-root . '
        '&& python kit-src/tools/refresh_takeover.py --repo-root .'
    )
    cmd = ' && '.join(export_full_cmds + [validation_cmd])
    cp = run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'run_remote_task.py'),
        '--repo-root', str(repo_root),
        '--profile', str(profile_path) if profile_path else '',
        '--remote-repo-root', remote_root,
        '--task-id', task_id,
        '--active-gate', active_gate,
        '--phase', 'formal',
        '--command', cmd,
    ])
    run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'pull_remote_canonical_results.py'),
        '--repo-root', str(repo_root),
        '--remote-host', host,
        '--remote-repo-root', remote_root,
        '--task-id', task_id,
        '--active-gate', active_gate,
        '--phase', 'formal',
    ])
    run_cmd([
        sys.executable,
        str(kit_root / 'tools' / 'run_validation.py'),
        '--repo-root', str(repo_root),
        '--plan', str(control_file(repo_root, 'VALIDATION_PLAN.json')),
        '--output', str(cache / 'validation_result_local_formal.json'),
        '--phase', 'formal',
        '--runner-scope', 'local',
    ])
    cp_context = run_cmd([sys.executable, str(kit_root / 'tools' / 'build_context_resume.py'), '--repo-root', str(repo_root)])
    cp_takeover = run_cmd([sys.executable, str(kit_root / 'tools' / 'refresh_takeover.py'), '--repo-root', str(repo_root)])
    final_status = 'PASS' if cp.returncode == 0 and cp_context.returncode == 0 and cp_takeover.returncode == 0 else 'FAIL'
    write_runtime_state(repo_root, 'exec_state', {
        'task_id': task_id,
        'active_gate': active_gate,
        'task_truth_stamp': task_truth['task_truth_stamp'],
        'phase': 'formal',
        'status': final_status,
        'returncode': 0 if final_status == 'PASS' else 2,
    })
    append_ops_log(repo_root, 'formal_run_finish', {'status': final_status})
    return 0 if final_status == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
