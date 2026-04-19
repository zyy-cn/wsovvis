from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class ArmSpec:
    arm_name: str
    mode: str
    lambda_frame: float | None
    lambda_cov: float | None
    selected_for_infer: str
    include_prealign: bool


@dataclass(frozen=True)
class CommandPlan:
    step_id: str
    stage: str
    argv: List[str]
    output_root: str
    arm_name: str


ARM_NAME_TO_SPEC: Dict[str, ArmSpec] = {
    'Full': ArmSpec('Full', mode='base_then_aug', lambda_frame=0.25, lambda_cov=1.0, selected_for_infer='augmented', include_prealign=True),
    'TrajOnly': ArmSpec('TrajOnly', mode='base_then_aug', lambda_frame=0.0, lambda_cov=1.0, selected_for_infer='augmented', include_prealign=True),
    'NoCoverage': ArmSpec('NoCoverage', mode='base_then_aug', lambda_frame=0.25, lambda_cov=0.0, selected_for_infer='augmented', include_prealign=True),
    'BaseOnly': ArmSpec('BaseOnly', mode='base_only', lambda_frame=0.25, lambda_cov=1.0, selected_for_infer='base_only', include_prealign=True),
    'PrealignOnly': ArmSpec('PrealignOnly', mode='prealign_only', lambda_frame=0.25, lambda_cov=None, selected_for_infer='prealign_only', include_prealign=True),
}


def load_training_defaults() -> Dict[str, Any]:
    return _read_json(repo_root() / 'package' / 'reference' / 'g7_training_execution_defaults.json')


def load_algorithm_contract() -> Dict[str, Any]:
    return _read_json(repo_root() / 'package' / 'reference' / 'g7_algorithm_contract.json')


def package_phase3_arms() -> List[str]:
    payload = load_training_defaults()
    arms = payload.get('mechanism_validation_chain_defaults', {}).get('phase_3_arms', [])
    return [str(item) for item in arms]


def paper_main_four_arms() -> List[str]:
    return ['Full', 'NoCoverage', 'TrajOnly', 'BaseOnly']


def arm_spec(arm_name: str) -> ArmSpec:
    try:
        return ARM_NAME_TO_SPEC[str(arm_name)]
    except KeyError as exc:
        raise KeyError(f'unsupported arm_name: {arm_name}') from exc


def _quoted(argv: Sequence[str]) -> str:
    def quote(token: str) -> str:
        safe = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:-')
        if token and all(ch in safe for ch in token):
            return token
        return "'" + token.replace("'", "'\\''") + "'"

    return ' '.join(quote(str(token)) for token in argv)


def build_eval_only_commands(*, exp_name: str, output_root: Path, device: str, seed: int, logit_chunk_size: int, dataset_name: str = 'lvvis_val', smoke: bool = False) -> List[CommandPlan]:
    common = ['--exp_name', exp_name, '--output_root', str(output_root), '--seed', str(seed)]
    infer_argv = [
        'python', 'videocutler/run_stageb_infer_ov.py',
        '--exp_name', exp_name,
        '--dataset_name', dataset_name,
        '--output_root', str(output_root),
        '--device', device,
        '--seed', str(seed),
        '--logit_chunk_size', str(logit_chunk_size),
    ]
    if smoke:
        infer_argv.append('--smoke')
    ext_argv = ['python', 'videocutler/run_stageb_eval_lvvis.py', *common]
    int_argv = ['python', 'videocutler/run_stageb_eval_internal.py', '--exp_name', exp_name, '--dataset_name', dataset_name, '--output_root', str(output_root), '--seed', str(seed)]
    if smoke:
        ext_argv.append('--smoke')
        int_argv.append('--smoke')
    return [
        CommandPlan(step_id='infer', stage='G8_infer', argv=infer_argv, output_root=str(output_root), arm_name='eval_only'),
        CommandPlan(step_id='eval_lvvis', stage='G8_external_lvvis', argv=ext_argv, output_root=str(output_root), arm_name='eval_only'),
        CommandPlan(step_id='eval_internal', stage='G8_internal', argv=int_argv, output_root=str(output_root), arm_name='eval_only'),
    ]


def build_train_eval_commands(*, arm_name: str, exp_name: str, output_root: Path, device: str, seed: int, logit_chunk_size: int, smoke: bool = False, dataset_name: str = 'lvvis_val') -> List[CommandPlan]:
    spec = arm_spec(arm_name)
    commands: List[CommandPlan] = []
    prealign_argv = ['python', 'videocutler/run_stageb_train_prealign.py', '--exp_name', exp_name, '--output_root', str(output_root), '--device', device, '--seed', str(seed)]
    if smoke:
        prealign_argv.append('--smoke')
    commands.append(CommandPlan(step_id='train_prealign', stage='G7_prealign', argv=prealign_argv, output_root=str(output_root), arm_name=arm_name))

    if spec.mode != 'prealign_only':
        softem_argv = ['python', 'videocutler/run_stageb_train_softem.py', '--exp_name', exp_name, '--output_root', str(output_root), '--device', device, '--seed', str(seed), '--mode', spec.mode]
        if spec.lambda_frame is not None:
            softem_argv.extend(['--lambda_frame', str(spec.lambda_frame)])
        if spec.lambda_cov is not None:
            softem_argv.extend(['--lambda_cov', str(spec.lambda_cov)])
        if smoke:
            softem_argv.append('--smoke')
        commands.append(CommandPlan(step_id='train_softem', stage='G7_softem', argv=softem_argv, output_root=str(output_root), arm_name=arm_name))

    commands.extend(
        build_eval_only_commands(
            exp_name=exp_name,
            output_root=output_root,
            device=device,
            seed=seed,
            logit_chunk_size=logit_chunk_size,
            dataset_name=dataset_name,
            smoke=smoke,
        )
    )
    return commands


def build_formal_package_manifest(*, package_kind: str, run_root: Path, device: str, seed: int, logit_chunk_size: int, smoke: bool = False) -> Dict[str, Any]:
    run_root = Path(run_root)
    if package_kind == 'full_single_arm_sanity':
        arm_names = ['Full']
        eval_only = False
    elif package_kind == 'paper_main_4arm':
        arm_names = paper_main_four_arms()
        eval_only = False
    elif package_kind == 'package_phase3_full':
        arm_names = package_phase3_arms()
        eval_only = False
    elif package_kind == 'eval_only_existing_workspace':
        arm_names = ['Full']
        eval_only = True
    else:
        raise ValueError(f'unsupported package_kind: {package_kind}')

    package_defaults = load_training_defaults()
    algo_contract = load_algorithm_contract()
    arms_payload: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    for arm_name in arm_names:
        arm_out = run_root if eval_only else (run_root / arm_name)
        exp_name = f'g8_{package_kind.lower()}_{arm_name.lower()}'
        commands = build_eval_only_commands(
            exp_name=exp_name,
            output_root=arm_out,
            device=device,
            seed=seed,
            logit_chunk_size=logit_chunk_size,
            smoke=smoke,
        ) if eval_only else build_train_eval_commands(
            arm_name=arm_name,
            exp_name=exp_name,
            output_root=arm_out,
            device=device,
            seed=seed,
            logit_chunk_size=logit_chunk_size,
            smoke=smoke,
        )
        arm_info = {
            'arm_name': arm_name,
            'exp_name': exp_name,
            'output_root': str(arm_out),
            'selected_for_infer_expected': arm_spec(arm_name).selected_for_infer,
            'command_count': len(commands),
            'commands': [
                {
                    'step_id': item.step_id,
                    'stage': item.stage,
                    'argv': list(item.argv),
                    'shell': _quoted(item.argv),
                }
                for item in commands
            ],
        }
        arms_payload.append(arm_info)
        all_steps.extend([{**item, 'arm_name': arm_name} for item in arm_info['commands']])

    return {
        'status': 'READY',
        'package_kind': package_kind,
        'smoke': bool(smoke),
        'seed': int(seed),
        'device': str(device),
        'logit_chunk_size': int(logit_chunk_size),
        'run_root': str(run_root),
        'defaults_ref': 'package/reference/g7_training_execution_defaults.json',
        'algorithm_contract_ref': 'package/reference/g7_algorithm_contract.json',
        'train_infer_handoff_ref': 'package/reference/train_infer_handoff_rule.json',
        'execution_defaults_subset': {
            'formal_selected_for_infer_value': package_defaults.get('formal_selected_for_infer_value'),
            'phase_3_arms': package_defaults.get('mechanism_validation_chain_defaults', {}).get('phase_3_arms', []),
            'lambda_frame': package_defaults.get('mechanism_validation_chain_defaults', {}).get('lambda_frame', {}),
            'lambda_cov': package_defaults.get('mechanism_validation_chain_defaults', {}).get('lambda_cov', {}),
        },
        'algorithm_phase3_authorized_arms': algo_contract.get('formal_mechanism_validation_chain', {}).get('phases', [])[2].get('arms', []) if len(algo_contract.get('formal_mechanism_validation_chain', {}).get('phases', [])) >= 3 else [],
        'arms': arms_payload,
        'step_count_total': int(sum(arm['command_count'] for arm in arms_payload)),
    }


def manifest_to_shell_lines(manifest: Mapping[str, Any]) -> List[str]:
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', '']
    lines.append(f"# package_kind={manifest.get('package_kind')}")
    lines.append(f"# run_root={manifest.get('run_root')}")
    lines.append('')
    for arm in manifest.get('arms', []):
        lines.append(f"# arm={arm['arm_name']} output_root={arm['output_root']}")
        for command in arm.get('commands', []):
            lines.append(str(command['shell']))
        lines.append('')
    return lines


def write_manifest_bundle(*, manifest: Mapping[str, Any], manifest_path: Path, shell_path: Path, summary_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shell_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shell_path.write_text('\n'.join(manifest_to_shell_lines(manifest)) + '\n', encoding='utf-8')
    arm_lines = []
    for arm in manifest.get('arms', []):
        arm_lines.append(f"- `{arm['arm_name']}` → `{arm['output_root']}` ({arm['command_count']} commands, selected_for_infer=`{arm['selected_for_infer_expected']}`)")
    summary = '\n'.join([
        '# G8 Formal Package Plan',
        '',
        f"- package_kind: `{manifest.get('package_kind')}`",
        f"- run_root: `{manifest.get('run_root')}`",
        f"- seed: `{manifest.get('seed')}`",
        f"- device: `{manifest.get('device')}`",
        f"- smoke: `{manifest.get('smoke')}`",
        f"- step_count_total: `{manifest.get('step_count_total')}`",
        '',
        '## Arms',
        *arm_lines,
        '',
        '## Notes',
        '- This bundle is add-only orchestration. It does not mutate existing training or evaluator defaults.',
        '- The generated commands call existing canonical G7/G8 CLIs only.',
        '- `paper_main_4arm` intentionally excludes `PrealignOnly`; use `package_phase3_full` when strict package-authorized phase-3 coverage is required.',
    ]) + '\n'
    summary_path.write_text(summary, encoding='utf-8')
