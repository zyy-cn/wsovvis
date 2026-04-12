from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import (
    changed_file_snapshot,
    control_file,
    parse_args,
    map_generated_relpaths,
    path_boundary_baseline_path,
    gate_output_path,
    read_json,
    resolve_package_ref,
    require_schema,
    require_task_truth,
    run_cmd,
    validate_json_schema,
    write_json,
)


def _split_schema_target(target: str) -> Tuple[str, str]:
    if "::" not in target:
        return "", ""
    a, b = target.split("::", 1)
    return a.strip(), b.strip()


def _matches_any(allow_paths: List[str], rel: str) -> bool:
    for p in allow_paths:
        p = p.rstrip("/")
        if rel == p or rel.startswith(p + "/"):
            return True
    return False


IGNORE_PREFIXES = ["codex/control/", "codex/state/", ".agents/", ".codex/", "schemas/", "kit-src/", "package/"]
IGNORE_EXACT = {"AGENTS.md", "CURRENT_TASK.json", "PATH_POLICY.json", "VALIDATION_PLAN.json", "TAKEOVER_LATEST.md", ".gitignore"}


def _tracked_delta_paths(repo_root: Path, baseline_snapshot: Dict[str, Dict[str, Any]]) -> List[str]:
    current_snapshot = changed_file_snapshot(repo_root)
    delta_paths: List[str] = []
    for rel, state in current_snapshot.items():
        if rel in IGNORE_EXACT or any(rel.startswith(prefix) for prefix in IGNORE_PREFIXES):
            continue
        if baseline_snapshot.get(rel) != state:
            delta_paths.append(rel)
    return sorted(delta_paths)


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(read_json_line(line))
        return records
    return read_json(path)


def read_json_line(line: str) -> Dict[str, Any]:
    import json
    return json.loads(line)


def _artifact_contracts(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    contracts_dir = resolve_package_ref(repo_root, "contracts/artifacts")
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(contracts_dir.glob("*.artifact_contract.json")):
        payload = read_json(path)
        artifact_id = str(payload.get("artifact_id", "")).strip()
        if artifact_id:
            out[artifact_id] = payload
    return out


def _gate_primary_artifacts(repo_root: Path) -> Dict[str, str]:
    ref = read_json(resolve_package_ref(repo_root, "reference/gate_primary_artifacts.json"))
    out: Dict[str, str] = {}
    for gate in ref.get("gates", []):
        gate_id = str(gate.get("gate_id", "")).strip()
        for rel in gate.get("primary_artifacts", []):
            rel = str(rel).strip()
            if gate_id and rel:
                out[rel] = gate_id
    return out


def _load_package_manifest(repo_root: Path) -> Dict[str, Any]:
    cache_manifest = repo_root / "codex" / "state" / "private-cache" / "package_manifest.json"
    if cache_manifest.exists():
        return read_json(cache_manifest)
    return read_json(resolve_package_ref(repo_root, "package_manifest.json"))


def _load_gate_contract(repo_root: Path, active_gate: str) -> Dict[str, Any]:
    manifest = _load_package_manifest(repo_root)
    gate_ref = str(manifest.get("active_gate_contract_ref", "")).strip()
    if gate_ref and active_gate == manifest.get("active_gate"):
        return read_json(resolve_package_ref(repo_root, gate_ref))
    return read_json(resolve_package_ref(repo_root, f"contracts/gates/{active_gate}.gate_contract.json"))


def _load_check_contract(repo_root: Path, check_id: str) -> Dict[str, Any]:
    return read_json(resolve_package_ref(repo_root, f"contracts/checks/{check_id}.check_contract.json"))


def _artifact_gate_id(repo_root: Path, canonical_rel: str, default_gate: str) -> str:
    return _gate_primary_artifacts(repo_root).get(canonical_rel, default_gate)


def _artifact_path_for_contract(repo_root: Path, gate_id: str, artifact_contract: Dict[str, Any]) -> Path:
    canonical_rel = str(artifact_contract.get("canonical_relative_path", "")).strip()
    artifact_base = str(artifact_contract.get("artifact_path_base", "")).strip()
    if artifact_base == "output_root_relative":
        return repo_root / canonical_rel
    return gate_output_path(repo_root, gate_id, canonical_rel)


def _resolve_artifact_specs(repo_root: Path, active_gate: str, check_id: str) -> List[Dict[str, Any]]:
    contracts = _artifact_contracts(repo_root)
    if check_id == "artifact_schema_valid":
        gate_contract = _load_gate_contract(repo_root, active_gate)
        refs: List[str] = []
        for canonical_rel in gate_contract.get("primary_artifacts", []):
            canonical_rel = str(canonical_rel).strip()
            if not canonical_rel:
                continue
            for artifact in contracts.values():
                if artifact.get("canonical_relative_path") == canonical_rel:
                    refs.append(f"contracts/artifacts/{artifact['artifact_id']}.artifact_contract.json")
                    break
    else:
        refs = [str(item).strip() for item in _load_check_contract(repo_root, check_id).get("input_artifact_refs", []) if str(item).strip()]

    specs: List[Dict[str, Any]] = []
    for ref in refs:
        artifact_contract = read_json(resolve_package_ref(repo_root, ref))
        canonical_rel = str(artifact_contract.get("canonical_relative_path", "")).strip()
        gate_id = _artifact_gate_id(repo_root, canonical_rel, active_gate)
        artifact_path = _artifact_path_for_contract(repo_root, gate_id, artifact_contract)
        specs.append({
            "contract": artifact_contract,
            "path": artifact_path,
            "gate_id": gate_id,
            "canonical_relative_path": canonical_rel,
        })
    return specs


def _records_for_checks(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _require_fields(records: List[Dict[str, Any]], fields: List[str], errors: List[str]) -> None:
    if not records:
        errors.append("no records available")
        return
    sample = records[0]
    for field in fields:
        if field not in sample:
            errors.append(f"missing field: {field}")


def _validate_weak_labels_records(records: List[Dict[str, Any]], check_id: str, phase: str, errors: List[str]) -> None:
    if not records:
        errors.append("no records available")
        return
    required_base = ["clip_id", "video_id", "observed_raw_ids", "observation_protocol_id", "completeness_status"]
    _require_fields(records, required_base, errors)
    if errors:
        return

    sample = records[0]
    if sample.get("completeness_status") != "unknown":
        errors.append("completeness_status must be unknown")
    protocol_id = str(sample.get("observation_protocol_id", "")).strip()
    if protocol_id not in {"keep80_seed42", "keep60_seed42", "keep40_seed42"}:
        errors.append("unsupported observation_protocol_id")
    observed_raw_ids = sample.get("observed_raw_ids")
    if not isinstance(observed_raw_ids, list) or any(not isinstance(item, int) for item in observed_raw_ids):
        errors.append("observed_raw_ids must be an integer list")

    if check_id in {"clip_level_weak_label_reader_readable", "weak_labels_run_scope_declared", "weak_labels_full_consumer_ready", "artifact_schema_valid"}:
        _require_fields(records, ["run_scope", "input_source_type", "data_scope", "consumer_target"], errors)
    if check_id in {"weak_labels_run_scope_declared", "weak_labels_full_consumer_ready", "artifact_schema_valid"}:
        _require_fields(records, ["record_count", "coverage_ratio", "consumer_ready"], errors)
    if check_id == "weak_labels_full_consumer_ready" and phase == "formal":
        if sample.get("run_scope") != "full":
            errors.append("run_scope must be full")
        if sample.get("input_source_type") != "official_lvvis_train_annotations":
            errors.append("input_source_type must be official_lvvis_train_annotations")
        if sample.get("consumer_ready") is not True:
            errors.append("consumer_ready must be true")
        if sample.get("data_scope") != "train":
            errors.append("data_scope must be train")


def _run_package_check(repo_root: Path, active_gate: str, check_id: str, phase: str, task: Dict[str, Any] | None = None) -> List[str]:
    errors: List[str] = []
    artifact_specs = _resolve_artifact_specs(repo_root, active_gate, check_id)
    if not artifact_specs:
        return [f"no package artifact refs resolved for check {check_id}"]
    package_contract = read_json(resolve_package_ref(repo_root, "contracts/package_contract.json"))
    minimum_takeover_fields = list(package_contract.get("required_summary_fields", []))

    for spec in artifact_specs:
        path = spec["path"]
        if not path.exists():
            errors.append(f"missing artifact: {path.relative_to(repo_root)}")
            continue
        data = _read_json_or_jsonl(path)
        records = _records_for_checks(data)

        schema_ref = str(spec["contract"].get("schema_ref", "")).strip()
        if str(spec["contract"].get("artifact_id", "")).strip() == "weak_labels_train":
            semantic_errors: List[str] = []
            _validate_weak_labels_records(records, check_id, phase, semantic_errors)
            if semantic_errors:
                errors.append(f"{path.relative_to(repo_root)} schema invalid: {'; '.join(semantic_errors[:5])}")
                continue
        elif schema_ref:
            schema_path = resolve_package_ref(repo_root, schema_ref)
            if isinstance(data, list):
                schema_errors: List[str] = []
                payloads = [item for item in data if isinstance(item, dict)]
                if len(payloads) != len(data):
                    schema_errors.append("JSONL payload contains non-object records")
                for idx, payload in enumerate(payloads):
                    ok, errs = validate_json_schema(payload, schema_path)
                    if not ok:
                        schema_errors.extend([f"record {idx}: {err}" for err in errs[:3]])
                        if len(schema_errors) >= 5:
                            break
                if schema_errors:
                    errors.append(f"{path.relative_to(repo_root)} schema invalid: {'; '.join(schema_errors[:5])}")
                    continue
            else:
                ok, schema_errors = validate_json_schema(data, schema_path)
                if not ok:
                    errors.append(f"{path.relative_to(repo_root)} schema invalid: {'; '.join(schema_errors[:3])}")
                    continue

        if check_id in {
            "clip_level_weak_label_reader_readable",
            "trajectory_sample_view_readable",
            "frame_feature_reader_parses_dsl",
            "carrier_reader_parses_dsl",
            "text_bank_reader_parses_dsl",
            "train_state_selected_for_infer_readable",
            "stage_local_snapshot_reader_readable",
            "lvvis_evaluator_reads_pred_main",
            "ytvis2019_evaluator_reads_pred_main",
            "internal_eval_reads_pred_diag",
            "report_collector_reads_canonical_artifacts",
        }:
            continue
        if check_id == "weak_labels_run_scope_declared":
            _require_fields(records, ["run_scope", "input_source_type", "data_scope", "consumer_target"], errors)
        elif check_id == "weak_labels_full_consumer_ready":
            _require_fields(records, ["run_scope", "input_source_type", "record_count", "coverage_ratio", "consumer_ready"], errors)
            if records and phase == "formal":
                sample = records[0]
                if sample.get("run_scope") != "full":
                    errors.append("run_scope must be full")
                if sample.get("input_source_type") != "official_lvvis_train_annotations":
                    errors.append("input_source_type must be official_lvvis_train_annotations")
                if sample.get("consumer_ready") is not True:
                    errors.append("consumer_ready must be true")
        elif check_id == "export_full_scope_ready":
            contract_path = ""
            current_task = task or read_json(control_file(repo_root, "CURRENT_TASK.json"))
            for candidate in current_task.get("required_outputs", []):
                if isinstance(candidate, str) and candidate.endswith("_contract_check.json"):
                    contract_path = candidate
                    break
            if not contract_path:
                errors.append("missing contract check artifact reference")
            else:
                artifact_path = repo_root / contract_path
                if not artifact_path.exists():
                    errors.append(f"missing contract check artifact: {contract_path}")
                else:
                    contract = read_json(artifact_path)
                    _require_fields([contract], ["run_scope", "input_source_type", "data_scope", "coverage_ratio", "consumer_ready"], errors)
                    if contract.get("run_scope") != "full":
                        errors.append("run_scope must be full")
                    if contract.get("consumer_ready") is not True:
                        errors.append("consumer_ready must be true")
                    if float(contract.get("coverage_ratio", 0.0) or 0.0) < 1.0:
                        errors.append("coverage_ratio must be >= 1.0")
        elif check_id == "frame_cache_coverage_ready":
            _require_fields(records, ["run_scope", "coverage_ratio", "consumer_ready"], errors)
            if records and "missing_frame_ratio" not in records[0]:
                errors.append("missing field: missing_frame_ratio")
        elif check_id == "carrier_bank_coverage_ready":
            _require_fields(records, ["run_scope", "coverage_ratio", "invalid_reason_stats", "consumer_ready"], errors)
        elif check_id == "text_bank_consumer_ready":
            _require_fields(records, ["run_scope", "class_coverage", "input_source_type", "consumer_ready"], errors)
        elif check_id == "selected_checkpoint_exists":
            for record in records:
                selected = str(record.get("selected_for_infer", "")).strip()
                if not selected:
                    errors.append("selected_for_infer missing")
                    continue
                selected_path = gate_output_path(repo_root, active_gate, selected) if not selected.startswith("codex/") else repo_root / selected
                if not selected_path.exists():
                    errors.append(f"selected checkpoint missing: {selected}")
        elif check_id == "eval_upstream_full_ready":
            _require_fields(records, ["upstream_run_scope", "upstream_consumer_ready", "evaluation_run_scope"], errors)
            if records and phase == "formal":
                sample = records[0]
                if sample.get("upstream_run_scope") != "full":
                    errors.append("upstream_run_scope must be full")
                if sample.get("upstream_consumer_ready") is not True:
                    errors.append("upstream_consumer_ready must be true")
                if sample.get("evaluation_run_scope") != "full":
                    errors.append("evaluation_run_scope must be full")
        elif check_id == "summary_single_file_proof_complete":
            _require_fields(records, minimum_takeover_fields, errors)
        elif check_id == "artifact_schema_valid":
            continue
    return errors


def main() -> int:
    parser = parse_args("Run validation from VALIDATION_PLAN.json")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="all", choices=["implementation", "formal", "all"])
    parser.add_argument("--runner-scope", default="all", choices=["local", "remote", "all"])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    plan_path = Path(args.plan)
    out_path = Path(args.output)
    schema_root = repo_root / "schemas" if (repo_root / "schemas").exists() else Path(__file__).resolve().parents[1] / "schemas"

    checks_run: List[str] = []
    checks_failed: List[str] = []
    missing_artifacts: List[str] = []
    notes: List[str] = []
    failure_code = ""
    path_boundary_status = "NOT_RUN"
    path_boundary_violations: List[str] = []
    contract_check_paths: List[str] = []
    baseline_ref = str(path_boundary_baseline_path(repo_root))
    task_truth_stamp = ""

    status = "FAIL"
    try:
        task_truth = require_task_truth(repo_root)
        task_truth_stamp = task_truth["task_truth_stamp"]
        plan = read_json(plan_path)
        require_schema(plan, schema_root / "VALIDATION_PLAN.schema.json", "VALIDATION_PLAN")

        def selected(check: Dict[str, Any]) -> bool:
            phase_ok = args.phase == 'all' or check.get('phase') == args.phase
            runner_ok = args.runner_scope == 'all' or check.get('runner') == args.runner_scope
            return phase_ok and runner_ok

        task = read_json(control_file(repo_root, "CURRENT_TASK.json"))
        contract_check_paths = [
            path for path in task.get("required_outputs", []) if path.endswith("_contract_check.json")
        ]

        # fail_if_missing only meaningful for local runner or all
        if args.runner_scope in {'local', 'all'}:
            for artifact in plan.get('fail_if_missing', []):
                if not (repo_root / artifact).exists():
                    missing_artifacts.append(artifact)
            for artifact in contract_check_paths:
                if not (repo_root / artifact).exists() and artifact not in missing_artifacts:
                    missing_artifacts.append(artifact)
                    failure_code = "FAIL_CONTRACT_MISSING"

        path_policy: Dict[str, Any] = {}
        path_policy_file = control_file(repo_root, 'PATH_POLICY.json')
        if path_policy_file.exists():
            path_policy = read_json(path_policy_file)
        allow_paths = path_policy.get('allow_paths', []) if isinstance(path_policy.get('allow_paths', []), list) else []
        forbid_paths = path_policy.get('forbid_paths', []) if isinstance(path_policy.get('forbid_paths', []), list) else []
        protected_paths = path_policy.get('protected_paths', []) if isinstance(path_policy.get('protected_paths', []), list) else []
        baseline_snapshot = {}
        baseline_path = path_boundary_baseline_path(repo_root)
        if baseline_path.exists():
            baseline = read_json(baseline_path)
            if baseline.get("task_truth_stamp") == task_truth_stamp:
                baseline_snapshot = baseline.get("snapshot", {})

        def fail(check_id: str, reason: str) -> None:
            checks_failed.append(check_id)
            notes.append(f"{check_id}: {reason}")

        for check in plan.get('checks', []):
            if not selected(check):
                continue
            cid = check['id']
            ctype = check['type']
            target = check['target']
            checks_run.append(cid)
            try:
                if ctype == 'artifact_exists':
                    if not (repo_root / target).exists():
                        raise FileNotFoundError(target)
                elif ctype == 'schema_check':
                    json_rel, schema_rel = _split_schema_target(target)
                    if not json_rel or not schema_rel:
                        raise ValueError('schema_check target must be <json>::<schema>')
                    data = read_json(repo_root / json_rel)
                    if (repo_root / schema_rel).exists():
                        schema_path = repo_root / schema_rel
                    else:
                        try:
                            schema_path = resolve_package_ref(repo_root, schema_rel)
                        except FileNotFoundError:
                            schema_path = schema_root / Path(schema_rel).name
                    payloads = _records_for_checks(data) if isinstance(data, list) else [data]
                    if isinstance(data, list) and payloads:
                        schema_errors: List[str] = []
                        for idx, payload in enumerate(payloads):
                            ok, errs = validate_json_schema(payload, schema_path)
                            if not ok:
                                schema_errors.extend([f"record {idx}: {err}" for err in errs[:3]])
                                if len(schema_errors) >= 5:
                                    break
                        if schema_errors:
                            raise ValueError('; '.join(schema_errors[:5]))
                    else:
                        ok, errs = validate_json_schema(data, schema_path)
                        if not ok:
                            raise ValueError('; '.join(errs[:5]))
                elif ctype == 'path_boundary_check':
                    delta_paths = _tracked_delta_paths(repo_root, baseline_snapshot)
                    current_task = read_json(control_file(repo_root, "CURRENT_TASK.json"))
                    task_allowed = map_generated_relpaths([
                        str(item).strip()
                        for item in current_task.get("required_outputs", [])
                        if isinstance(item, str) and str(item).strip()
                    ])
                    task_allowed = [item for item in task_allowed if item]
                    violations: List[str] = []
                    for rel in delta_paths:
                        if _matches_any(task_allowed, rel):
                            continue
                        for p in forbid_paths + protected_paths:
                            p2 = p.rstrip('/')
                            if rel == p2 or rel.startswith(p2 + '/'):
                                violations.append(f"{rel} touches forbidden/protected path {p}")
                                break
                        if allow_paths and not _matches_any(allow_paths, rel):
                            violations.append(f"{rel} not in allow_paths")
                    path_boundary_violations = violations
                    path_boundary_status = 'PASS' if not violations else 'FAIL'
                    if violations:
                        failure_code = 'FAIL_PATH_BOUNDARY'
                        raise ValueError('; '.join(violations[:5]))
                elif ctype in {'unit_test','smoke_test','contract_check','vendor_binding_check'}:
                    cp = run_cmd(['bash','-lc', target], cwd=repo_root)
                    if cp.returncode != 0:
                        raise RuntimeError((cp.stderr or cp.stdout).strip()[:1200])
                elif ctype in {'consumer_check', 'required_check'}:
                    current_task = read_json(control_file(repo_root, "CURRENT_TASK.json"))
                    check_errors = _run_package_check(repo_root, str(current_task.get("active_gate", "")), target, str(check.get("phase", "implementation")), current_task)
                    if check_errors:
                        raise ValueError("; ".join(check_errors[:5]))
                else:
                    raise ValueError(f'unhandled check type: {ctype}')
            except Exception as e:
                fail(cid, str(e))

        status = 'PASS'
        if missing_artifacts or checks_failed:
            status = 'FAIL'
        if status == 'FAIL' and not failure_code:
            if any(path.endswith('_contract_check.json') for path in missing_artifacts):
                failure_code = 'FAIL_CONTRACT_MISSING'
    except Exception as exc:
        status = 'FAIL'
        if str(exc).startswith('FAIL_STALE_TASK_TRUTH:'):
            failure_code = 'FAIL_STALE_TASK_TRUTH'
        notes.append(str(exc))

    out = {
        'task_id': read_json(control_file(repo_root, 'CURRENT_TASK.json')).get('task_id', '') if control_file(repo_root, 'CURRENT_TASK.json').exists() else '',
        'active_gate': read_json(control_file(repo_root, 'CURRENT_TASK.json')).get('active_gate', '') if control_file(repo_root, 'CURRENT_TASK.json').exists() else '',
        'task_truth_stamp': task_truth_stamp,
        'phase': args.phase,
        'runner_scope': args.runner_scope,
        'status': status,
        'failure_code': failure_code,
        'checks_run': checks_run,
        'checks_failed': checks_failed,
        'missing_artifacts': missing_artifacts,
        'contract_check_paths': contract_check_paths,
        'path_boundary_status': path_boundary_status,
        'path_boundary_baseline_ref': baseline_ref,
        'path_boundary_violations': path_boundary_violations,
        'notes': notes,
    }
    try:
        require_schema(out, schema_root / 'validation_result.schema.json', 'validation_result')
    except Exception as exc:
        out['status'] = 'FAIL'
        out['failure_code'] = out.get('failure_code') or 'FAIL_TAKEOVER_INCOMPLETE'
        out['notes'] = list(out.get('notes', [])) + [str(exc)]
    write_json(out_path, out)
    return 0 if status == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
