from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableSequence, Sequence, Tuple, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class DynamicBatchPlan:
    batches: List[List[int]]
    batch_budget: int
    batch_count: int
    max_batch_cost: int
    total_cost: int
    min_candidate_count: int
    max_candidate_count: int
    bucket_histogram: Dict[str, int]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_default_batch_budget(*, smoke: bool, explicit: int | None) -> int:
    if explicit is not None:
        value = int(explicit)
        if value <= 0:
            raise ValueError('batch_budget must be positive when provided')
        return value
    payload = _load_json(_repo_root() / 'package' / 'reference' / 'g7_training_execution_defaults.json')
    defaults = payload.get('batch_budget_policy', {}).get('default_budget_by_scope', {})
    scope_key = 'smoke' if bool(smoke) else 'full'
    value = int(defaults.get(scope_key, 8192 if bool(smoke) else 32768))
    if value <= 0:
        raise ValueError(f'invalid default batch budget for scope={scope_key}: {value}')
    return value


def build_dynamic_microbatches(
    records: Sequence[T],
    *,
    batch_budget: int,
    cost_fn: Callable[[T], int],
    bucket_key_fn: Callable[[T], Tuple[int, int]],
) -> DynamicBatchPlan:
    if batch_budget <= 0:
        raise ValueError('batch_budget must be positive')
    indexed = list(enumerate(records))
    indexed.sort(key=lambda item: (bucket_key_fn(item[1]), item[0]))

    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_cost = 0
    bucket_histogram: Dict[str, int] = {}
    total_cost = 0
    max_batch_cost = 0
    candidate_counts: List[int] = []

    for index, record in indexed:
        raw_cost = int(cost_fn(record))
        cost = max(1, raw_cost)
        bucket_key = bucket_key_fn(record)
        bucket_histogram[f'{int(bucket_key[0])}x{int(bucket_key[1])}'] = int(bucket_histogram.get(f'{int(bucket_key[0])}x{int(bucket_key[1])}', 0)) + 1
        candidate_counts.append(int(bucket_key[1]))
        total_cost += cost
        if current_batch and current_cost + cost > int(batch_budget):
            batches.append(list(current_batch))
            max_batch_cost = max(max_batch_cost, int(current_cost))
            current_batch = []
            current_cost = 0
        current_batch.append(int(index))
        current_cost += int(cost)
        if current_cost >= int(batch_budget):
            batches.append(list(current_batch))
            max_batch_cost = max(max_batch_cost, int(current_cost))
            current_batch = []
            current_cost = 0
    if current_batch:
        batches.append(list(current_batch))
        max_batch_cost = max(max_batch_cost, int(current_cost))

    if not candidate_counts:
        candidate_counts = [0]
    return DynamicBatchPlan(
        batches=batches,
        batch_budget=int(batch_budget),
        batch_count=int(len(batches)),
        max_batch_cost=int(max_batch_cost),
        total_cost=int(total_cost),
        min_candidate_count=int(min(candidate_counts)),
        max_candidate_count=int(max(candidate_counts)),
        bucket_histogram=bucket_histogram,
    )
