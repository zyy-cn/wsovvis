from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def set_coverage_recall(gt_entities: Iterable[int], predicted_entities: Iterable[int]) -> float:
    """Compute Set-Coverage Recall (SCR) for one video.

    SCR(Y*, Y_hat) = |Y* ∩ Y_hat| / |Y*|, with SCR=1 when |Y*|=0.
    """

    gt = set(gt_entities)
    pred = set(predicted_entities)
    if not gt:
        return 1.0
    return float(len(gt & pred)) / float(len(gt))


def build_missing_rate_curve(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Validate and sort robustness points as (missing_rate, recall).

    Each point must satisfy missing_rate in [0, 1] and recall in [0, 1].
    """

    curve = sorted((float(m), float(r)) for m, r in points)
    if not curve:
        raise ValueError("curve must contain at least one point")

    dedup: dict[float, float] = {}
    for m, r in curve:
        if not (0.0 <= m <= 1.0):
            raise ValueError(f"missing_rate out of range: {m}")
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"recall out of range: {r}")
        dedup[m] = r
    return sorted(dedup.items())


def missing_rate_curve_from_predictions(
    gt_entities: Iterable[int],
    predictions_by_missing_rate: Mapping[float, Iterable[int]],
) -> list[tuple[float, float]]:
    """Build missing-rate robustness curve from per-rate predicted sets.

    For each missing rate m, recall(m) = SCR(Y*, Y_hat_m).
    """

    points = [
        (float(missing_rate), set_coverage_recall(gt_entities, predicted))
        for missing_rate, predicted in predictions_by_missing_rate.items()
    ]
    return build_missing_rate_curve(points)


def aurc_from_curve(curve: Sequence[tuple[float, float]]) -> float:
    """Compute normalized AURC using trapezoidal integration.

    AURC = (1 / (m_max - m_min)) * integral recall(m) dm over the provided
    missing-rate domain. If only one point is provided, returns its recall.
    """

    pts = build_missing_rate_curve(curve)
    if len(pts) == 1:
        return pts[0][1]

    m_min = pts[0][0]
    m_max = pts[-1][0]
    if m_max == m_min:
        return pts[0][1]

    area = 0.0
    for (m0, r0), (m1, r1) in zip(pts[:-1], pts[1:]):
        area += (m1 - m0) * (r0 + r1) * 0.5
    return area / (m_max - m_min)
