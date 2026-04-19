from __future__ import annotations

import numpy as np

from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _rank_and_top1, _match_rate, STAGE_TO_SELECTED


def test_rank_and_top1_best_case() -> None:
    logits = np.asarray([0.1, 0.7, 0.2], dtype=np.float32)
    rank, norm_rank, top1 = _rank_and_top1(logits, 1)
    assert rank == 1
    assert norm_rank == 0.0
    assert top1 == 1


def test_rank_and_top1_non_top1_case() -> None:
    logits = np.asarray([0.9, 0.4, 0.6, 0.2], dtype=np.float32)
    rank, norm_rank, top1 = _rank_and_top1(logits, 2)
    assert rank == 2
    assert np.isclose(norm_rank, (2 - 1) / (4 - 1))
    assert top1 == 0


def test_match_rate_and_stage_map() -> None:
    assert np.isclose(_match_rate(2, 5), 0.4)
    assert STAGE_TO_SELECTED["prealign"] == "prealign_only"
    assert STAGE_TO_SELECTED["softem_base"] == "base_only"
    assert STAGE_TO_SELECTED["softem_aug"] == "augmented"
