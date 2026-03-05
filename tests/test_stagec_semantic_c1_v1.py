from __future__ import annotations

import numpy as np

from wsovvis.track_feature_export import (
    STAGEC_BG_LABEL_ID,
    STAGEC_UNK_FG_LABEL_ID,
    compute_stagec_clip_text_cache_key_v1,
    get_or_build_stagec_clip_text_prototype_cache_v1,
)
from wsovvis.track_feature_export.stagec_semantic_slice_v1 import (
    build_stagec_candidate_set_v1,
    merge_stagec_candidate_label_ids_v1,
    select_stagec_topk_label_ids_from_scores_v1,
)
from wsovvis.training import build_stagec_semantic_plumbing_c1_clip_text_default


def test_candidate_merge_dedup_stable_order() -> None:
    merged = merge_stagec_candidate_label_ids_v1(
        positive_label_ids=(3, 1, 3, 2),
        topk_label_ids=(2, 9, 1, 7),
        merge_mode="yp_plus_topk",
    )
    assert merged == (3, 1, 2, 9, 7)


def test_candidate_set_yp_only() -> None:
    out = build_stagec_candidate_set_v1(
        n_track=2,
        positive_label_ids=(5, 2, 2),
        merge_mode="yp_only",
    )
    assert out.candidate_label_ids == (5, 2)
    assert out.candidate_matrix.shape == (2, 2)
    assert np.allclose(out.candidate_matrix, 1.0)


def test_candidate_set_topk_only_with_score_adapter() -> None:
    topk = select_stagec_topk_label_ids_from_scores_v1(
        label_score_items=((10, 0.1), (11, 0.9), (10, 0.7), (12, 0.7)),
        topk_k=2,
    )
    out = build_stagec_candidate_set_v1(
        n_track=1,
        topk_label_ids=topk,
        merge_mode="topk_only",
    )
    assert topk == (11, 10)
    assert out.candidate_label_ids == (11, 10)
    assert out.candidate_matrix.shape == (1, 2)


def test_candidate_set_yp_plus_topk_and_optional_slots() -> None:
    out = build_stagec_candidate_set_v1(
        n_track=3,
        positive_label_ids=(1, 2),
        topk_label_ids=(2, 8),
        merge_mode="yp_plus_topk",
        include_bg=True,
        include_unk_fg=True,
    )
    assert out.candidate_label_ids == (1, 2, 8, STAGEC_BG_LABEL_ID, STAGEC_UNK_FG_LABEL_ID)
    assert out.candidate_matrix.shape == (3, 5)


def test_clip_text_cache_miss_then_hit_and_deterministic_key(tmp_path) -> None:
    cache_root = tmp_path / "clip_cache"
    labels = (1, 5, STAGEC_BG_LABEL_ID)
    key1 = compute_stagec_clip_text_cache_key_v1(
        candidate_label_ids=labels,
        label_text_by_id={1: "person", 5: "dog", STAGEC_BG_LABEL_ID: "background"},
        model_name="clip-vit-b32",
        prompt_variant="default",
        embedding_dim=8,
    )
    key2 = compute_stagec_clip_text_cache_key_v1(
        candidate_label_ids=labels,
        label_text_by_id={1: "person", 5: "dog", STAGEC_BG_LABEL_ID: "background"},
        model_name="clip-vit-b32",
        prompt_variant="default",
        embedding_dim=8,
    )
    assert key1 == key2

    miss = get_or_build_stagec_clip_text_prototype_cache_v1(
        cache_root=cache_root,
        candidate_label_ids=labels,
        label_text_by_id={1: "person", 5: "dog", STAGEC_BG_LABEL_ID: "background"},
        model_name="clip-vit-b32",
        prompt_variant="default",
        embedding_dim=8,
    )
    assert miss.cache_hit is False
    assert miss.prototype_features.shape == (3, 8)
    assert miss.metadata_path.exists()
    assert miss.tensor_path.exists()

    hit = get_or_build_stagec_clip_text_prototype_cache_v1(
        cache_root=cache_root,
        candidate_label_ids=labels,
        label_text_by_id={1: "person", 5: "dog", STAGEC_BG_LABEL_ID: "background"},
        model_name="clip-vit-b32",
        prompt_variant="default",
        embedding_dim=8,
    )
    assert hit.cache_hit is True
    assert hit.cache_key == miss.cache_key
    assert np.allclose(hit.prototype_features, miss.prototype_features)
    assert hit.candidate_label_ids == labels


def test_stagec_c1_minimal_smoke_path(tmp_path) -> None:
    track_features = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    out = build_stagec_semantic_plumbing_c1_clip_text_default(
        {"enabled": True, "loss_key": "loss_stage_c_semantic"},
        track_features=track_features,
        positive_label_ids=(3, 7),
        topk_label_ids=(7, 9),
        merge_mode="yp_plus_topk",
        include_bg=True,
        cache_root=tmp_path / "clip_cache_smoke",
    )
    assert out["enabled"] is True
    assert out["interface_version"] == "stagec_semantic_c1_v1"
    assert out["shape_summary"] == {"F": (2, 4), "G": (4, 4), "Y_hat": (2, 4), "P": (2, 4)}
    assert out["candidate_summary"]["candidate_label_ids"] == (3, 7, 9, STAGEC_BG_LABEL_ID)
    assert out["prototype_cache"]["backend"] == "clip_text_default_v1"
