from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

clip_output = pytest.importorskip("detectron2.projects.seqformer.models.clip_output")

Videos = clip_output.Videos


def _build_clip(
    *,
    frame_idx: list[int],
    mask_fills: list[float],
    cls_probs: list[list[float]],
    quality_scores: list[float],
    query_embeddings: list[list[float]],
):
    num_frames = len(frame_idx)
    mask_logits = torch.stack(
        [torch.full((num_frames, 2, 2), fill_value=float(fill), dtype=torch.float32) for fill in mask_fills],
        dim=0,
    )
    cls_probs_tensor = torch.tensor(cls_probs, dtype=torch.float32)
    quality_scores_tensor = torch.tensor(quality_scores, dtype=torch.float32)
    query_embeddings_tensor = torch.tensor(query_embeddings, dtype=torch.float32)
    cls_logits_tensor = torch.logit(cls_probs_tensor.clamp(min=1e-4, max=1 - 1e-4))
    return SimpleNamespace(
        frame_idx=frame_idx,
        frame_set=set(frame_idx),
        scores=quality_scores_tensor,
        cls_probs=cls_probs_tensor,
        cls_logits=cls_logits_tensor,
        quality_scores=quality_scores_tensor,
        query_embeddings=query_embeddings_tensor,
        mask_logits=mask_logits,
        mask_probs=mask_logits.sigmoid(),
        num_instance=len(mask_fills),
    )


def test_videos_update_requires_geometry_gate_before_query_match() -> None:
    state = Videos(num_frames=2, video_length=3, num_classes=1, image_size=(2, 2), device="cpu")
    state.update(
        _build_clip(
            frame_idx=[0, 1],
            mask_fills=[8.0],
            cls_probs=[[0.4]],
            quality_scores=[0.8],
            query_embeddings=[[1.0, 0.0]],
        )
    )

    state.update(
        _build_clip(
            frame_idx=[1, 2],
            mask_fills=[-8.0],
            cls_probs=[[0.9]],
            quality_scores=[0.8],
            query_embeddings=[[1.0, 0.0]],
        )
    )

    assert state.num_inst == 2


def test_videos_update_uses_query_similarity_before_semantics_when_geometry_ties() -> None:
    state = Videos(num_frames=2, video_length=3, num_classes=1, image_size=(2, 2), device="cpu")
    state.update(
        _build_clip(
            frame_idx=[0, 1],
            mask_fills=[8.0],
            cls_probs=[[0.3]],
            quality_scores=[0.7],
            query_embeddings=[[1.0, 0.0]],
        )
    )

    state.update(
        _build_clip(
            frame_idx=[1, 2],
            mask_fills=[8.0, 8.0],
            cls_probs=[[0.99], [0.2]],
            quality_scores=[0.7, 0.7],
            query_embeddings=[[-1.0, 0.0], [1.0, 0.0]],
        )
    )

    pred_cls, _ = state.get_result()

    assert state.num_inst == 2
    assert state.saved_query_embeddings is not None
    assert state.saved_query_embeddings[0, 0].item() > 0.9
    assert pred_cls[0, 0].item() < 0.4
    assert pred_cls[1, 0].item() > 0.9


def test_videos_get_result_uses_quality_weighted_logit_averaging() -> None:
    state = Videos(num_frames=2, video_length=3, num_classes=1, image_size=(2, 2), device="cpu")
    state.update(
        _build_clip(
            frame_idx=[0, 1],
            mask_fills=[8.0],
            cls_probs=[[0.2]],
            quality_scores=[0.9],
            query_embeddings=[[1.0, 0.0]],
        )
    )
    state.update(
        _build_clip(
            frame_idx=[1, 2],
            mask_fills=[8.0],
            cls_probs=[[0.9]],
            quality_scores=[0.1],
            query_embeddings=[[1.0, 0.0]],
        )
    )

    pred_cls, _ = state.get_result()
    expected_logit = 0.9 * torch.logit(torch.tensor(0.2)) + 0.1 * torch.logit(torch.tensor(0.9))
    expected_prob = torch.sigmoid(expected_logit).item()

    assert pred_cls[0, 0].item() == pytest.approx(expected_prob, abs=1e-6)
    assert pred_cls[0, 0].item() < (0.2 + 0.9) / 2.0
