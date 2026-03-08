from typing import List
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom

from ..util.misc import interpolate


class Videos:
    """
    This structure is to support Section 3.3: Clip-level instance tracking.
    NOTE most errors occuring in this structure is due to
    the number of predictions exceeding num_max_inst.
    TODO make further GPU-memory friendly while maintaining speed,
    and max number of instances be dynamically changed.
    """
    def __init__(self, num_frames, video_length, num_classes, image_size, device):
        self.num_frames = num_frames
        self.video_length = video_length
        self.device = device

        num_max_inst = 120
        self.match_threshold = 0.01
        self.geometry_weight = 0.5
        self.query_weight = 0.5
        self.quality_epsilon = 1e-6

        self.num_inst = 0
        self.num_clip = 0
        self.saved_idx_set = set()

        self.saved_logits = torch.zeros((video_length, num_max_inst, self.video_length, *image_size), dtype=torch.float, device=device)
        self.saved_masks  = torch.zeros((video_length, num_max_inst, self.video_length, *image_size), dtype=torch.float, device=device)
        self.saved_valid  = torch.zeros((video_length, num_max_inst, self.video_length), dtype=torch.bool, device=device)
        self.saved_cls_logits = torch.zeros((video_length, num_max_inst, num_classes), dtype=torch.float, device=device)
        self.saved_quality = torch.zeros((video_length, num_max_inst), dtype=torch.float, device=device)
        self.saved_query_embeddings = None
        self.saved_query_weights = None

    def get_siou(self, input_masks, saved_masks, saved_valid):
        # input_masks : N_i, T, H, W
        # saved_masks : C, N_s, T, H, W
        # saved_valid : C, N_s, T

        input_masks = input_masks.flatten(-2)   #    N_i, T, HW
        saved_masks = saved_masks.flatten(-2)   # C, N_s, T, HW

        input_masks = input_masks[None, None]   # 1, 1, N_i, T, HW
        saved_masks = saved_masks.unsqueeze(2)  # C, N_s, 1, T, HW
        saved_valid = saved_valid[:, :, None, :, None]  # C, N_s, 1, T, 1

        # C, N_s, N_i, T, HW
        numerator = saved_masks * input_masks
        denominator = saved_masks + input_masks - saved_masks * input_masks

        numerator = (numerator * saved_valid).sum(dim=(-1, -2))
        denominator = (denominator * saved_valid).sum(dim=(-1, -2))

        siou = numerator / (denominator + 1e-6) # C, N_s, N_i

        # To divide only the frames that are being compared
        num_valid_clip = (saved_valid.flatten(2).sum(dim=2) > 0).sum(dim=0) # N_s,

        siou = siou.sum(dim=0) / (num_valid_clip[..., None] + 1e-6)

        return siou

    def _ensure_query_storage(self, embedding_dim):
        if self.saved_query_embeddings is None:
            num_max_inst = self.saved_masks.shape[1]
            self.saved_query_embeddings = torch.zeros((num_max_inst, embedding_dim), dtype=torch.float, device=self.device)
            self.saved_query_weights = torch.zeros((num_max_inst,), dtype=torch.float, device=self.device)
            return
        if self.saved_query_embeddings.shape[1] != embedding_dim:
            raise RuntimeError(
                f"query embedding dim mismatch: expected {self.saved_query_embeddings.shape[1]}, got {embedding_dim}"
            )

    def _resolve_query_scores(self, input_clip):
        if input_clip.query_embeddings is None or self.num_inst == 0:
            return None
        self._ensure_query_storage(input_clip.query_embeddings.shape[1])
        saved_weights = self.saved_query_weights[:self.num_inst]
        if not torch.any(saved_weights > 0):
            return None
        saved_queries = self.saved_query_embeddings[:self.num_inst]
        saved_queries = F.normalize(saved_queries, dim=-1, eps=1e-6)
        input_queries = F.normalize(input_clip.query_embeddings, dim=-1, eps=1e-6)
        query_scores = torch.matmul(saved_queries, input_queries.transpose(0, 1))
        query_scores = (query_scores + 1.0) * 0.5
        query_scores = query_scores * (saved_weights[:, None] > 0).float()
        return query_scores

    def _update_saved_query(self, track_idx, query_embedding, quality_score):
        if query_embedding is None:
            return
        self._ensure_query_storage(query_embedding.shape[0])
        quality = float(max(float(quality_score), self.quality_epsilon))
        prev_weight = float(self.saved_query_weights[track_idx].item())
        if prev_weight <= 0.0:
            self.saved_query_embeddings[track_idx] = query_embedding
            self.saved_query_weights[track_idx] = quality
            return
        total_weight = prev_weight + quality
        self.saved_query_embeddings[track_idx] = (
            self.saved_query_embeddings[track_idx] * prev_weight + query_embedding * quality
        ) / total_weight
        self.saved_query_weights[track_idx] = total_weight

    def update(self, input_clip):
        # gather intersection
        inter_input_idx, inter_saved_idx = [], []
        for o_i, f_i in enumerate(input_clip.frame_idx):
            if f_i in self.saved_idx_set:
                inter_input_idx.append(o_i)
                inter_saved_idx.append(f_i)

        # compute sIoU
        i_masks = input_clip.mask_probs[:, inter_input_idx]
        s_masks = self.saved_masks[
            max(self.num_clip-len(input_clip.frame_idx), 0) : self.num_clip, : self.num_inst, inter_saved_idx
        ]
        s_valid = self.saved_valid[
            max(self.num_clip-len(input_clip.frame_idx), 0) : self.num_clip, : self.num_inst, inter_saved_idx
        ]

        scores = self.get_siou(i_masks, s_masks, s_valid)   # N_s, N_i
        query_scores = self._resolve_query_scores(input_clip)
        if query_scores is not None:
            # Geometry remains the gate; query similarity resolves ambiguous overlaps.
            scores = self.geometry_weight * scores + self.query_weight * query_scores
        cls_logits = input_clip.cls_logits
        quality_scores = input_clip.quality_scores.clamp(min=self.quality_epsilon)

        # bipartite match
        above_thres = self.get_siou(i_masks, s_masks, s_valid) > self.match_threshold
        scores = scores * above_thres.float()

        row_idx, col_idx = linear_sum_assignment(scores.cpu(), maximize=True)

        existed_idx = []
        for is_above, r, c in zip(above_thres[row_idx, col_idx], row_idx, col_idx):
            if not is_above:
                continue

            self.saved_logits[self.num_clip, r, input_clip.frame_idx] = input_clip.mask_logits[c]
            self.saved_masks[self.num_clip, r, input_clip.frame_idx] = input_clip.mask_probs[c]
            self.saved_valid[self.num_clip, r, input_clip.frame_idx] = True
            self.saved_cls_logits[self.num_clip, r] = cls_logits[c]
            self.saved_quality[self.num_clip, r] = quality_scores[c]
            if input_clip.query_embeddings is not None:
                self._update_saved_query(r, input_clip.query_embeddings[c], quality_scores[c])
            existed_idx.append(c)

        left_idx = [i for i in range(input_clip.num_instance) if i not in existed_idx]
        try:
            self.saved_logits[self.num_clip,
                            self.num_inst:self.num_inst+len(left_idx),
                            input_clip.frame_idx] = input_clip.mask_logits[left_idx]
        except:
            print('shape mismatch error!')
        self.saved_masks[self.num_clip,
                         self.num_inst:self.num_inst+len(left_idx),
                         input_clip.frame_idx] = input_clip.mask_probs[left_idx]
        self.saved_valid[self.num_clip,
                         self.num_inst:self.num_inst+len(left_idx),
                         input_clip.frame_idx] = True
        self.saved_cls_logits[self.num_clip, self.num_inst:self.num_inst+len(left_idx)] = cls_logits[left_idx]
        self.saved_quality[self.num_clip, self.num_inst:self.num_inst+len(left_idx)] = quality_scores[left_idx]
        if input_clip.query_embeddings is not None:
            for offset, clip_idx in enumerate(left_idx):
                self._update_saved_query(
                    self.num_inst + offset,
                    input_clip.query_embeddings[clip_idx],
                    quality_scores[clip_idx],
                )

        # Update status
        self.saved_idx_set.update(input_clip.frame_set)
        self.num_clip += 1
        self.num_inst += len(left_idx)

    def get_result(self,):
        _mask_logits = self.saved_logits[:self.num_clip, :self.num_inst]
        _valid = self.saved_valid[:self.num_clip, :self.num_inst]
        _cls_logits = self.saved_cls_logits[:self.num_clip, :self.num_inst]
        _quality = self.saved_quality[:self.num_clip, :self.num_inst]

        _mask_logits = _mask_logits.sum(dim=0) / _valid.sum(dim=0)[..., None, None]

        cls_valid = (_valid.sum(dim=2) > 0).float()
        cls_weights = (_quality * cls_valid).clamp(min=0.0)
        cls_weight_sum = cls_weights.sum(dim=0).clamp(min=self.quality_epsilon)
        out_cls_logits = (_cls_logits * cls_weights[..., None]).sum(dim=0) / cls_weight_sum[..., None]
        out_cls = out_cls_logits.sigmoid()
        # out_masks = retry_if_cuda_oom(lambda x: x > 0.0)(_mask_logits)

        return out_cls, _mask_logits


class Clips:
    def __init__(self, frame_idx: List[int], results: List[Instances]):
        self.frame_idx = frame_idx
        self.frame_set = set(frame_idx)

        self.classes = results.pred_classes
        self.scores = results.scores
        self.cls_probs = results.cls_probs
        if results.has("cls_logits"):
            self.cls_logits = results.cls_logits
        else:
            self.cls_logits = torch.logit(self.cls_probs.clamp(min=1e-4, max=1 - 1e-4))
        if results.has("quality_scores"):
            self.quality_scores = results.quality_scores
        else:
            self.quality_scores = self.scores
        if results.has("query_embeddings"):
            self.query_embeddings = results.query_embeddings
        else:
            self.query_embeddings = None
        self.mask_logits = results.pred_masks
        self.mask_probs = results.pred_masks.sigmoid()

        self.num_instance = len(self.scores)
