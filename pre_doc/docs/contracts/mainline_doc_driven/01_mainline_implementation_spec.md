# Mainline Implementation Spec

本合同定义当前主线的“唯一默认实现版本”，用于降低 Codex 在关键研究决策上的自由度。

## 1. 协议层
### 1.1 数据协议
- 训练单位：固定长度 window，stride = T/2。
- window 级真实标签集合为 `Y*(w)`，训练使用不完备正证据集合 `Y'(w)`。
- 主线只默认启用：`uniform` 与 `long_tail` 两种 missing 协议。

### 1.2 仓库锚点
- 协议工具：`tools/build_wsovvis_labelset_protocol.py`
- 协议测试：`tests/test_build_wsovvis_labelset_protocol.py`

## 2. Basis 层
### 2.1 Stage B 输出链
当前主线默认复用仓库内已存在的：
- `wsovvis/track_feature_export/*`
- `tools/build_stageb_track_feature_export_v1.py`
- `tools/build_stageb_bridge_input_from_real_stageb_sidecar_v1.py`
- 相关 `tests/test_stageb_*`

### 2.2 原则
- 不重新定义 Stage B schema；
- 新任务优先复用现有 export / bridge / loader；
- 若 schema 需要变更，必须先新增 contract，再允许实现。

## 3. 语义桥层
### 3.1 默认实现
- mixed track representation = query projection + mask-pooled CLIP visual feature
- CLIP text prototypes 作为默认文本空间
- 主线默认只做轻量桥接，不做大规模 joint fine-tune

### 3.2 仓库锚点
- `wsovvis/track_feature_export/stagec_clip_text_prototype_cache_v1.py`
- `wsovvis/track_feature_export/stagec_loader_v1.py`
- `wsovvis/track_feature_export/stagec_semantic_slice_v1.py`
- `tests/test_stagec_loader_v1.py`
- `tests/test_stagec_semantic_slice_v1.py`

## 4. Attribution 层
### 4.1 默认启用顺序（强制）
必须按以下顺序逐步启用：
1. closed-world baseline
2. open-world（仅 `Y'(w)` + bg + unk）
3. `+ coverage penalty`
4. `+ retrieved candidates`

不得跳级直接堆满所有机制。

### 4.2 默认实现
- bg / unk 为显式列
- coverage 使用 soft penalty
- retrieved candidates 默认保守开启
- 若实现过重，可退化为 row-wise soft assignment + bg/unk + coverage

### 4.3 仓库锚点
- `wsovvis/track_feature_export/stagec1_attribution_mil_v1.py`
- `wsovvis/training/stagec_semantic_plumbing_v1.py`
- `wsovvis/training/staged_attribution_plumbing_v1.py`
- `tests/test_stagec1_attribution_mil_v1.py`
- `tests/test_stagec3_otlite_decoder_v1.py`
- `tests/test_stagec4_sinkhorn_scorer_v1.py`
- `tests/test_stagec4_em_scorer_v1.py`

## 5. Linking 层
### 5.1 默认实现
- 先做 adjacent-window linking
- linking 默认以 IoU + query consistency 为主
- 语义项仅作弱辅助
- global class 默认采用 quality-weighted logit average
- memory aggregation 仅在 G4 通过后作为增强项考虑

## 6. Refinement 层
### 6.1 默认实现
- 只做 1 轮 refinement
- keep/drop 以 mask / temporal 为主
- semantic 只作辅助判定，不主导 keep/drop
- label-set expansion 默认关闭

### 6.2 主线退出条件
若 refinement 引入负迁移或显著不稳定，则整块降级为附录增强，不保留在主线默认实现中。
