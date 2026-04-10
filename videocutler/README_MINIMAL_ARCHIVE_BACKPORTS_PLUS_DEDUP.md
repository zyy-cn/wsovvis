# Minimal archive backports + duplicate static-frame decode de-dup

This overlay is based on the previous minimal archive backports package and keeps its existing enhancements:

1. dataset_mapper.py
   - skip malformed segmentation only for AssertionError raised by
     transform_instance_annotations(...)
   - all other exceptions still raise

2. train_loop.py
   - frame-specific valid_choice filtering before copy_and_paste indexing
   - if the chosen frame has no legal indices, skip this copy-paste branch

3. train_loop.py
   - if keep_count == 0 after IoU filtering, revert to original target sample

4. NEW in this overlay: dataset_mapper.py
   - per-sample duplicate static-image decode de-dup
   - only deduplicates exact same file_name within the current sample
   - always returns image.copy() after cache hit/miss to avoid aliasing

Why this is low-risk:
- no global cache
- no cross-sample sharing
- no change for normal VIS samples with unique file_names
- only reduces repeated read_image / JPEG decode on duplicate static-frame pseudo videos
