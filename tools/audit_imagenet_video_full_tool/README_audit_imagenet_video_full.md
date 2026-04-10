# Full audit for ImageNet synthetic video annotations

## What it checks
- Missing / unreadable image files
- Image actual size vs video JSON `height/width`
- `video.length` vs `file_names` length
- Annotation `segmentations` / `bboxes` length vs `video_length`
- RLE `size` vs actual image size
- Polygon validity
- Track/frame presence inconsistencies that may violate official `copy_and_paste()` assumptions

## Default command
```bash
cd /mnt/sda/zyy/code/wsovvis
python /mnt/data/audit_imagenet_video_full.py \
  --datasets-root datasets \
  --json datasets/imagenet/annotations/video_imagenet_train_fixsize480_tau0.15_N3.json \
  --out-dir audit_imagenet_video_full_out
```

## Optional deeper RLE validation
If `pycocotools` is installed:
```bash
python /mnt/data/audit_imagenet_video_full.py \
  --datasets-root datasets \
  --json datasets/imagenet/annotations/video_imagenet_train_fixsize480_tau0.15_N3.json \
  --out-dir audit_imagenet_video_full_out \
  --decode-rle
```

## Outputs
- `summary.json`
- `findings.jsonl`
- `report.md`
