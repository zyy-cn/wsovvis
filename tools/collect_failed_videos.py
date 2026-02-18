#!/usr/bin/env python3
import argparse, glob, os, re
from pathlib import Path

VID_RE = re.compile(r"\[GPU\s+\d+\]\s+(\d+)")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory", re.IGNORECASE)
TB_RE  = re.compile(r"Traceback \(most recent call last\):")

def parse_logs(log_glob: str):
    oom, exc = set(), set()
    for lf in glob.glob(log_glob):
        cur = None
        in_tb = False
        tb_has_oom = False

        with open(lf, "r", errors="ignore") as f:
            for line in f:
                m = VID_RE.search(line)
                if m:
                    cur = m.group(1)
                    in_tb = False
                    tb_has_oom = False
                    continue

                if TB_RE.search(line):
                    in_tb = True
                    tb_has_oom = False
                    continue

                if in_tb and OOM_RE.search(line):
                    tb_has_oom = True

                # 结束 traceback：以新的 [GPU x] 或 detectron2 Arguments 等作为“分段”
                # 这里简单地：如果 in_tb 并且出现下一段视频开始标志，上一段就算完
                # （实际上上面 VID_RE 会 reset）

                # 直接抓 oom 行也可以（有时不在 traceback 中）
                if OOM_RE.search(line):
                    if cur:
                        oom.add(cur)

            # 文件结束时，如果最后停在 traceback，也归类一次
            if in_tb and cur:
                (oom if tb_has_oom else exc).add(cur)

    # 把 oom 从 exc 里剔除
    exc -= oom
    return oom, exc

def collect_incomplete(img_root: str, mask_root: str):
    """mask_*.png < frames count => incomplete; 0 => empty"""
    incomplete = set()
    img_root = Path(img_root)
    mask_root = Path(mask_root)

    for vid_dir in sorted(img_root.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        frames = sorted(vid_dir.glob("*.jpg"))
        if len(frames) == 0:
            continue

        out_dir = mask_root / vid
        masks = sorted(out_dir.glob("mask_*.png")) if out_dir.exists() else []
        if 0 < len(masks) < len(frames):
            incomplete.add(vid)
    return incomplete

def read_list(p):
    s=set()
    if p and os.path.exists(p):
        with open(p, "r") as f:
            for line in f:
                t=line.strip()
                if t:
                    s.add(t)
    return s

def write_list(p, items):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for x in sorted(items):
            f.write(x + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_glob", required=True, help="e.g. outputs/.../logs/gpu*.log")
    ap.add_argument("--img_root", required=True, help="e.g. data/LV-VIS/train/JPEGImages")
    ap.add_argument("--mask_root", required=True, help="e.g. outputs/.../train/masks")
    ap.add_argument("--empty_list", default="", help="e.g. outputs/.../train/masks/empty_videos.txt")
    ap.add_argument("--out_dir", default="outputs/videocutler_lvvis_png/train/retry_lists")
    args = ap.parse_args()

    oom, exc = parse_logs(args.log_glob)
    incomplete = collect_incomplete(args.img_root, args.mask_root)
    empty = read_list(args.empty_list)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_list(str(out_dir/"failed_oom.txt"), oom)
    write_list(str(out_dir/"failed_exception.txt"), exc)
    write_list(str(out_dir/"failed_incomplete.txt"), incomplete)
    write_list(str(out_dir/"failed_empty.txt"), empty)

    all_failed = set().union(oom, exc, incomplete, empty)
    write_list(str(out_dir/"failed_all.txt"), all_failed)

    print(f"[collect] oom={len(oom)} exc={len(exc)} incomplete={len(incomplete)} empty={len(empty)} all={len(all_failed)}")
    print(f"[collect] lists saved to: {out_dir}")

if __name__ == "__main__":
    main()
