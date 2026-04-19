from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videocutler.ext_stageb_ovvis.eval.formal_plan import build_formal_package_manifest, write_manifest_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare add-only G8 formal execution package manifests.')
    parser.add_argument('--package_kind', required=True, choices=('full_single_arm_sanity', 'paper_main_4arm', 'package_phase3_full', 'eval_only_existing_workspace'))
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--logit_chunk_size', required=True, type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--manifest_path', default=None)
    parser.add_argument('--shell_path', default=None)
    parser.add_argument('--summary_path', default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve() if args.manifest_path else run_root / 'g8_formal_package_manifest.json'
    shell_path = Path(args.shell_path).expanduser().resolve() if args.shell_path else run_root / 'g8_formal_package_commands.sh'
    summary_path = Path(args.summary_path).expanduser().resolve() if args.summary_path else run_root / 'g8_formal_package_summary.md'
    manifest = build_formal_package_manifest(
        package_kind=args.package_kind,
        run_root=run_root,
        device=args.device,
        seed=int(args.seed),
        logit_chunk_size=int(args.logit_chunk_size),
        smoke=bool(args.smoke),
    )
    write_manifest_bundle(manifest=manifest, manifest_path=manifest_path, shell_path=shell_path, summary_path=summary_path)
    print(manifest_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
