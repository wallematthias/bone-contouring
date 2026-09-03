"""Command line entry point for bone-contouring workflows."""

from __future__ import annotations

import argparse

from .batch import run_bone_contouring_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bone-contouring")
    commands = parser.add_subparsers(dest="command")
    batch = commands.add_parser("run-batch", help="write BoneContours derivatives for a normalized dataset")
    batch.add_argument("dataset_root")
    batch.add_argument("--modality", default="xct1")
    batch.add_argument("--site", default="radius")
    batch.add_argument("--segmentation", default="laplace_hamming")
    batch.add_argument("--outer-contour", default="standard")
    batch.add_argument("--inner-contour", default="standard")
    batch.add_argument("--profile", default="")
    batch.add_argument("--subject", default="")
    batch.add_argument("--session", default="")
    batch.add_argument("--voi", default="")
    batch.add_argument("--output-root", default="")
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "run-batch":
        records = run_bone_contouring_batch(
            args.dataset_root,
            modality=args.modality,
            site=args.site,
            segmentation=args.segmentation,
            outer_contour=args.outer_contour,
            inner_contour=args.inner_contour,
            profile=args.profile,
            subject_id=args.subject,
            session_id=args.session,
            voi=args.voi,
            output_root=args.output_root or None,
            force=args.force,
            dry_run=args.dry_run,
        )
        print(f"wrote {len(records)} BoneContours artifact(s)")
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
