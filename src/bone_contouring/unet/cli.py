"""Minimal batch command; usable locally or in a noninteractive SSH shell."""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Published radius/tibia U-Net compartment segmentation.")
    parser.add_argument("inputs", nargs="+", help="AIM images or input folders (recursive; masks excluded)")
    parser.add_argument("--output", required=True, help="Results folder outside input/raw folders")
    parser.add_argument("--device", choices=("auto","cpu","cuda","mps"), default="auto")
    parser.add_argument("--dataset-root", help="Normalized dataset: add XCTII LH SEG and standard BoneContours manifest")
    args = parser.parse_args(argv)
    try:
        if args.dataset_root:
            if len(args.inputs) != 1:
                raise ValueError("Normalized batch mode requires one AIM case per command.")
            from .batch import run_normalized_case
            run_normalized_case(args.inputs[0], args.output, args.dataset_root, args.device)
        else:
            from .aim import run_batch
            run_batch(args.inputs, args.output, args.device)
    except ImportError as exc:
        print(f"ERROR: U-Net dependencies are unavailable ({exc}). Install 'bone-contouring[unet]'.",
              file=sys.stderr, flush=True)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Cancelled; incomplete cases have no completion marker.", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
