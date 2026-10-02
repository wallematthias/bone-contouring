"""Internal Slicer worker: frozen BMD NPZ -> masks NPZ. No Slicer imports."""
import argparse
from pathlib import Path

import numpy as np

from .inference import Segmenter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--device", choices=("auto","cpu","cuda","mps"), default="auto")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.input, allow_pickle=False) as snapshot:
        density = snapshot["density"]
    masks = Segmenter(args.device).segment(density, progress=lambda message: print(message, flush=True))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **masks)
    print(f"Completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
