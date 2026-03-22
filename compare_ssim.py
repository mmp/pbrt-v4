#!/usr/bin/env python3
# SPDX: same as pbrt-v4 / Apache-2.0 where applicable
"""Print one SSIM value (0-1) for two images to stdout. Used by compare-skipmip.ps1.

Requires (same resolution, full image, RGB or grayscale as in skimage):
  pip install numpy scikit-image pillow
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_ssim.py <image_a> <image_b>", file=sys.stderr)
        return 1
    try:
        import numpy as np
        from PIL import Image
        from skimage.metrics import structural_similarity as ssim
    except ImportError as e:
        print(
            "ImportError: %s\nInstall: pip install numpy scikit-image pillow" % e,
            file=sys.stderr,
        )
        return 2
    paths = sys.argv[1:3]
    for p in paths:
        if not Path(p).is_file():
            print("not found: %s" % p, file=sys.stderr)
            return 3

    def load(path: str) -> "np.ndarray":
        im = Image.open(path)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        return np.asarray(im)

    a, b = load(paths[0]), load(paths[1])
    if a.shape != b.shape:
        print("shape mismatch: %s vs %s" % (a.shape, b.shape), file=sys.stderr)
        return 4
    if a.ndim == 2:
        val = float(ssim(a, b, data_range=255))
    else:
        try:
            val = float(ssim(a, b, channel_axis=2, data_range=255))
        except TypeError:
            val = float(ssim(a, b, multichannel=True, data_range=255))
    print("%.6f" % val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
