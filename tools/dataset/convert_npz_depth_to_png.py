#!/usr/bin/env python
"""Convert NPZ depth modality files to single-channel PNG files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_npz_array(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=True)
    try:
        keys = list(raw.files)
        if "arr_0" in keys:
            array = raw["arr_0"]
        elif len(keys) == 1:
            array = raw[keys[0]]
        else:
            raise ValueError(f"NPZ file with multiple arrays must contain key 'arr_0', got keys={keys} for {path}")
    finally:
        raw.close()
    return np.asarray(array)


def depth_to_uint8(array: np.ndarray) -> np.ndarray:
    depth = np.squeeze(array).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be 2D after squeeze, got shape={tuple(depth.shape)}")

    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    valid = depth[finite]
    min_v = float(valid.min())
    max_v = float(valid.max())
    out = np.zeros(depth.shape, dtype=np.float32)
    if max_v > min_v:
        out[finite] = (depth[finite] - min_v) / (max_v - min_v)
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def convert_file(src_path: Path, dst_path: Path, *, overwrite: bool = False) -> bool:
    if dst_path.exists() and not overwrite:
        return False

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    png = depth_to_uint8(read_npz_array(src_path))
    Image.fromarray(png, mode="L").save(dst_path)
    return True


def iter_npz_files(src: Path):
    yield from sorted(src.rglob("*.npz"))


def convert_directory(src: Path | str, dst: Path | str, *, limit: int | None = None, overwrite: bool = False):
    src = Path(src)
    dst = Path(dst)
    written = []
    for index, src_path in enumerate(iter_npz_files(src)):
        if limit is not None and index >= limit:
            break
        rel_path = src_path.relative_to(src).with_suffix(".png")
        dst_path = dst / rel_path
        if convert_file(src_path, dst_path, overwrite=overwrite):
            written.append(dst_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="Source directory containing .npz depth files.")
    parser.add_argument("--dst", required=True, type=Path, help="Output directory for .png files.")
    parser.add_argument("--limit", type=int, default=None, help="Convert at most this many files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = convert_directory(args.src, args.dst, limit=args.limit, overwrite=args.overwrite)
    print(f"converted {len(written)} files to {args.dst}")


if __name__ == "__main__":
    main()
