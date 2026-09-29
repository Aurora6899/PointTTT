#!/usr/bin/env python3
"""Tightly crop saved SUN RGB-D panels and compose title-free vertical strips."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PANELS = ("ground_truth", "octformer", "3det_mamba", "pointttt")


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument(
      "--scene", default=None,
      help="Process only this scene directory name (useful for large images).")
  parser.add_argument("--padding", type=int, default=24)
  parser.add_argument("--threshold", type=int, default=250)
  parser.add_argument(
      "--min-axis-pixels", type=int, default=24,
      help="Ignore rows/columns supported only by tiny isolated point splats.")
  parser.add_argument(
      "--min-component-pixels", type=int, default=1024,
      help="Ignore disconnected foreground components smaller than this area.")
  parser.add_argument("--dpi", type=int, default=1200)
  parser.add_argument(
      "--scale", type=float, default=1.0,
      help="Scale output width, height, and padding by this factor.")
  parser.add_argument("--output-name", default="comparison_vertical.png")
  parser.add_argument(
      "--overwrite-panels", action="store_true",
      help="Replace the four source PNGs with their tightly cropped versions.")
  return parser.parse_args()


def foreground_crop(path: Path, threshold: int, min_axis_pixels: int,
                    min_component_pixels: int):
  with Image.open(path) as image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
  mask = np.any(rgb < threshold, axis=2)
  if not np.any(mask):
    raise RuntimeError("No non-white foreground found in %s" % path)

  count, components, stats, _ = cv2.connectedComponentsWithStats(
      mask.astype(np.uint8), connectivity=8)
  keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_component_pixels) + 1
  robust_mask = np.isin(components, keep) if keep.size else mask
  supported_columns = np.flatnonzero(
      robust_mask.sum(axis=0) >= min_axis_pixels)
  supported_rows = np.flatnonzero(robust_mask.sum(axis=1) >= min_axis_pixels)
  if supported_columns.size == 0 or supported_rows.size == 0:
    rows, columns = np.nonzero(mask)
    supported_columns, supported_rows = columns, rows
  left, right = int(supported_columns.min()), int(supported_columns.max()) + 1
  top, bottom = int(supported_rows.min()), int(supported_rows.max()) + 1
  return Image.fromarray(rgb[top:bottom, left:right].copy(), mode="RGB")


def resize_to_width(image: Image.Image, width: int):
  if image.width == width:
    return image
  height = max(1, int(round(image.height * width / image.width)))
  return image.resize((width, height), Image.Resampling.LANCZOS)


def add_white_padding(image: Image.Image, padding: int):
  canvas = Image.new(
      "RGB", (image.width + 2 * padding, image.height + 2 * padding), "white")
  canvas.paste(image, (padding, padding))
  return canvas


def process_scene(scene_dir: Path, args):
  paths = [scene_dir / (name + ".png") for name in PANELS]
  if not all(path.is_file() for path in paths):
    return False

  foregrounds = [
      foreground_crop(path, args.threshold, args.min_axis_pixels,
                      args.min_component_pixels)
      for path in paths
  ]
  # Equal content widths allow direct vertical concatenation without adding
  # variable left/right blank canvases to narrower panels.
  content_width = max(image.width for image in foregrounds)
  target_width = max(1, int(round(content_width * args.scale)))
  target_padding = max(0, int(round(args.padding * args.scale)))
  target_heights = [
      max(1, int(round(image.height * target_width / image.width)))
      for image in foregrounds
  ]
  output = Image.new(
      "RGB",
      (target_width + 2 * target_padding,
       sum(height + 2 * target_padding for height in target_heights)),
      "white")
  top = 0
  for path, foreground, target_height in zip(
      paths, foregrounds, target_heights):
    resized = foreground.resize(
        (target_width, target_height), Image.Resampling.LANCZOS)
    panel = add_white_padding(resized, target_padding)
    output.paste(panel, (0, top))
    top += panel.height
    if args.overwrite_panels:
      panel.save(path, dpi=(args.dpi, args.dpi))
    resized.close()
    panel.close()
  output_path = scene_dir / args.output_name
  output.save(output_path, dpi=(args.dpi, args.dpi))
  print("[saved] %s size=%dx%d" %
        (output_path, output.width, output.height))
  output.close()
  for foreground in foregrounds:
    foreground.close()
  gc.collect()
  return True


def main():
  args = parse_args()
  if args.padding < 0:
    raise ValueError("--padding must be non-negative")
  if args.scale <= 0:
    raise ValueError("--scale must be positive")
  if not 0 <= args.threshold <= 255:
    raise ValueError("--threshold must be in [0, 255]")
  if args.min_axis_pixels < 1:
    raise ValueError("--min-axis-pixels must be positive")
  if args.min_component_pixels < 1:
    raise ValueError("--min-component-pixels must be positive")
  root = args.root.resolve()
  if args.scene is not None:
    scene_dir = root / args.scene
    if not scene_dir.is_dir():
      raise FileNotFoundError(scene_dir)
    scene_dirs = [scene_dir]
  else:
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
  processed = sum(process_scene(scene_dir, args) for scene_dir in scene_dirs)
  print("Processed %d scenes in %s" % (processed, root))


if __name__ == "__main__":
  main()
