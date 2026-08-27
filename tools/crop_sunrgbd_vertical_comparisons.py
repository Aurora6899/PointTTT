#!/usr/bin/env python3
"""Tightly crop saved SUN RGB-D panels and compose title-free vertical strips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PANELS = (
    "input_scene", "octformer", "3det_mamba", "pointttt", "ground_truth")


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--padding", type=int, default=24)
  parser.add_argument("--threshold", type=int, default=250)
  parser.add_argument(
      "--min-axis-pixels", type=int, default=24,
      help="Ignore rows/columns supported only by tiny isolated point splats.")
  parser.add_argument(
      "--min-component-pixels", type=int, default=1024,
      help="Ignore disconnected foreground components smaller than this area.")
  parser.add_argument("--dpi", type=int, default=1200)
  parser.add_argument("--output-name", default="comparison_vertical.png")
  parser.add_argument(
      "--overwrite-panels", action="store_true",
      help="Replace the five source PNGs with their tightly cropped versions.")
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
  panels = [
      add_white_padding(resize_to_width(image, content_width), args.padding)
      for image in foregrounds
  ]

  if args.overwrite_panels:
    for path, panel in zip(paths, panels):
      panel.save(path, dpi=(args.dpi, args.dpi))

  output = Image.new(
      "RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
  top = 0
  for panel in panels:
    output.paste(panel, (0, top))
    top += panel.height
  output_path = scene_dir / args.output_name
  output.save(output_path, dpi=(args.dpi, args.dpi))
  print("[saved] %s size=%dx%d" %
        (output_path, output.width, output.height))
  return True


def main():
  args = parse_args()
  if args.padding < 0:
    raise ValueError("--padding must be non-negative")
  if not 0 <= args.threshold <= 255:
    raise ValueError("--threshold must be in [0, 255]")
  if args.min_axis_pixels < 1:
    raise ValueError("--min-axis-pixels must be positive")
  if args.min_component_pixels < 1:
    raise ValueError("--min-component-pixels must be positive")
  root = args.root.resolve()
  scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
  processed = sum(process_scene(scene_dir, args) for scene_dir in scene_dirs)
  print("Processed %d scenes in %s" % (processed, root))


if __name__ == "__main__":
  main()
