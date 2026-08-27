#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense SUN RGB-D detection visualization with Open3D point splats.

Unlike the ShapeNetPart/Mitsuba renderer, this renderer keeps all 100,000
XYZRGB points by default. It produces paper-style rows for Input, three saved
prediction sets and Ground Truth, with one shared camera per scene.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


os.environ.setdefault("EGL_PLATFORM", "surfaceless")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "visual_results/sunrgbd_detection_comparison"
PANELS = (
    ("input_scene", "Input"),
    ("octformer", "OctFormer"),
    ("3det_mamba", "3DET-Mamba"),
    ("pointttt", "PointTTT (Ours)"),
    ("ground_truth", "GT"),
)
BOX_EDGES = np.asarray((
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)), dtype=np.int32)


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument(
      "--scenes", nargs="*", default=None,
      help="Saved scene directory names; default uses every manifest scene.")
  parser.add_argument("--max-points", type=int, default=0,
                      help="0 keeps all points; positive values subsample once.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--width", type=int, default=1600)
  parser.add_argument("--height", type=int, default=1200)
  parser.add_argument("--dpi", type=int, default=1200)
  parser.add_argument("--point-size", type=float, default=2.0)
  parser.add_argument("--line-width", type=float, default=3.0)
  parser.add_argument("--fov", type=float, default=50.0)
  parser.add_argument("--view-direction", type=float, nargs=3,
                      default=(0.34, -0.85, 0.41))
  parser.add_argument("--contrast", type=float, default=1.12)
  parser.add_argument("--prediction-color", type=float, nargs=3,
                      default=(1.0, 0.52, 0.0))
  parser.add_argument("--gt-color", type=float, nargs=3,
                      default=(0.05, 0.72, 0.12))
  parser.add_argument("--row-label-width", type=int, default=None)
  parser.add_argument("--crop-padding", type=int, default=24)
  parser.add_argument("--column-gap", type=int, default=None)
  parser.add_argument("--row-gap", type=int, default=None)
  parser.add_argument("--columns-per-page", type=int, default=5)
  parser.add_argument("--overview-panel-width", type=int, default=900,
                      help="Only downsizes overview pages; individual panels stay full resolution.")
  parser.add_argument("--with-row-labels", action="store_true",
                      help="Add method names; final paper pages are label-free by default.")
  parser.add_argument("--skip-existing", action="store_true")
  parser.add_argument("--list-scenes", action="store_true")
  return parser.parse_args()


def discover_scenes(root: Path, requested):
  if requested:
    paths = [root / name for name in requested]
  else:
    with (root / "manifest.json").open("r") as handle:
      records = json.load(handle)["scenes"]
    paths = [Path(record["directory"]) for record in records]
  paths = [path.resolve() for path in paths]
  for path in paths:
    if not path.is_dir():
      raise FileNotFoundError(path)
  return paths


def load_points(path: Path):
  from plyfile import PlyData
  vertex = PlyData.read(str(path))["vertex"].data
  xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
      np.float64)
  rgb = np.column_stack(
      (vertex["red"], vertex["green"], vertex["blue"])).astype(np.float64)
  rgb /= 255.0
  return xyz, rgb


def load_boxes(path: Path):
  with np.load(path, allow_pickle=False) as archive:
    corners = np.asarray(archive["corners"], dtype=np.float64)
  if corners.ndim != 3 or corners.shape[1:] != (8, 3):
    raise ValueError("Invalid corners in %s: %s" % (path, corners.shape))
  return corners


def choose_points(xyz, rgb, maximum, seed):
  if maximum <= 0 or len(xyz) <= maximum:
    return xyz, rgb
  rng = np.random.default_rng(seed)
  indices = np.sort(rng.choice(len(xyz), maximum, replace=False))
  return xyz[indices], rgb[indices]


def make_point_cloud(o3d, xyz, rgb, contrast):
  colors = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)
  point_cloud = o3d.geometry.PointCloud()
  point_cloud.points = o3d.utility.Vector3dVector(xyz)
  point_cloud.colors = o3d.utility.Vector3dVector(colors)
  return point_cloud


def make_box_lines(o3d, corners, color):
  line_set = o3d.geometry.LineSet()
  if not len(corners):
    line_set.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
    line_set.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
    return line_set
  points = corners.reshape(-1, 3)
  lines = np.concatenate(
      [BOX_EDGES + box_index * 8 for box_index in range(len(corners))], axis=0)
  colors = np.repeat(np.asarray(color, dtype=np.float64)[None], len(lines), axis=0)
  line_set.points = o3d.utility.Vector3dVector(points)
  line_set.lines = o3d.utility.Vector2iVector(lines)
  line_set.colors = o3d.utility.Vector3dVector(colors)
  return line_set


def camera_from_points(xyz, direction, fov):
  low, high = np.percentile(xyz, [0.25, 99.75], axis=0)
  target = (low + high) * 0.5
  radius = float(np.linalg.norm((high - low) * 0.5))
  direction = np.asarray(direction, dtype=np.float64)
  direction /= max(np.linalg.norm(direction), 1.0e-12)
  distance = radius / max(np.sin(np.deg2rad(fov * 0.5)), 1.0e-3) * 1.08
  eye = target + direction * distance
  return target, eye


def write_image(o3d, image, path: Path, dpi):
  path.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_image(str(path), image, quality=9)
  with Image.open(path) as png:
    rgb = np.asarray(png.convert("RGB"), dtype=np.uint8).copy()
  rgb[np.all(rgb >= 253, axis=2)] = 255
  Image.fromarray(rgb).save(path, dpi=(dpi, dpi))


def crop_consistently(paths, padding, dpi):
  """Apply the ShapeNetPart-style union crop without adding a new canvas."""
  images, union = [], None
  for path in paths:
    with Image.open(path) as image:
      rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.any(rgb < 250, axis=2)
    union = mask if union is None else np.logical_or(union, mask)
    images.append(rgb)
  rows, columns = np.nonzero(union)
  if rows.size == 0:
    raise RuntimeError("Could not find Open3D point-cloud foreground.")
  left, right = int(columns.min()), int(columns.max()) + 1
  top, bottom = int(rows.min()), int(rows.max()) + 1
  left = max(0, left - padding)
  right = min(images[0].shape[1], right + padding)
  top = max(0, top - padding)
  bottom = min(images[0].shape[0], bottom + padding)
  for path, rgb in zip(paths, images):
    cropped = rgb[top:bottom, left:right].copy()
    cropped[np.all(cropped >= 253, axis=2)] = 255
    Image.fromarray(cropped, mode="RGB").save(path, dpi=(dpi, dpi))


def render_scene_panels(o3d, scene_dir, output_dir, args, scene_index):
  xyz, rgb = load_points(scene_dir / "input_scene.ply")
  input_count = len(xyz)
  xyz, rgb = choose_points(xyz, rgb, args.max_points,
                           args.seed + scene_index)
  point_cloud = make_point_cloud(o3d, xyz, rgb, args.contrast)
  boxes = {key: load_boxes(scene_dir / (key + ".npz"))
           for key, _ in PANELS if key != "input_scene"}
  target, eye = camera_from_points(xyz, args.view_direction, args.fov)

  renderer = o3d.visualization.rendering.OffscreenRenderer(
      args.width, args.height)
  renderer.scene.set_background(np.asarray([1.0, 1.0, 1.0, 1.0]))
  renderer.scene.scene.enable_sun_light(False)
  renderer.scene.view.set_post_processing(False)
  point_material = o3d.visualization.rendering.MaterialRecord()
  point_material.shader = "defaultUnlit"
  point_material.point_size = float(args.point_size)
  line_material = o3d.visualization.rendering.MaterialRecord()
  line_material.shader = "unlitLine"
  line_material.line_width = float(args.line_width)
  renderer.scene.add_geometry("points", point_cloud, point_material)
  renderer.setup_camera(float(args.fov), target, eye, np.asarray([0.0, 0.0, 1.0]))

  files = {}
  for key, _ in PANELS:
    path = output_dir / scene_dir.name / (key + ".png")
    files[key] = path
    if args.skip_existing and path.is_file():
      continue
    if renderer.scene.has_geometry("boxes"):
      renderer.scene.remove_geometry("boxes")
    if key != "input_scene":
      color = args.gt_color if key == "ground_truth" else args.prediction_color
      line_set = make_box_lines(o3d, boxes[key], color)
      renderer.scene.add_geometry("boxes", line_set, line_material)
    image = renderer.render_to_image()
    write_image(o3d, image, path, args.dpi)
    print("[saved] %s points=%d boxes=%d" %
          (path, len(xyz), 0 if key == "input_scene" else len(boxes[key])))

  crop_consistently(list(files.values()), args.crop_padding, args.dpi)

  del renderer
  return {
      "scene": scene_dir.name,
      "input_points": input_count,
      "rendered_points": len(xyz),
      "camera_target": target.tolist(),
      "camera_eye": eye.tolist(),
      "files": {key: str(path) for key, path in files.items()},
  }


def compose_grid_page(records, output: Path, args):
  rows = []
  for key, title in PANELS:
    images = []
    for record in records:
      with Image.open(record["files"][key]) as image:
        panel = image.convert("RGB").copy()
      if 0 < args.overview_panel_width < panel.width:
        resized_height = max(
            1, int(round(panel.height * args.overview_panel_width / panel.width)))
        panel = panel.resize(
            (args.overview_panel_width, resized_height), Image.Resampling.LANCZOS)
      images.append(panel)
    rows.append((title, images))
  scale = args.width / 1600.0
  label_width = 0
  if args.with_row_labels:
    label_width = (args.row_label_width if args.row_label_width is not None
                   else int(round(430 * scale)))
  column_gap = (args.column_gap if args.column_gap is not None
                else int(round(16 * scale)))
  row_gap = (args.row_gap if args.row_gap is not None
             else int(round(10 * scale)))
  column_widths = [max(rows[row][1][column].width for row in range(len(rows)))
                   for column in range(len(records))]
  row_heights = [max(image.height for image in images) for _, images in rows]
  canvas_width = (label_width + sum(column_widths) +
                  max(0, len(records) - 1) * column_gap)
  canvas_height = sum(row_heights) + max(0, len(rows) - 1) * row_gap
  canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
  draw = ImageDraw.Draw(canvas) if args.with_row_labels else None
  font = None
  if args.with_row_labels:
    try:
      font = ImageFont.truetype(
          "DejaVuSans-Bold.ttf", max(16, int(round(42 * scale))))
    except OSError:
      font = ImageFont.load_default()
  top = 0
  for row_index, (title, images) in enumerate(rows):
    if args.with_row_labels:
      box = draw.textbbox((0, 0), title, font=font)
      text_height = box[3] - box[1]
      draw.text((int(round(18 * scale)),
                 top + (row_heights[row_index] - text_height) / 2), title,
                fill="black", font=font)
    left = label_width
    for column, image in enumerate(images):
      x = left + (column_widths[column] - image.width) // 2
      y = top + (row_heights[row_index] - image.height) // 2
      canvas.paste(image, (x, y))
      left += column_widths[column] + column_gap
    top += row_heights[row_index] + row_gap
  output.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output, dpi=(args.dpi, args.dpi))


def compose_grid(records, output_dir: Path, args):
  per_page = max(1, args.columns_per_page)
  pages = []
  for page_index, start in enumerate(range(0, len(records), per_page), start=1):
    page_records = records[start:start + per_page]
    suffix = "" if len(records) <= per_page else "_page_%02d" % page_index
    output = output_dir / ("comparison_grid%s.png" % suffix)
    compose_grid_page(page_records, output, args)
    pages.append(str(output))
  if len(pages) > 1:
    # Keep a predictable convenience filename pointing to the first page.
    with Image.open(pages[0]) as first_page:
      first_page.convert("RGB").save(
          output_dir / "comparison_grid.png", dpi=(args.dpi, args.dpi))
  return pages


def main():
  args = parse_args()
  args.results_root = args.results_root.resolve()
  args.output_dir = ((args.results_root / "open3d_dense")
                     if args.output_dir is None else args.output_dir.resolve())
  scenes = discover_scenes(args.results_root, args.scenes)
  for path in scenes:
    print(path.name)
  if args.list_scenes:
    return
  import open3d as o3d
  records = [render_scene_panels(o3d, path, args.output_dir, args, index)
             for index, path in enumerate(scenes)]
  pages = compose_grid(records, args.output_dir, args)
  with (args.output_dir / "manifest.json").open("w") as handle:
    json.dump({
        "renderer": "Open3D dense point splats",
        "prediction_color": list(args.prediction_color),
        "ground_truth_color": list(args.gt_color),
        "point_size": args.point_size,
        "resolution": [args.width, args.height],
        "comparison_pages": pages,
        "scenes": records,
    }, handle, indent=2)
  print("Saved dense SUN RGB-D figures to %s" % args.output_dir)


if __name__ == "__main__":
  main()
