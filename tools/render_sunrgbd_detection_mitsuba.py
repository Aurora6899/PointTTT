#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mitsuba/PointFlow-style rendering for saved SUN RGB-D detections.

This is the detection counterpart of
``shapenetpart_visual/render_shapenetpart_mitsuba.py``.  Every point is a
shaded 3-D sphere and every box edge is a shaded 3-D cylinder.  Input Scene,
OctFormer, 3DET-Mamba and PointTTT share the exact same point subset,
normalization, camera, lighting and crop for a fair paper figure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER_DIR = REPO_ROOT / "shapenetpart_visual"
if str(RENDERER_DIR) not in sys.path:
  sys.path.insert(0, str(RENDERER_DIR))
import render_pointcloud_mitsuba as renderer  # noqa: E402


DEFAULT_RESULTS = REPO_ROOT / "visual_results/sunrgbd_detection_comparison"
PANEL_SPECS = (
    ("input_scene", "Input Scene"),
    ("octformer", "OctFormer"),
    ("3det_mamba", "3DET-Mamba"),
    ("pointttt", "PointTTT"),
)
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
  parser.add_argument("--output-dir", type=Path, default=None)
  parser.add_argument(
      "--scenes", nargs="*", default=None,
      help="Scene directory names such as 2307_002308; default renders all.")
  parser.add_argument("--render-points", type=int, default=8192)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--radius", type=float, default=0.0065,
                      help="Sphere radius in normalized scene coordinates.")
  parser.add_argument("--box-radius", type=float, default=0.0035,
                      help="Box-cylinder radius in normalized coordinates.")
  parser.add_argument("--point-pastel", type=float, default=0.02)
  parser.add_argument("--box-pastel", type=float, default=0.0)
  parser.add_argument("--width", type=int, default=1600)
  parser.add_argument("--height", type=int, default=1200)
  parser.add_argument("--spp", type=int, default=128)
  parser.add_argument("--dpi", type=int, default=1200)
  parser.add_argument("--fov", type=float, default=34.0)
  parser.add_argument("--camera-origin", type=renderer.parse_vec3,
                      default=(1.05, -2.60, 1.25),
                      help="Default matches the original SUN RGB-D view.")
  parser.add_argument("--camera-target", type=renderer.parse_vec3,
                      default=(0.0, 0.0, 0.0))
  parser.add_argument("--environment", type=float, default=1.0)
  parser.add_argument("--light", type=float, default=5.0)
  parser.add_argument("--crop-padding", type=int, default=20)
  parser.add_argument("--title-height", type=int, default=70)
  parser.add_argument("--no-titles", action="store_true")
  parser.add_argument("--with-floor", action="store_true",
                      help="Keep the ShapeNetPart shadow floor; default is pure white.")
  parser.add_argument("--variant", choices=(
      "auto", "cuda_ad_rgb", "llvm_ad_rgb", "scalar_rgb"),
      default="scalar_rgb",
      help="Scalar is stable while both GPUs train; request CUDA explicitly.")
  parser.add_argument("--skip-existing", action="store_true")
  parser.add_argument("--recompose-existing", action="store_true",
                      help="Crop/compose saved panels without rerendering them.")
  parser.add_argument("--list-scenes", action="store_true")
  return parser.parse_args()


def load_ply_points(path: Path):
  points, colors = renderer.load_point_cloud(path)
  if colors is None:
    colors = np.full((len(points), 3), 0.68, dtype=np.float32)
  return points.astype(np.float32), colors.astype(np.float32)


def load_detection(path: Path):
  if not path.is_file():
    raise FileNotFoundError(path)
  with np.load(path, allow_pickle=False) as data:
    corners = np.asarray(data["corners"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int64)
  if corners.shape != (len(labels), 8, 3):
    raise ValueError("Invalid box corners in %s: %s" % (path, corners.shape))
  return corners, labels


def discover_scenes(results_root: Path, requested):
  if requested:
    scenes = [results_root / name for name in requested]
  else:
    manifest = results_root / "manifest.json"
    if manifest.is_file():
      with manifest.open("r") as handle:
        records = json.load(handle).get("scenes", [])
      scenes = [Path(record["directory"]) for record in records]
    else:
      scenes = sorted(path for path in results_root.iterdir()
                      if path.is_dir() and (path / "input_scene.ply").is_file())
  for scene in scenes:
    if not scene.is_absolute():
      scene = results_root / scene
    if not scene.is_dir():
      raise FileNotFoundError(scene)
  return [scene.resolve() for scene in scenes]


def shared_geometry(points, all_corners, render_points, seed):
  """Sample once and normalize points/boxes with one common transform."""
  rng = np.random.default_rng(seed)
  if render_points > 0 and len(points) > render_points:
    indices = np.sort(rng.choice(len(points), render_points, replace=False))
  else:
    indices = np.arange(len(points))
  sampled = points[indices]

  low, high = np.percentile(points, [0.25, 99.75], axis=0)
  valid = [corners.reshape(-1, 3) for corners in all_corners if corners.size]
  if valid:
    box_points = np.concatenate(valid, axis=0)
    low = np.minimum(low, box_points.min(axis=0))
    high = np.maximum(high, box_points.max(axis=0))
  center = (low + high) * 0.5
  scale = float(np.max(high - low))
  if not np.isfinite(scale) or scale <= 1.0e-8:
    raise ValueError("Degenerate SUN RGB-D scene geometry.")
  sampled = ((sampled - center) / scale).astype(np.float32)
  normalized_corners = [((corners - center) / scale).astype(np.float32)
                        for corners in all_corners]
  return sampled, indices, normalized_corners, center, scale


def material(color, rough=True):
  value = [float(x) for x in color]
  if not rough:
    return {"type": "diffuse", "reflectance": {
        "type": "rgb", "value": value}}
  return {
      "type": "roughplastic", "distribution": "ggx", "alpha": 0.16,
      "int_ior": 1.46,
      "diffuse_reflectance": {"type": "rgb", "value": value},
  }


def add_boxes(scene, corners, labels, palette, radius):
  """Append solid cylinders and corner spheres to a Mitsuba scene dict."""
  for box_index, (corner, label) in enumerate(zip(corners, labels)):
    color = palette[int(label)]
    bsdf = material(color)
    for edge_index, (begin, end) in enumerate(BOX_EDGES):
      p0, p1 = corner[begin], corner[end]
      scene["box_%03d_edge_%02d" % (box_index, edge_index)] = {
          "type": "cylinder",
          "p0": [float(x) for x in p0],
          "p1": [float(x) for x in p1],
          "radius": float(radius),
          "bsdf": bsdf,
      }
    for corner_index, point in enumerate(corner):
      scene["box_%03d_joint_%02d" % (box_index, corner_index)] = {
          "type": "sphere", "center": [float(x) for x in point],
          "radius": float(radius * 1.18), "bsdf": bsdf,
      }


def render_panel(mi, output, points, colors, corners, labels, palette, args):
  floor_z = float(points[:, 2].min() - args.radius * 1.2)
  scene_dict = renderer.build_scene(
      mi=mi, points=points, colors=colors, radius=args.radius,
      width=args.width, height=args.height, spp=args.spp, fov=args.fov,
      camera_origin=args.camera_origin, camera_target=args.camera_target,
      floor_z=floor_z, floor_size=10.0,
      environment_radiance=args.environment, light_radiance=args.light,
      sphere_material="roughplastic")
  if not args.with_floor:
    scene_dict.pop("floor", None)
    scene_dict["environment"]["radiance"]["value"] = [1.0, 1.0, 1.0]
  add_boxes(scene_dict, corners, labels, palette, args.box_radius)
  print("[render] %s points=%d boxes=%d %dx%d spp=%d" %
        (output.name, len(points), len(labels), args.width, args.height,
         args.spp))
  scene = mi.load_dict(scene_dict)
  image = mi.render(scene, spp=args.spp)
  output.parent.mkdir(parents=True, exist_ok=True)
  renderer.write_png_600dpi(mi, image, output, dpi=args.dpi)
  # Thousands of analytic spheres create a sizeable Dr.Jit scene graph.
  # Release it between panels instead of retaining five scenes on the GPU.
  del image, scene, scene_dict
  gc.collect()
  try:
    import drjit as dr
    dr.sync_thread()
  except (ImportError, AttributeError):
    pass


def crop_consistently(paths, padding, dpi):
  images, union = [], None
  for path in paths:
    with Image.open(path) as image:
      rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.any(rgb < 254, axis=2)
    union = mask if union is None else np.logical_or(union, mask)
    images.append(rgb)
  rows, columns = np.nonzero(union)
  if rows.size == 0:
    raise RuntimeError("No Mitsuba foreground pixels found.")
  height, width = union.shape
  left = max(0, int(columns.min()) - padding)
  right = min(width, int(columns.max()) + 1 + padding)
  top = max(0, int(rows.min()) - padding)
  bottom = min(height, int(rows.max()) + 1 + padding)
  for path, rgb in zip(paths, images):
    cropped = rgb[top:bottom, left:right].copy()
    cropped[np.all(cropped >= 254, axis=2)] = 255
    Image.fromarray(cropped, mode="RGB").save(path, dpi=(dpi, dpi))


def compose(paths, titles, output, title_height, dpi, show_titles):
  panels = []
  for path in paths:
    with Image.open(path) as image:
      panels.append(image.convert("RGB").copy())
  title_space = title_height if show_titles else 0
  canvas = Image.new(
      "RGB", (sum(panel.width for panel in panels),
              max(panel.height for panel in panels) + title_space), "white")
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", max(14, title_height // 2))
  except OSError:
    font = ImageFont.load_default()
  left = 0
  for title, panel in zip(titles, panels):
    if show_titles:
      box = draw.textbbox((0, 0), title, font=font)
      text_width = box[2] - box[0]
      draw.text((left + (panel.width - text_width) / 2, title_height * 0.2),
                title, fill="black", font=font)
    canvas.paste(panel, (left, title_space))
    left += panel.width
  canvas.save(output, dpi=(dpi, dpi))


def write_manifest(args, records):
  payload = {
      "renderer": "Mitsuba 3 / ShapeNetPart PointFlow sphere renderer",
      "pure_white_background": not args.with_floor,
      "sphere_radius": args.radius, "box_radius": args.box_radius,
      "spp": args.spp, "resolution": [args.width, args.height],
      "scenes": records,
  }
  args.output_dir.mkdir(parents=True, exist_ok=True)
  # Workers only own their scene manifest. The parent process writes the
  # aggregate manifest after every isolated scene succeeds.
  if os.environ.get("SUNRGBD_MITSUBA_WORKER") != "1":
    with (args.output_dir / "manifest.json").open("w") as handle:
      json.dump(payload, handle, indent=2)
  for record in records:
    scene_manifest = args.output_dir / record["scene"] / "render_manifest.json"
    with scene_manifest.open("w") as handle:
      json.dump(record, handle, indent=2)


def main():
  args = parse_args()
  args.results_root = args.results_root.resolve()
  args.output_dir = ((args.results_root / "mitsuba") if args.output_dir is None
                     else args.output_dir.resolve())
  scenes = discover_scenes(args.results_root, args.scenes)
  for scene in scenes:
    print(scene.name)
  if args.list_scenes:
    return
  if not scenes:
    raise ValueError("No saved SUN RGB-D visualization scenes found.")

  # Mitsuba/Dr.Jit retains compiled scene state across renders. Isolate each
  # large indoor scene in its own process so a multi-scene paper render has
  # bounded memory and remains resumable.
  if len(scenes) > 1 and os.environ.get("SUNRGBD_MITSUBA_WORKER") != "1":
    records = []
    for position, scene in enumerate(scenes, start=1):
      print("\n[isolated scene %d/%d] %s" %
            (position, len(scenes), scene.name), flush=True)
      command = [sys.executable, str(Path(__file__).resolve())]
      command.extend(sys.argv[1:])
      command.extend(["--scenes", scene.name])
      environment = os.environ.copy()
      environment["SUNRGBD_MITSUBA_WORKER"] = "1"
      subprocess.run(command, cwd=str(REPO_ROOT), env=environment, check=True)
      scene_manifest = args.output_dir / scene.name / "render_manifest.json"
      with scene_manifest.open("r") as handle:
        records.append(json.load(handle))
    write_manifest(args, records)
    print("\nSaved Mitsuba SUN RGB-D figures to %s" % args.output_dir)
    return

  palette_path = args.results_root / "class_palette.json"
  with palette_path.open("r") as handle:
    palette_dict = json.load(handle)
  palette = np.asarray(list(palette_dict.values()), dtype=np.float32) / 255.0
  palette = np.clip((1.0 - args.box_pastel) * palette + args.box_pastel,
                    0.001, 0.98)
  mi = renderer.select_mitsuba_variant(args.variant)
  records = []

  for position, scene in enumerate(scenes, start=1):
    print("\n[%d/%d] %s" % (position, len(scenes), scene.name))
    raw_points, raw_colors = load_ply_points(scene / "input_scene.ply")
    detections = {}
    all_corners = []
    for key in ("ground_truth", "octformer", "3det_mamba", "pointttt"):
      detections[key] = load_detection(scene / (key + ".npz"))
      all_corners.append(detections[key][0])
    points, indices, normalized, center, scale = shared_geometry(
        raw_points, all_corners, args.render_points, args.seed + position)
    colors = raw_colors[indices]
    colors = np.clip(
        (1.0 - args.point_pastel) * colors + args.point_pastel, 0.001, 0.98)
    corner_map = dict(zip(
        ("ground_truth", "octformer", "3det_mamba", "pointttt"),
        normalized))

    rendered = []
    rendered_now = []
    for key, _ in PANEL_SPECS:
      output = args.output_dir / scene.name / (key + ".png")
      if args.skip_existing and output.is_file():
        print("[skip] %s" % output)
      else:
        if key == "input_scene":
          corners = np.empty((0, 8, 3), dtype=np.float32)
          labels = np.empty(0, dtype=np.int64)
        else:
          corners = corner_map[key]
          labels = detections[key][1]
        render_panel(mi, output, points, colors, corners, labels,
                     palette, args)
        rendered_now.append(output)
      rendered.append(output)

    gt_output = args.output_dir / scene.name / "ground_truth.png"
    if not (args.skip_existing and gt_output.is_file()):
      render_panel(mi, gt_output, points, colors, corner_map["ground_truth"],
                   detections["ground_truth"][1], palette, args)
      rendered_now.append(gt_output)
    if rendered_now:
      # Crop only the panels rendered in this invocation. Existing panels may
      # already be cropped and therefore no longer share the film dimensions.
      crop_consistently(rendered_now, args.crop_padding, args.dpi)
    comparison = args.output_dir / scene.name / "comparison.png"
    panel_sizes = []
    for path in rendered:
      with Image.open(path) as image:
        panel_sizes.append(image.size)
    if len(set(panel_sizes)) == 1:
      compose(rendered, [title for _, title in PANEL_SPECS], comparison,
              args.title_height, args.dpi, not args.no_titles)
    elif args.recompose_existing:
      raise ValueError(
          "Existing panels have different crop sizes. Rerun this scene "
          "without --skip-existing to regenerate a shared crop: %s" % scene)
    else:
      print("[keep] comparison unchanged; saved panels have mixed crop sizes")
    records.append({
        "scene": scene.name, "rendered_points": int(len(points)),
        "normalization_center": center.tolist(), "normalization_scale": scale,
        "comparison": str(comparison), "ground_truth": str(gt_output),
    })

  write_manifest(args, records)
  print("\nSaved Mitsuba SUN RGB-D figures to %s" % args.output_dir)


if __name__ == "__main__":
  main()
