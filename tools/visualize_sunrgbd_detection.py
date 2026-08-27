#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and visualize SUN RGB-D detection checkpoints fairly.

The script uses the same raw 100,000-point validation scene for every
checkpoint, then renders publication-style RGB point clouds and class-colored
oriented 3-D boxes from an identical orthographic camera.  The four-panel
comparison is: Input Scene | OctFormer | 3DET-Mamba | PointTTT.

The checkpoint names are user-facing aliases. Their actual architecture is
recorded in ``manifest.json`` from checkpoint metadata instead of inferred
from the alias.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


CLASS_NAMES = (
    "bed", "table", "sofa", "chair", "toilet", "desk", "dresser",
    "night_stand", "bookshelf", "bathtub")

# Stable high-contrast colors shared by GT and all detectors.
CLASS_COLORS = np.asarray([
    (230, 25, 75),    # bed
    (60, 180, 75),    # table
    (0, 130, 200),    # sofa
    (245, 130, 48),   # chair
    (145, 30, 180),   # toilet
    (70, 240, 240),   # desk
    (240, 50, 230),   # dresser
    (210, 245, 60),   # night stand
    (250, 190, 212),  # bookshelf
    (0, 128, 128),    # bathtub
], dtype=np.uint8)

MODEL_SPECS = (
    ("octformer", "OctFormer", "work_dirs/pointttt_sunrgbd/epoch_13.pth"),
    ("3det_mamba", "3DET-Mamba", "work_dirs/pointttt_sunrgbd/epoch_12.pth"),
    ("pointttt", "PointTTT", "work_dirs/pointttt_sunrgbd/epoch_18.pth"),
)

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--config", default="configs/det_sunrgbd.py")
  parser.add_argument("--data-root", default="data/sunrgbd")
  parser.add_argument("--ann-file", default="sunrgbd_infos_val.pkl")
  parser.add_argument(
      "--output", default="visual_results/sunrgbd_detection_comparison")
  parser.add_argument("--octformer", default=MODEL_SPECS[0][2])
  parser.add_argument("--3det-mamba", dest="det_mamba",
                      default=MODEL_SPECS[1][2])
  parser.add_argument("--pointttt", default=MODEL_SPECS[2][2])
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument(
      "--indices", type=int, nargs="*", default=None,
      help="Zero-based indices in sunrgbd_infos_val.pkl.")
  parser.add_argument(
      "--num-scenes", type=int, default=6,
      help="Number of representative scenes selected when --indices is absent.")
  parser.add_argument("--list-scenes", action="store_true",
                      help="List selected scenes without loading a model.")
  parser.add_argument("--score-thr", type=float, default=0.15)
  parser.add_argument("--max-boxes", type=int, default=64)
  parser.add_argument("--render-points", type=int, default=50000,
                      help="Points drawn in PNG; inference always uses all points.")
  parser.add_argument("--width", type=int, default=1200)
  parser.add_argument("--height", type=int, default=900)
  parser.add_argument("--dpi", type=int, default=300)
  parser.add_argument("--elev", type=float, default=24.0)
  parser.add_argument("--azim", type=float, default=-68.0)
  parser.add_argument("--camera-zoom", type=float, default=1.65)
  parser.add_argument("--point-size", type=float, default=0.35)
  parser.add_argument("--line-width", type=float, default=1.8)
  parser.add_argument("--show-labels", action="store_true")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--skip-existing", action="store_true")
  return parser.parse_args()


def patch_detection_environment():
  # Reuse the compatibility bridges used by the training launcher.
  import detection
  detection._find_mmdet3d()
  detection._patch_legacy_mmseg_api()
  detection._patch_mmcv_ddp_api()
  from models import PointTTTdet  # noqa: F401


def load_infos(path: Path):
  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open("rb") as handle:
    infos = pickle.load(handle)
  if not isinstance(infos, list) or not infos:
    raise ValueError("SUN RGB-D annotation file must contain a non-empty list.")
  return infos


def valid_gt(info):
  ann = info.get("annos", {})
  boxes = np.asarray(ann.get("gt_boxes_upright_depth", []), dtype=np.float32)
  labels = np.asarray(ann.get("class", []), dtype=np.int64)
  if boxes.size == 0:
    boxes = np.empty((0, 7), dtype=np.float32)
  boxes = boxes.reshape(-1, 7)
  labels = labels.reshape(-1)
  valid = np.logical_and(labels >= 0, labels < len(CLASS_NAMES))
  return boxes[valid], labels[valid]


def choose_representative_scenes(infos, number):
  """Greedy class-coverage selection with deterministic richness tie-breaks."""
  number = min(max(1, number), len(infos))
  candidates = []
  for index, info in enumerate(infos):
    _, labels = valid_gt(info)
    unique = set(labels.tolist())
    candidates.append((index, unique, len(labels)))

  selected, covered = [], set()
  remaining = set(range(len(infos)))
  while remaining and len(selected) < number:
    best = max(
        remaining,
        key=lambda i: (
            len(candidates[i][1] - covered),
            len(candidates[i][1]), candidates[i][2], -i))
    selected.append(best)
    covered.update(candidates[best][1])
    remaining.remove(best)
  return selected


def checkpoint_description(path: Path):
  checkpoint = torch.load(str(path), map_location="cpu")
  if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
    raise TypeError("Unsupported MMDetection3D checkpoint: %s" % path)
  meta = checkpoint.get("meta", {})
  config_text = str(meta.get("config", ""))
  state = checkpoint["state_dict"]
  first_key = next(iter(state), "")
  detector_type = "unknown"
  for marker in (
      "PointTTTSingleStage3DDetector", "OctFormer", "3DET-Mamba", "TR3D"):
    if marker in config_text:
      detector_type = marker
      break
  result = {
      "path": str(path.resolve()),
      "epoch": int(meta.get("epoch", -1)),
      "state_dict_keys": len(state),
      "first_state_key": first_key,
      "detector_type_from_metadata": detector_type,
      "mmdet3d_version": str(meta.get("mmdet3d_version", "")),
  }
  del checkpoint
  return result


def build_model(config_path: Path, checkpoint_path: Path, device):
  import mmcv
  from mmdet3d.models import build_model

  cfg = mmcv.Config.fromfile(str(config_path))
  cfg.model.pretrained = None
  cfg.model.train_cfg = None
  # PointTTT construction prints every per-block TTT config and MMCV init
  # diagnostic. Keep visualization logs focused on scenes and checkpoints.
  with contextlib.redirect_stdout(io.StringIO()):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
  checkpoint = load_checkpoint_into(model, checkpoint_path)
  model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", CLASS_NAMES)
  model.cfg = cfg
  model.to(device).eval()
  return model


def load_checkpoint_into(model, path: Path):
  checkpoint = torch.load(str(path), map_location="cpu")
  if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
    raise TypeError("Unsupported MMDetection3D checkpoint: %s" % path)
  state = checkpoint["state_dict"]
  model_keys = set(model.state_dict())
  extra = set(state) - model_keys
  allowed_suffixes = (".token_idx", ".rotary_emb.inv_freq")
  disallowed = sorted(
      key for key in extra if not key.endswith(allowed_suffixes))
  if disallowed:
    raise RuntimeError(
        "Checkpoint has unexpected learned/state keys: %s" % disallowed[:10])
  filtered = {key: value for key, value in state.items() if key in model_keys}
  missing, unexpected = model.load_state_dict(filtered, strict=False)
  if missing or unexpected:
    raise RuntimeError(
        "Checkpoint does not match inference model: missing=%s unexpected=%s" %
        (missing, unexpected))
  print("Loaded %s (filtered %d non-persistent TTT cache tensors)" %
        (path, len(extra)))
  model.eval()
  return checkpoint


def load_points(data_root: Path, info):
  point_path = data_root / info["pts_path"]
  if not point_path.is_file():
    raise FileNotFoundError(point_path)
  features = int(info.get("point_cloud", {}).get("num_features", 6))
  points = np.fromfile(point_path, dtype=np.float32)
  if points.size % features:
    raise ValueError("Invalid SUN RGB-D point file: %s" % point_path)
  points = points.reshape(-1, features)
  if points.shape[1] < 6:
    raise ValueError("Visualization requires XYZRGB SUN RGB-D points.")
  return np.ascontiguousarray(points[:, :6]), point_path


def predict(model, points, device, score_thr, max_boxes):
  from mmdet3d.core.bbox import get_box_type

  tensor = torch.from_numpy(points).to(device, non_blocking=True)
  # FCAF3D constructs the final box object through this callable. The normal
  # MMDetection3D pipeline inserts it in Collect3D; this standalone entry must
  # provide the same metadata explicitly.
  box_type_3d, box_mode_3d = get_box_type("Depth")
  img_meta = dict(box_type_3d=box_type_3d, box_mode_3d=box_mode_3d)
  with torch.inference_mode():
    result = model.simple_test([tensor], [img_meta])[0]
  boxes_obj = result["boxes_3d"]
  scores = result["scores_3d"].detach().cpu().numpy()
  labels = result["labels_3d"].detach().cpu().numpy().astype(np.int64)
  corners = boxes_obj.corners.detach().cpu().numpy()
  box_tensor = boxes_obj.tensor.detach().cpu().numpy()
  keep = np.flatnonzero(scores >= score_thr)
  if max_boxes > 0 and keep.size > max_boxes:
    keep = keep[np.argsort(scores[keep])[::-1][:max_boxes]]
  keep = keep[np.argsort(scores[keep])[::-1]]
  return {
      "boxes": box_tensor[keep].astype(np.float32),
      "corners": corners[keep].astype(np.float32),
      "scores": scores[keep].astype(np.float32),
      "labels": labels[keep].astype(np.int64),
  }


def box_corners_numpy(boxes):
  if len(boxes) == 0:
    return np.empty((0, 8, 3), dtype=np.float32)
  from mmdet3d.core.bbox import DepthInstance3DBoxes
  return DepthInstance3DBoxes(
      torch.from_numpy(np.asarray(boxes, dtype=np.float32)),
      box_dim=7, with_yaw=True).corners.numpy()


def render_indices(num_points, keep, seed):
  if keep <= 0 or num_points <= keep:
    return np.arange(num_points)
  rng = np.random.default_rng(seed)
  return np.sort(rng.choice(num_points, size=keep, replace=False))


def scene_limits(xyz, box_corner_sets=()):
  low, high = np.percentile(xyz, [0.25, 99.75], axis=0)
  # Keep every GT/predicted box visible. All panels use this single union, so
  # no detector benefits from a different crop or apparent box scale.
  valid_sets = [corners.reshape(-1, 3) for corners in box_corner_sets
                if corners.size]
  if valid_sets:
    box_xyz = np.concatenate(valid_sets, axis=0)
    low = np.minimum(low, box_xyz.min(axis=0))
    high = np.maximum(high, box_xyz.max(axis=0))
  center = (low + high) * 0.5
  span = np.maximum(high - low, 1.0e-3)
  # Preserve scene proportions while adding a small, consistent margin.
  span *= 1.06
  return center - span * 0.5, center + span * 0.5


def render_scene(path, points, corners, labels, limits, args):
  width_in, height_in = args.width / args.dpi, args.height / args.dpi
  figure = plt.figure(figsize=(width_in, height_in), dpi=args.dpi,
                      facecolor="white")
  axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
  axis.set_position([0, 0, 1, 1])
  xyz, rgb = points[:, :3], np.clip(points[:, 3:6], 0.0, 1.0)
  axis.scatter(
      xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=args.point_size,
      marker=".", linewidths=0, depthshade=False, rasterized=True, zorder=1)

  for box_index, corner in enumerate(corners):
    label = int(labels[box_index])
    color = CLASS_COLORS[label].astype(np.float32) / 255.0
    for begin, end in BOX_EDGES:
      segment = corner[[begin, end]]
      axis.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                color=color, linewidth=args.line_width, zorder=10)
    if args.show_labels:
      anchor = corner[:, 2].argmax()
      position = corner[anchor]
      axis.text(
          position[0], position[1], position[2], CLASS_NAMES[label],
          color=color, fontsize=5, zorder=11,
          bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=0.4))

  low, high = limits
  axis.set_xlim(low[0], high[0])
  axis.set_ylim(low[1], high[1])
  axis.set_zlim(low[2], high[2])
  axis.set_box_aspect(tuple(high - low), zoom=args.camera_zoom)
  axis.view_init(elev=args.elev, azim=args.azim)
  axis.set_proj_type("ortho")
  axis.set_axis_off()
  axis.set_facecolor("white")
  figure.subplots_adjust(0, 0, 1, 1)
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, dpi=args.dpi, facecolor="white", edgecolor="none",
                 pad_inches=0)
  plt.close(figure)


def compose_comparison(paths, titles, output, dpi):
  panels = []
  for path in paths:
    with Image.open(path) as image:
      panels.append(image.convert("RGB").copy())
  title_height = max(34, int(panels[0].height * 0.065))
  canvas = Image.new(
      "RGB", (sum(panel.width for panel in panels),
              max(panel.height for panel in panels) + title_height), "white")
  draw = ImageDraw.Draw(canvas)
  font_size = max(16, int(title_height * 0.46))
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
  except OSError:
    font = ImageFont.load_default()
  left = 0
  for title, panel in zip(titles, panels):
    box = draw.textbbox((0, 0), title, font=font)
    text_width = box[2] - box[0]
    draw.text((left + (panel.width - text_width) / 2, title_height * 0.22),
              title, fill="black", font=font)
    canvas.paste(panel, (left, title_height))
    left += panel.width
  canvas.save(output, dpi=(dpi, dpi))


def save_class_legend(path, dpi):
  """Save one reusable legend instead of shrinking every comparison panel."""
  width, height = 1800, 150
  canvas = Image.new("RGB", (width, height), "white")
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 25)
  except OSError:
    font = ImageFont.load_default()
  cell_width = width // 5
  for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    row, column = divmod(index, 5)
    x, y = column * cell_width + 18, row * 70 + 14
    draw.rectangle((x, y, x + 34, y + 25), fill=tuple(color.tolist()))
    draw.text((x + 47, y - 3), name, fill="black", font=font)
  canvas.save(path, dpi=(dpi, dpi))


def save_points_ply(path, points):
  from plyfile import PlyData, PlyElement
  vertex = np.empty(len(points), dtype=[
      ("x", "f4"), ("y", "f4"), ("z", "f4"),
      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
  vertex["x"], vertex["y"], vertex["z"] = points[:, :3].T
  rgb = np.clip(np.rint(points[:, 3:6] * 255), 0, 255).astype(np.uint8)
  vertex["red"], vertex["green"], vertex["blue"] = rgb.T
  PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def save_box_lines_ply(path, corners, labels):
  from plyfile import PlyData, PlyElement
  if len(corners) == 0:
    vertices = np.empty(0, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    edges = np.empty(0, dtype=[("vertex1", "i4"), ("vertex2", "i4")])
  else:
    vertices = np.empty(len(corners) * 8, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    flat = corners.reshape(-1, 3)
    vertices["x"], vertices["y"], vertices["z"] = flat.T
    color = np.repeat(CLASS_COLORS[labels], 8, axis=0)
    vertices["red"], vertices["green"], vertices["blue"] = color.T
    edges = np.empty(len(corners) * len(BOX_EDGES), dtype=[
        ("vertex1", "i4"), ("vertex2", "i4")])
    cursor = 0
    for box_index in range(len(corners)):
      for begin, end in BOX_EDGES:
        edges[cursor] = (box_index * 8 + begin, box_index * 8 + end)
        cursor += 1
  PlyData([
      PlyElement.describe(vertices, "vertex"),
      PlyElement.describe(edges, "edge")], text=False).write(str(path))


def main():
  args = parse_args()
  data_root = (REPO_ROOT / args.data_root).resolve()
  ann_path = Path(args.ann_file)
  if not ann_path.is_absolute():
    ann_path = data_root / ann_path
  output_root = (REPO_ROOT / args.output).resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  infos = load_infos(ann_path)
  indices = (args.indices if args.indices is not None else
             choose_representative_scenes(infos, args.num_scenes))
  if not indices:
    raise ValueError("No SUN RGB-D scenes selected.")
  if min(indices) < 0 or max(indices) >= len(infos):
    raise IndexError("SUN RGB-D validation index is out of range.")

  for index in indices:
    info = infos[index]
    _, labels = valid_gt(info)
    lidar_id = int(info["point_cloud"]["lidar_idx"])
    names = sorted({CLASS_NAMES[int(label)] for label in labels})
    print("index=%d lidar=%06d gt=%d classes=%s" %
          (index, lidar_id, len(labels), ",".join(names)))
  if args.list_scenes:
    return

  checkpoints = (
      ("octformer", "OctFormer", Path(args.octformer)),
      ("3det_mamba", "3DET-Mamba", Path(args.det_mamba)),
      ("pointttt", "PointTTT", Path(args.pointttt)),
  )
  checkpoints = tuple(
      (key, title, path if path.is_absolute() else REPO_ROOT / path)
      for key, title, path in checkpoints)
  for _, _, path in checkpoints:
    if not path.is_file():
      raise FileNotFoundError(path)

  patch_detection_environment()
  device = torch.device("cuda:%d" % args.gpu)
  model = build_model(
      (REPO_ROOT / args.config).resolve(), checkpoints[0][2], device)
  current_checkpoint = checkpoints[0][2].resolve()
  metadata = {
      key: dict(alias=title, **checkpoint_description(path))
      for key, title, path in checkpoints
  }
  records = []

  for scene_position, index in enumerate(indices, start=1):
    info = infos[index]
    lidar_id = int(info["point_cloud"]["lidar_idx"])
    stem = "%04d_%06d" % (index, lidar_id)
    scene_dir = output_root / stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    points, point_path = load_points(data_root, info)
    gt_boxes, gt_labels = valid_gt(info)
    gt_corners = box_corners_numpy(gt_boxes)
    chosen = render_indices(len(points), args.render_points,
                            args.seed + index)
    render_points = points[chosen]
    save_points_ply(scene_dir / "input_scene.ply", points)
    np.savez_compressed(
        scene_dir / "ground_truth.npz", boxes=gt_boxes,
        corners=gt_corners, labels=gt_labels)
    save_box_lines_ply(scene_dir / "ground_truth_boxes.ply",
                       gt_corners, gt_labels)
    predictions = {}
    box_counts = {}
    for key, title, checkpoint in checkpoints:
      prediction_path = scene_dir / (key + ".npz")
      image_path = scene_dir / (key + ".png")
      boxes_path = scene_dir / (key + "_boxes.ply")
      if args.skip_existing and prediction_path.is_file():
        with np.load(prediction_path) as saved:
          prediction = {name: saved[name] for name in saved.files}
        print("[%d/%d] scene=%s model=%s (reused)" %
              (scene_position, len(indices), stem, title))
        save_box_lines_ply(boxes_path, prediction["corners"],
                           prediction["labels"])
        predictions[key] = prediction
        box_counts[key] = int(len(prediction["scores"]))
        continue
      if checkpoint.resolve() != current_checkpoint:
        load_checkpoint_into(model, checkpoint)
        current_checkpoint = checkpoint.resolve()
      print("[%d/%d] scene=%s model=%s" %
            (scene_position, len(indices), stem, title))
      prediction = predict(
          model, points, device, args.score_thr, args.max_boxes)
      np.savez_compressed(prediction_path, **prediction)
      save_box_lines_ply(boxes_path, prediction["corners"],
                         prediction["labels"])
      predictions[key] = prediction
      box_counts[key] = int(len(prediction["scores"]))

    limits = scene_limits(
        points[:, :3],
        [gt_corners] + [predictions[key]["corners"]
                        for key, _, _ in checkpoints])
    render_scene(scene_dir / "input_scene.png", render_points,
                 np.empty((0, 8, 3)), np.empty(0, dtype=np.int64),
                 limits, args)
    render_scene(scene_dir / "ground_truth.png", render_points,
                 gt_corners, gt_labels, limits, args)
    prediction_files = []
    for key, _, _ in checkpoints:
      image_path = scene_dir / (key + ".png")
      prediction = predictions[key]
      render_scene(image_path, render_points, prediction["corners"],
                   prediction["labels"], limits, args)
      prediction_files.append(image_path)

    comparison = scene_dir / "comparison.png"
    compose_comparison(
        [scene_dir / "input_scene.png"] + prediction_files,
        ["Input Scene", "OctFormer", "3DET-Mamba", "PointTTT"],
        comparison, args.dpi)
    records.append({
        "validation_index": index,
        "lidar_idx": lidar_id,
        "point_file": str(point_path),
        "input_points": int(len(points)),
        "rendered_points": int(len(render_points)),
        "ground_truth_boxes": int(len(gt_boxes)),
        "prediction_boxes": box_counts,
        "directory": str(scene_dir),
    })

  del model
  torch.cuda.empty_cache()
  manifest = {
      "dataset": "SUN RGB-D validation split",
      "annotation_file": str(ann_path),
      "classes": list(CLASS_NAMES),
      "class_colors_rgb": CLASS_COLORS.tolist(),
      "score_threshold": args.score_thr,
      "max_boxes": args.max_boxes,
      "camera": {"elevation": args.elev, "azimuth": args.azim,
                 "projection": "orthographic", "zoom": args.camera_zoom},
      "checkpoint_alias_warning": (
          "Aliases are user-supplied. detector_type_from_metadata records "
          "the architecture actually stored in each checkpoint."),
      "checkpoints": metadata,
      "scenes": records,
  }
  with (output_root / "manifest.json").open("w") as handle:
    json.dump(manifest, handle, indent=2)
  with (output_root / "class_palette.json").open("w") as handle:
    json.dump(dict(zip(CLASS_NAMES, CLASS_COLORS.tolist())), handle, indent=2)
  save_class_legend(output_root / "class_legend.png", args.dpi)
  print("Saved SUN RGB-D visualization results to %s" % output_root)


if __name__ == "__main__":
  main()
