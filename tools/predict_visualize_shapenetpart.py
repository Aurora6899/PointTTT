#!/usr/bin/env python3
"""Predict and visualize the complete ShapeNetPart test split.

The saved arrays retain the original HDF5 test order. Predictions are
restricted to the valid part ids of each object category, matching the
project's Point-BERT-style ShapeNetPart evaluation.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ocnn
from plyfile import PlyData, PlyElement
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import builder  # noqa: E402
from datasets.shapenetpart import (  # noqa: E402
    SHAPENETPART_CATEGORIES, SHAPENETPART_PARTS,
    SHAPENETPART_UTONIA_TTA, build_shapenetpart_tta_points)
from thsolver.config import FLAGS, _load_from_file  # noqa: E402


DEFAULT_CHECKPOINTS = {
    'ptv3': ('logs/shapenetpart/from_scratch_2048/checkpoints/'
             '00298.model.pth'),
    'pcm': 'logs/shapenetpart/from_scratch_2048/best_model.pth',
    'pointTTT': 'logs/shapenetpart/from_scratch_2048/best_model.pth',
}


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='configs/seg_shapenetpart.yaml')
  parser.add_argument('--ptv3', default=DEFAULT_CHECKPOINTS['ptv3'])
  parser.add_argument('--pcm', default=DEFAULT_CHECKPOINTS['pcm'])
  parser.add_argument('--pointttt', default=DEFAULT_CHECKPOINTS['pointTTT'])
  parser.add_argument(
      '--output', default='visual_results/shapenetpart_test_comparison')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--batch-size', type=int, default=8)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument(
      '--visualize-indices', type=int, nargs='*', default=None,
      help=('Test indices to render. By default, the first example of every '
            'ShapeNetPart category is rendered.'))
  parser.add_argument('--point-size', type=float, default=4.0)
  parser.add_argument('--skip-prediction', action='store_true')
  parser.add_argument(
      '--reuse-existing', action='store_true',
      help=('Reuse any prediction archives already present in --output and '
            'only run missing models.'))
  parser.add_argument(
      '--tta-seed', type=int, default=0,
      help='Random seed for the Utonia random-flip vote used by PointTTT.')
  return parser.parse_args()


def load_config(path):
  cfg = FLAGS.clone()
  cfg.defrost()
  for item in _load_from_file(path):
    cfg.merge_from_other_cfg(item)
  cfg.DATA.test.batch_size = 1
  cfg.DATA.test.shuffle = False
  cfg.DATA.test.num_workers = 0
  cfg.freeze()
  return cfg


def checkpoint_state(path):
  state = torch.load(path, map_location='cpu')
  if isinstance(state, dict) and 'model_dict' in state:
    state = state['model_dict']
  if not isinstance(state, dict):
    raise TypeError('Checkpoint does not contain a state_dict: ' + path)
  if state and next(iter(state)).startswith('module.'):
    state = {key[7:]: value for key, value in state.items()}
  return state


def state_digest(state):
  digest = hashlib.sha256()
  for key in sorted(state):
    digest.update(key.encode('utf-8'))
    digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
  return digest.hexdigest()


def build_model(cfg, checkpoint, device):
  # The current PointTTT modules print every TTT configuration at
  # construction. Suppress that diagnostic noise for batch inference.
  with contextlib.redirect_stdout(io.StringIO()):
    model = builder.get_segmentation_model(cfg.MODEL)
  state = checkpoint_state(checkpoint)
  model.load_state_dict(state, strict=True)
  model.to(device).eval()
  return model, state_digest(state)


def build_batch(points_list, flags, device):
  points_list = [points.to(device) for points in points_list]
  octrees = []
  for points in points_list:
    octree = ocnn.octree.Octree(
        flags.depth, flags.full_depth, device=device)
    octree.build_octree(points)
    octrees.append(octree)
  octree = ocnn.octree.merge_octrees(octrees)
  octree.construct_all_neigh()
  points = ocnn.octree.merge_points(points_list)
  feature = ocnn.modules.InputFeature('P', True)(octree)
  query = torch.cat([points.points, points.batch_id], dim=1)
  return feature, octree, query, points


def restrict_predictions(logit, categories, batch_npt):
  output = []
  start = 0
  for category, npt in zip(categories, batch_npt):
    category, npt = int(category), int(npt)
    valid_parts = torch.as_tensor(
        SHAPENETPART_PARTS[category], dtype=torch.long, device=logit.device)
    local = logit[start:start + npt, valid_parts].argmax(dim=1)
    output.append(valid_parts[local])
    start += npt
  return torch.cat(output, dim=0)


def predict(model, loader, flags, device, num_samples, num_points):
  prediction = np.empty((num_samples, num_points), dtype=np.uint8)
  cursor = 0
  with torch.inference_mode():
    for batch in tqdm(loader, ncols=80, desc='predict', leave=False):
      feature, octree, query, points = build_batch(
          batch['points'], flags, device)
      logit = model(feature, octree, octree.depth, query)
      pred = restrict_predictions(logit, batch['label'], points.batch_npt)
      split = torch.split(pred.cpu(), points.batch_npt.tolist())
      for item in split:
        if item.numel() != num_points:
          raise ValueError('Unexpected ShapeNetPart point count.')
        prediction[cursor] = item.numpy().astype(np.uint8, copy=False)
        cursor += 1
  if cursor != num_samples:
    raise RuntimeError('Prediction count mismatch: %d vs %d.' %
                       (cursor, num_samples))
  return prediction


def predict_pointttt_tta(
    model, loader, flags, device, num_samples, num_points, seed):
  r'''Runs the official Utonia ten-vote ShapeNetPart TTA protocol.

  Each augmented branch is forwarded independently. Its softmax scores are
  summed point-wise, then the final label is selected only from the parts
  valid for the object's category, exactly as in the project's final TTA
  evaluator.
  '''
  prediction = np.empty((num_samples, num_points), dtype=np.uint8)
  cursor = 0
  np.random.seed(seed)
  with torch.inference_mode():
    for batch in tqdm(loader, ncols=80, desc='PointTTT 10-vote TTA',
                      leave=False):
      probability_sum = None
      batch_npt = None
      for scale, flip_probability in SHAPENETPART_UTONIA_TTA:
        augmented = build_shapenetpart_tta_points(
            batch['points'], scale, flip_probability,
            float(getattr(flags, 'octree_bound', 0.999)))
        feature, octree, query, merged_points = build_batch(
            augmented, flags, device)
        logit = model(feature, octree, octree.depth, query)
        probability = torch.softmax(logit, dim=1)
        if probability_sum is None:
          probability_sum = probability
          batch_npt = merged_points.batch_npt
        else:
          if not torch.equal(batch_npt, merged_points.batch_npt):
            raise RuntimeError('Point counts changed between TTA votes.')
          probability_sum.add_(probability)

      pred = restrict_predictions(
          probability_sum, batch['label'], batch_npt)
      split = torch.split(pred.cpu(), batch_npt.tolist())
      for item in split:
        if item.numel() != num_points:
          raise ValueError('Unexpected ShapeNetPart point count.')
        prediction[cursor] = item.numpy().astype(np.uint8, copy=False)
        cursor += 1
  if cursor != num_samples:
    raise RuntimeError('Prediction count mismatch: %d vs %d.' %
                       (cursor, num_samples))
  return prediction


def collect_dataset(dataset):
  num_samples = len(dataset)
  first = dataset[0]
  num_points = int(first['points'].points.shape[0])
  points = np.empty((num_samples, num_points, 3), dtype=np.float32)
  labels = np.empty((num_samples, num_points), dtype=np.uint8)
  categories = np.empty(num_samples, dtype=np.uint8)
  filenames = []
  for index in tqdm(range(num_samples), ncols=80, desc='load test data'):
    sample = dataset[index]
    xyz = sample['points'].points.numpy()
    part = sample['points'].labels.numpy()
    if xyz.shape != (num_points, 3) or part.shape != (num_points,):
      raise ValueError('Inconsistent ShapeNetPart test sample shape.')
    points[index] = xyz
    labels[index] = part.astype(np.uint8, copy=False)
    categories[index] = int(sample['label'])
    filenames.append(sample['filename'])
  return points, labels, categories, np.asarray(filenames)


def metric_summary(prediction, target, categories):
  shape_ious = []
  category_ious = [[] for _ in SHAPENETPART_CATEGORIES]
  for pred, label, category in zip(prediction, target, categories):
    part_ious = []
    for part in SHAPENETPART_PARTS[int(category)]:
      pred_mask, label_mask = pred == part, label == part
      union = np.logical_or(pred_mask, label_mask).sum()
      iou = 1.0 if union == 0 else (
          np.logical_and(pred_mask, label_mask).sum() / union)
      part_ious.append(float(iou))
    shape_iou = float(np.mean(part_ious))
    shape_ious.append(shape_iou)
    category_ious[int(category)].append(shape_iou)
  return {
      'point_accuracy': float((prediction == target).mean()),
      'mIoUI': float(np.mean(shape_ious)),
      'mIoUC': float(np.mean([np.mean(values) for values in category_ious])),
      'category_iou': {
          name: float(np.mean(values))
          for name, values in zip(SHAPENETPART_CATEGORIES, category_ious)},
  }


def load_colors(data_root):
  path = Path(data_root) / 'part_color_mapping.json'
  with path.open('r') as handle:
    colors = np.asarray(json.load(handle), dtype=np.float32)
  if colors.shape != (50, 3):
    raise ValueError('Expected a [50, 3] ShapeNetPart color mapping.')
  return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def write_colored_ply(path, points, labels, colors):
  path.parent.mkdir(parents=True, exist_ok=True)
  vertex = np.empty(points.shape[0], dtype=[
      ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')])
  vertex['x'], vertex['y'], vertex['z'] = points.T
  rgb = colors[labels]
  vertex['red'], vertex['green'], vertex['blue'] = rgb.T
  vertex['label'] = labels
  PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(str(path))


def equal_axes(ax, points):
  center = (points.min(0) + points.max(0)) * 0.5
  radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 1.0e-4)
  ax.set_xlim(center[0] - radius, center[0] + radius)
  ax.set_ylim(center[1] - radius, center[1] + radius)
  ax.set_zlim(center[2] - radius, center[2] + radius)
  ax.set_box_aspect((1, 1, 1))


def render_comparison(path, points, label_sets, colors, point_size):
  figure = plt.figure(figsize=(16, 4), dpi=180)
  for panel, (name, labels) in enumerate(label_sets.items(), start=1):
    ax = figure.add_subplot(1, len(label_sets), panel, projection='3d')
    rgb = colors[labels].astype(np.float32) / 255.0
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb,
               s=point_size, marker='.', linewidths=0, depthshade=False)
    equal_axes(ax, points)
    ax.view_init(elev=20, azim=45)
    ax.set_title(name)
    ax.set_axis_off()
  figure.tight_layout(pad=0.2)
  path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(path, bbox_inches='tight', facecolor='white')
  plt.close(figure)


def default_visual_indices(categories):
  indices = []
  for category in range(len(SHAPENETPART_CATEGORIES)):
    matches = np.flatnonzero(categories == category)
    if matches.size:
      indices.append(int(matches[0]))
  return indices


def main():
  args = parse_args()
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  cfg = load_config(args.config)
  dataset, collate = builder.get_segmentation_dataset(cfg.DATA.test)
  points, ground_truth, categories, filenames = collect_dataset(dataset)
  num_samples, num_points = ground_truth.shape
  np.savez_compressed(
      output / 'ground_truth.npz', points=points, labels=ground_truth,
      categories=categories, filenames=filenames)

  checkpoints = {
      'ptv3': args.ptv3,
      'pcm': args.pcm,
      'pointTTT': args.pointttt,
  }
  device = torch.device('cuda:%d' % args.gpu)
  manifest = {
      'config': str(Path(args.config).resolve()),
      'test_samples': num_samples,
      'points_per_sample': num_points,
      'models': {},
  }
  predictions = {}
  for name, checkpoint in checkpoints.items():
    result_path = output / (name + '.npz')
    reuse_result = args.skip_prediction or (
        args.reuse_existing and result_path.is_file())
    if reuse_result:
      with np.load(result_path, allow_pickle=False) as saved:
        prediction = saved['labels']
      state = checkpoint_state(checkpoint)
      digest = state_digest(state)
    else:
      model, digest = build_model(cfg, checkpoint, device)
      loader = DataLoader(
          dataset, batch_size=args.batch_size, shuffle=False,
          num_workers=args.num_workers, collate_fn=collate,
          pin_memory=True)
      print('Predicting %s from %s' % (name, checkpoint))
      if name == 'pointTTT':
        prediction = predict_pointttt_tta(
            model, loader, cfg.DATA.test, device, num_samples, num_points,
            args.tta_seed)
      else:
        prediction = predict(
            model, loader, cfg.DATA.test, device, num_samples, num_points)
      np.savez_compressed(result_path, labels=prediction)
      del model
      torch.cuda.empty_cache()
    if prediction.shape != ground_truth.shape:
      raise ValueError('%s prediction shape is incorrect.' % name)
    predictions[name] = prediction
    manifest['models'][name] = {
        'checkpoint': str(Path(checkpoint).resolve()),
        'sha256_state_dict': digest,
        'prediction_file': str(result_path),
        'prediction_protocol': (
            'Utonia 10-vote TTA; summed softmax; category-restricted argmax'
            if name == 'pointTTT' else
            'single-pass; category-restricted argmax'),
        'metrics': metric_summary(prediction, ground_truth, categories),
    }

  colors = load_colors(cfg.DATA.test.location)
  indices = (args.visualize_indices if args.visualize_indices is not None
             else default_visual_indices(categories))
  for index in indices:
    if index < 0 or index >= num_samples:
      raise IndexError('Visualization index out of range: %d.' % index)
    category_name = SHAPENETPART_CATEGORIES[int(categories[index])]
    stem = '%04d_%s' % (index, category_name)
    folder = output / 'visualizations' / stem
    labels = {'Ground Truth': ground_truth[index]}
    labels.update({name: prediction[index]
                   for name, prediction in predictions.items()})
    for name, part in labels.items():
      file_name = name.lower().replace(' ', '_') + '.ply'
      write_colored_ply(folder / file_name, points[index], part, colors)
    render_comparison(
        folder / 'comparison.png', points[index], labels, colors,
        args.point_size)

  # Make identical inputs/results explicit rather than silently presenting
  # duplicated files as independent experiments.
  names = list(checkpoints)
  manifest['identical_predictions'] = []
  for left_index, left in enumerate(names):
    for right in names[left_index + 1:]:
      if np.array_equal(predictions[left], predictions[right]):
        manifest['identical_predictions'].append([left, right])
  with (output / 'manifest.json').open('w') as handle:
    json.dump(manifest, handle, indent=2)
  print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
  main()
