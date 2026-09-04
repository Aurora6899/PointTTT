#!/usr/bin/env python3
"""Profile one 2048-point ShapeNetPart forward pass."""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import ocnn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from models.point_ttt_seg import PointTTTSeg


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint',
      default='logs/shapenetpart/from_scratch_2048/best_model.pth')
  parser.add_argument(
      '--sample',
      default=('data/shapenet_part_seg_hdf5_data/hdf5_data/'
               'ply_data_test0.h5'))
  parser.add_argument('--sample-index', type=int, default=0)
  parser.add_argument('--warmup', type=int, default=10)
  return parser.parse_args()


def load_points(path, sample_index, device):
  with h5py.File(path, 'r') as handle:
    xyz = np.asarray(handle['data'][sample_index], dtype=np.float32)
  xyz -= xyz.mean(axis=0, keepdims=True)
  radius = np.sqrt(np.sum(xyz * xyz, axis=1)).max()
  if radius > 1.0e-12:
    xyz /= radius
  max_abs = float(np.abs(xyz).max())
  if max_abs > 0.999:
    xyz *= 0.999 / max_abs
  if xyz.shape != (2048, 3):
    raise ValueError(f'Expected a [2048, 3] sample, got {xyz.shape}.')
  return ocnn.octree.Points(torch.from_numpy(xyz).to(device))


def main():
  args = parse_args()
  device = torch.device('cuda:0')
  with contextlib.redirect_stdout(io.StringIO()):
    model = PointTTTSeg(
        in_channels=3,
        out_channels=50,
        channels=[96, 192, 384, 384],
        num_blocks=[2, 2, 18, 2],
        drop_path=0.5,
        nempty=True,
        stem_down=2,
        head_up=2,
        fpn_channel=168,
        head_drop=[0.5, 0.5],
    )
  state_dict = torch.load(args.checkpoint, map_location='cpu')
  model.load_state_dict(state_dict, strict=True)
  model.to(device).eval()

  points = load_points(args.sample, args.sample_index, device)
  octree = ocnn.octree.Octree(8, 2, device=device)
  octree.build_octree(points)
  octree.construct_all_neigh()
  feature = ocnn.modules.InputFeature('P', True)(octree)
  batch_id = torch.zeros(
      (points.points.shape[0], 1), dtype=points.points.dtype, device=device)
  query_pts = torch.cat([points.points, batch_id], dim=1)

  with torch.inference_mode():
    for _ in range(args.warmup):
      model(feature, octree, octree.depth, query_pts)
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, with_flops=True, record_shapes=False) as prof:
      output = model(feature, octree, octree.depth, query_pts)
    torch.cuda.synchronize()

  counted_ops = {
      event.key: int(event.flops)
      for event in prof.key_averages() if event.flops
  }
  counted_flops = sum(counted_ops.values())
  result = {
      'checkpoint': args.checkpoint,
      'gpu': torch.cuda.get_device_name(device),
      'batch_size': 1,
      'input_points': int(points.points.shape[0]),
      'output_shape': list(output.shape),
      'parameters': sum(parameter.numel() for parameter in model.parameters()),
      'forward_gflops_counted': counted_flops / 1.0e9,
      'counted_ops': counted_ops,
      'note': ('Custom octree CUDA operators without PyTorch profiler FLOP '
               'formulas are not included.'),
  }
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
