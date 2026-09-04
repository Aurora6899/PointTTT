#!/usr/bin/env python3
"""Measure batch-1 ModelNet40 latency and peak memory for a trained model."""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import ocnn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from datasets.modelnet40 import read_file
from models.point_ttt_cls import PointTTTCls


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--base-lr', type=float, default=1.0)
  parser.add_argument('--update', choices=('on', 'off'), default='on')
  parser.add_argument('--warmup', type=int, default=50)
  parser.add_argument('--repeat', type=int, default=200)
  parser.add_argument(
      '--sample', type=Path,
      default=Path('data/ModelNet40/ModelNet40.ply.normalize/airplane/test/'
                   'airplane_0627.ply'))
  return parser.parse_args()


def load_state_dict(path):
  checkpoint = torch.load(path, map_location='cpu')
  if isinstance(checkpoint, dict):
    for key in ('model_dict', 'state_dict', 'model'):
      if isinstance(checkpoint.get(key), dict):
        checkpoint = checkpoint[key]
        break
  return {
      key[len('module.'):] if key.startswith('module.') else key: value
      for key, value in checkpoint.items()
  }


def build_input(path, device):
  sample = read_file(str(path))
  points = ocnn.octree.Points(
      torch.from_numpy(sample['points']).to(device),
      torch.from_numpy(sample['normals']).to(device))
  points.orient_normal('xyz')
  points.clip(min=-1.0, max=1.0)
  octree = ocnn.octree.Octree(6, 2, device=device)
  octree.build_octree(points)
  octree.construct_all_neigh()
  feature = ocnn.modules.InputFeature('ND', nempty=False)(octree)
  return feature, octree, int(points.points.shape[0])


def main():
  args = parse_args()
  if args.warmup < 1 or args.repeat < 1:
    raise ValueError('--warmup and --repeat must be positive')
  torch.cuda.set_device(args.gpu)
  device = torch.device('cuda', args.gpu)
  update_enabled = args.update == 'on'

  with contextlib.redirect_stdout(io.StringIO()):
    model = PointTTTCls(
        in_channels=4,
        out_channels=40,
        channels=[192],
        num_blocks=[2],
        drop_path=0.3,
        nempty=False,
        stem_down=2,
        head_drop=0.5,
        ttt_base_lr=args.base_lr,
        ttt_update_train=update_enabled,
        ttt_update_test=update_enabled)
  model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
  model.to(device).eval()
  feature, octree, input_points = build_input(args.sample, device)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  with torch.no_grad():
    for _ in range(args.warmup):
      output = model(feature, octree, octree.depth)
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    start.record()
    for _ in range(args.repeat):
      output = model(feature, octree, octree.depth)
    end.record()
    torch.cuda.synchronize(device)

  result = {
      'checkpoint': str(args.checkpoint.resolve()),
      'gpu': torch.cuda.get_device_name(device),
      'batch_size': 1,
      'input_points': input_points,
      'base_lr': args.base_lr,
      'test_time_update': update_enabled,
      'warmup': args.warmup,
      'repeat': args.repeat,
      'latency_ms': start.elapsed_time(end) / args.repeat,
      'peak_memory_mib': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
      'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
      'output_shape': list(output.shape),
      'parameters': sum(parameter.numel() for parameter in model.parameters()),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
