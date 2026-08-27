#!/usr/bin/env python3
"""Benchmark PointTTT ScanNet scalability on one GPU.

The benchmark uses nested, deterministic subsets from one real ScanNet scene.
Disk I/O and metric computation are outside the measured regions.
"""

import argparse
import contextlib
import csv
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ocnn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import builder


POINT_COUNTS = [
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
]


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint',
      default='logs/scannet/from_scratch_1cm/best_model.pth')
  parser.add_argument(
      '--scene', default='data/scannet.npz/train/scene0054_00.npz')
  parser.add_argument(
      '--output',
      default='logs/scannet/from_scratch_1cm/scalability_rtx3090.json')
  parser.add_argument('--warmup', type=int, default=50)
  parser.add_argument('--repeats', type=int, default=200)
  parser.add_argument('--seed', type=int, default=2026)
  return parser.parse_args()


def load_scene(path, seed):
  raw = np.load(path)
  xyz = raw['points'].astype(np.float32)
  normals = raw['normals'].astype(np.float32)
  colors = raw['colors'].astype(np.float32) / 255.0
  if xyz.shape[0] < max(POINT_COUNTS):
    raise ValueError(
        f'{path} has only {xyz.shape[0]} points; '
        f'{max(POINT_COUNTS)} are required.')

  # Match ScanNetTransform while keeping the normalization fixed for every
  # nested subset from this scene.
  center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
  xyz = (xyz - center) / 10.24
  xyz[:, 2] -= xyz[:, 2].min()

  generator = np.random.default_rng(seed)
  order = generator.permutation(xyz.shape[0])[:max(POINT_COUNTS)]
  return (
      torch.from_numpy(xyz[order]),
      torch.from_numpy(normals[order]),
      torch.from_numpy(colors[order]),
      int(raw['points'].shape[0]),
  )


def build_model(checkpoint, device):
  # PointMamba currently prints every repeated TTT configuration at
  # construction time; suppress that debug-only output in benchmark logs.
  with contextlib.redirect_stdout(io.StringIO()):
    model = builder.PointMambaSeg_base(
        in_channels=10, out_channels=21, interp='nearest', nempty=True)
  state_dict = torch.load(checkpoint, map_location='cpu')
  model.load_state_dict(state_dict, strict=True)
  model.to(device).eval()
  return model


def make_pipeline(model, xyz, normals, colors, device):
  # The input tensors are transferred before measurement, so host-to-device
  # transfer is treated as data loading and excluded.
  xyz = xyz.to(device)
  normals = normals.to(device)
  colors = colors.to(device)

  def pipeline(return_inputs=False):
    points = ocnn.octree.Points(xyz, normals, colors)
    octree = ocnn.octree.Octree(11, 2, device=device)
    octree.build_octree(points)
    octree.construct_all_neigh()
    feature = ocnn.modules.InputFeature('NDFP', True)(octree)
    batch_id = torch.zeros(
        (points.points.shape[0], 1), dtype=points.points.dtype, device=device)
    query_pts = torch.cat([points.points, batch_id], dim=1)
    logits = model(feature, octree, octree.depth, query_pts)
    if return_inputs:
      return logits, feature, octree, query_pts
    return logits

  return pipeline


def cuda_event_latency(pipeline, warmup, repeats):
  for _ in range(warmup):
    pipeline()
  torch.cuda.synchronize()

  starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
  ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
  for idx in range(repeats):
    starts[idx].record()
    pipeline()
    ends[idx].record()
  torch.cuda.synchronize()
  samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
  return float(np.mean(samples)), float(np.std(samples))


def peak_memory(pipeline):
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  pipeline()
  torch.cuda.synchronize()
  peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
  peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
  return float(peak_allocated), float(peak_reserved)


def profile_forward_flops(model, prepared):
  _, feature, octree, query_pts = prepared
  activities = [torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
  with torch.profiler.profile(
      activities=activities, with_flops=True, record_shapes=False) as prof:
    model(feature, octree, octree.depth, query_pts)
  torch.cuda.synchronize()

  counted = 0
  counted_ops = {}
  for event in prof.key_averages():
    flops = int(event.flops or 0)
    if flops:
      counted += flops
      counted_ops[event.key] = flops
  return counted / 1.0e9, counted_ops


def write_outputs(output_path, metadata, results):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {'metadata': metadata, 'results': results}
  output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

  csv_path = output_path.with_suffix('.csv')
  fields = [
      'num_points', 'forward_gflops_counted', 'latency_ms_mean',
      'latency_ms_std', 'peak_allocated_mib', 'peak_reserved_mib', 'status',
  ]
  with csv_path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for result in results:
      writer.writerow({field: result.get(field) for field in fields})
  return str(output_path), str(csv_path)


def main():
  args = parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required for this benchmark.')
  device = torch.device('cuda:0')
  gpu_name = torch.cuda.get_device_name(device)
  if 'RTX 3090' not in gpu_name:
    print(f'WARNING: requested RTX 3090, detected {gpu_name}', flush=True)

  xyz, normals, colors, scene_points = load_scene(args.scene, args.seed)
  model = build_model(args.checkpoint, device)
  parameters = sum(parameter.numel() for parameter in model.parameters())
  metadata = {
      'checkpoint': args.checkpoint,
      'scene': args.scene,
      'scene_raw_points': scene_points,
      'seed': args.seed,
      'gpu': gpu_name,
      'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
      'batch_size': 1,
      'warmup': args.warmup,
      'repeats': args.repeats,
      'parameters': parameters,
      'torch': torch.__version__,
      'flops_scope': 'one model forward on a prebuilt octree',
      'flops_note': (
          'PyTorch profiler counted FLOPs only; custom octree CUDA operators '
          'without profiler FLOP formulas are not included.'),
      'latency_scope': (
          'GPU-resident points to logits: octree/neighbors, NDFP feature, '
          'TTT inner computation, and segmentation forward'),
      'memory_scope': 'full latency pipeline including resident model',
  }
  print(json.dumps(metadata, indent=2), flush=True)

  results = []
  with torch.inference_mode():
    for num_points in POINT_COUNTS:
      print(f'BENCHMARK_START n={num_points}', flush=True)
      result = {'num_points': num_points, 'status': 'ok'}
      started = time.time()
      try:
        pipeline = make_pipeline(
            model, xyz[:num_points], normals[:num_points],
            colors[:num_points], device)

        # Build once outside the forward-only FLOP region.
        prepared = pipeline(return_inputs=True)
        torch.cuda.synchronize()
        gflops, counted_ops = profile_forward_flops(model, prepared)
        result['forward_gflops_counted'] = gflops
        result['flop_counted_ops'] = counted_ops
        del prepared

        latency_mean, latency_std = cuda_event_latency(
            pipeline, args.warmup, args.repeats)
        result['latency_ms_mean'] = latency_mean
        result['latency_ms_std'] = latency_std
        peak_allocated, peak_reserved = peak_memory(pipeline)
        result['peak_allocated_mib'] = peak_allocated
        result['peak_reserved_mib'] = peak_reserved
        result['elapsed_seconds'] = time.time() - started
        print('BENCHMARK_RESULT ' + json.dumps(result), flush=True)
        del pipeline
      except torch.cuda.OutOfMemoryError as error:
        result.update({
            'status': 'OOM',
            'error': str(error),
            'elapsed_seconds': time.time() - started,
        })
        print('BENCHMARK_RESULT ' + json.dumps(result), flush=True)
      finally:
        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()
        write_outputs(args.output, metadata, results)

  json_path, csv_path = write_outputs(args.output, metadata, results)
  print(f'OUTPUT_JSON {json_path}', flush=True)
  print(f'OUTPUT_CSV {csv_path}', flush=True)


if __name__ == '__main__':
  main()
