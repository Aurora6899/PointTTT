#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from datasets.semantickitti import (  # noqa: E402
    SEMANTICKITTI_LEARNING_MAP, SEMANTICKITTI_SPLITS,
    resolve_semantickitti_roots)


def frame_files(folder, suffix):
  if not os.path.isdir(folder):
    return []
  return sorted(file for file in os.listdir(folder) if file.endswith(suffix))


def main():
  parser = argparse.ArgumentParser(
      description='Checks the raw SemanticKITTI archives used by PointTTT.')
  parser.add_argument('--root', default='data/SemanticKITTI')
  parser.add_argument(
      '--check-label-values', action='store_true',
      help='Also scan every label value (slower than the default size checks).')
  args = parser.parse_args()

  point_root, label_root = resolve_semantickitti_roots(args.root)
  expected_train = set(SEMANTICKITTI_SPLITS['train'])
  expected_val = set(SEMANTICKITTI_SPLITS['val'])
  expected_test = set(SEMANTICKITTI_SPLITS['test'])
  errors = []
  totals = {'train': 0, 'val': 0, 'test': 0}

  print('Point sequences:', point_root)
  print('Label sequences:', label_root)
  print('seq  split   scans  labels')
  for sequence in range(22):
    name = '%02d' % sequence
    point_dir = os.path.join(point_root, name, 'velodyne')
    label_dir = os.path.join(label_root, name, 'labels')
    points = frame_files(point_dir, '.bin')
    labels = frame_files(label_dir, '.label')
    if sequence in expected_train:
      split = 'train'
    elif sequence in expected_val:
      split = 'val'
    else:
      split = 'test'
    totals[split] += len(points)
    print('%s   %-5s  %5d  %6d' % (name, split, len(points), len(labels)))

    if not points:
      errors.append('Sequence %s has no .bin scans.' % name)
      continue
    if sequence in expected_test:
      if labels:
        errors.append('Test sequence %s unexpectedly contains labels.' % name)
      continue
    expected_labels = [os.path.splitext(file)[0] + '.label' for file in points]
    if labels != expected_labels:
      errors.append('Sequence %s point/label file names do not match.' % name)
      continue

    for point_file, label_file in zip(points, labels):
      point_path = os.path.join(point_dir, point_file)
      label_path = os.path.join(label_dir, label_file)
      point_bytes, label_bytes = os.path.getsize(point_path), os.path.getsize(label_path)
      if point_bytes % 16 != 0 or label_bytes % 4 != 0 or \
          point_bytes // 16 != label_bytes // 4:
        errors.append('Point/label length mismatch: %s' % point_path)
        continue
      if args.check_label_values:
        raw = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        unknown = set(np.unique(raw).tolist()) - set(SEMANTICKITTI_LEARNING_MAP)
        if unknown:
          errors.append('%s has unknown labels %r.' % (label_path, unknown))

  print('Totals: train=%d, val=%d, test=%d' %
        (totals['train'], totals['val'], totals['test']))
  if errors:
    print('\nFAILED:')
    for error in errors[:50]:
      print(' -', error)
    if len(errors) > 50:
      print(' - ... and %d more errors' % (len(errors) - 50))
    raise SystemExit(1)
  print('SemanticKITTI integrity check passed.')


if __name__ == '__main__':
  main()
