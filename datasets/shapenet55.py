import os

import numpy as np
import torch
from ocnn.dataset import CollateBatch
from ocnn.octree import Points

from .utils import Transform


def _read_filelist(filelist: str):
  if not os.path.isfile(filelist):
    raise FileNotFoundError('ShapeNet55 file list not found: ' + filelist)

  with open(filelist, 'r') as fid:
    filenames = [line.strip() for line in fid if line.strip()]
  if not filenames:
    raise ValueError('ShapeNet55 file list is empty: ' + filelist)
  return filenames


def _taxonomy_id(filename: str) -> str:
  basename = os.path.basename(filename)
  fields = basename.split('-', 1)
  if len(fields) != 2 or not fields[0]:
    raise ValueError(
        'Invalid ShapeNet55 filename %r; expected TAXONOMY-MODEL.npy.' %
        filename)
  return fields[0]


def _class_ids(filelist: str):
  r'''Builds one stable label mapping shared by the train and test splits.'''
  list_root = os.path.dirname(os.path.abspath(filelist))
  split_lists = [os.path.join(list_root, split + '.txt')
                 for split in ('train', 'test')]
  filenames = []
  for split_file in split_lists:
    if os.path.isfile(split_file):
      filenames.extend(_read_filelist(split_file))
  if not filenames:
    filenames = _read_filelist(filelist)
  return sorted({_taxonomy_id(filename) for filename in filenames})


class ShapeNet55Transform(Transform):

  def __init__(self, flags):
    super().__init__(flags)
    self.split = flags.split.lower()
    self.num_points = int(flags.num_points)
    self.sampling = flags.sampling.lower()
    self.normalize = bool(flags.normalize)

    if self.num_points < 1:
      raise ValueError('ShapeNet55 num_points must be positive.')
    if self.sampling not in ('random', 'first', 'none'):
      raise ValueError(
          'Unknown ShapeNet55 sampling mode %r. Choose random, first, or none.'
          % self.sampling)

  def _sample_points(self, points: np.ndarray, idx: int) -> np.ndarray:
    num_input = points.shape[0]
    if self.sampling == 'none':
      if self.num_points != num_input:
        raise ValueError(
            'sampling=none keeps all %d input points, but num_points is %d.' %
            (num_input, self.num_points))
      return points

    if self.num_points > num_input:
      raise ValueError(
          'Cannot sample %d points from a ShapeNet55 cloud with %d points.' %
          (self.num_points, num_input))
    if self.sampling == 'first':
      return points[:self.num_points]

    # Point-BERT randomly samples 1024 points from each 8192-point ShapeNet55
    # cloud. Keep training stochastic, and make testing deterministic so that
    # validation accuracy is reproducible across epochs and worker counts.
    if self.split == 'train':
      indices = np.random.permutation(num_input)[:self.num_points]
    else:
      indices = np.random.RandomState(idx).permutation(num_input)
      indices = indices[:self.num_points]
    return points[indices]

  @staticmethod
  def _normalize_points(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    radius = np.sqrt(np.sum(points * points, axis=1)).max()
    if radius > 1.0e-6:
      points = points / radius
    return points

  def preprocess(self, sample: dict, idx: int):
    points = np.asarray(sample['points'], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
      raise ValueError(
          'Expected ShapeNet55 points with shape [N, 3], got %s.' %
          (points.shape,))
    if not np.isfinite(points).all():
      raise ValueError('ShapeNet55 point cloud contains NaN or Inf values.')

    points = self._sample_points(points, idx)
    if self.normalize:
      points = self._normalize_points(points)
    points = np.ascontiguousarray(points, dtype=np.float32)
    return Points(torch.from_numpy(points))


class ShapeNet55Dataset(torch.utils.data.Dataset):

  def __init__(self, point_root: str, filelist: str, transform,
               take: int = -1, expected_num_classes: int = 55):
    super().__init__()
    self.point_root = os.path.abspath(point_root)
    self.filelist = os.path.abspath(filelist)
    self.transform = transform

    if not os.path.isdir(self.point_root):
      raise FileNotFoundError(
          'ShapeNet55 point-cloud directory not found: ' + self.point_root)

    filenames = _read_filelist(self.filelist)
    class_ids = _class_ids(self.filelist)
    if expected_num_classes > 0 and len(class_ids) != expected_num_classes:
      raise ValueError(
          'Expected %d ShapeNet55 classes, found %d in %s.' %
          (expected_num_classes, len(class_ids), self.filelist))

    self.class_to_idx = {
        taxonomy_id: idx for idx, taxonomy_id in enumerate(class_ids)}
    if take > 0:
      filenames = filenames[:take]
    self.samples = []
    for filename in filenames:
      taxonomy_id = _taxonomy_id(filename)
      self.samples.append((filename, self.class_to_idx[taxonomy_id]))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    filename, label = self.samples[idx]
    path = os.path.join(self.point_root, filename)
    if not os.path.isfile(path):
      raise FileNotFoundError('ShapeNet55 point cloud not found: ' + path)

    sample = {'points': np.load(path, allow_pickle=False)}
    output = self.transform(sample, idx)
    output['label'] = label
    output['filename'] = filename
    return output


def get_shapenet55_dataset(flags):
  transform = ShapeNet55Transform(flags)
  dataset = ShapeNet55Dataset(
      flags.point_root, flags.filelist, transform,
      take=getattr(flags, 'take', -1),
      expected_num_classes=getattr(flags, 'num_classes', 55))
  return dataset, CollateBatch()
