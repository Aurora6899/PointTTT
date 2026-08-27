import os

import h5py
import numpy as np
import torch
from ocnn.dataset import CollateBatch
from ocnn.octree import Points

from .utils import Transform


def farthest_point_sample(points: np.ndarray, num_points: int) -> np.ndarray:
  r'''Samples point indices with a deterministic CPU implementation of FPS.'''
  num_input = points.shape[0]
  if num_points > num_input:
    raise ValueError(
        'Cannot sample %d points from a cloud with %d points.' %
        (num_points, num_input))

  centroids = np.empty(num_points, dtype=np.int64)
  distance = np.full(num_input, np.inf, dtype=np.float32)
  farthest = 0
  for i in range(num_points):
    centroids[i] = farthest
    centroid = points[farthest]
    dist = np.sum((points - centroid) ** 2, axis=1)
    distance = np.minimum(distance, dist)
    farthest = int(np.argmax(distance))
  return centroids


class ScanObjectNNTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)
    self.split = flags.split.lower()
    self.num_points = int(flags.num_points)
    self.sampling = flags.sampling.lower()
    self.pointbert_candidate_points = int(flags.pointbert_candidate_points)
    self.normalize = bool(flags.normalize)
    self.cache_fps = bool(flags.cache_fps)
    self._fps_cache = {}

    valid_sampling = ('none', 'fps', 'pointbert')
    if self.sampling not in valid_sampling:
      raise ValueError(
          'Unknown ScanObjectNN sampling mode %r. Choose from %s.' %
          (self.sampling, valid_sampling))

  def _fps_indices(self, points: np.ndarray, num_points: int,
                   idx: int) -> np.ndarray:
    key = (idx, num_points)
    indices = self._fps_cache.get(key)
    if indices is None:
      indices = farthest_point_sample(points, num_points).astype(np.int32)
      if self.cache_fps:
        self._fps_cache[key] = indices
    return indices

  def _sample_points(self, points: np.ndarray, idx: int) -> np.ndarray:
    num_input = points.shape[0]

    if self.sampling == 'none':
      if self.num_points != num_input:
        raise ValueError(
            'sampling=none keeps every input point, so num_points must be %d '
            'instead of %d.' % (num_input, self.num_points))
      return points

    if self.sampling == 'fps':
      indices = self._fps_indices(points, self.num_points, idx)
      return points[indices]

    # Reproduce Point-BERT's sampling schedule. During training it first uses
    # FPS to select 1200 candidates and randomly keeps 1024 of them. During
    # testing it directly uses FPS to select 1024 points.
    if self.split == 'train':
      candidate_num = min(self.pointbert_candidate_points, num_input)
      if self.num_points > candidate_num:
        raise ValueError(
            'num_points (%d) must not exceed pointbert_candidate_points (%d).'
            % (self.num_points, candidate_num))
      candidate_idx = self._fps_indices(points, candidate_num, idx)
      selected = np.random.choice(candidate_num, self.num_points, replace=False)
      return points[candidate_idx[selected]]

    indices = self._fps_indices(points, self.num_points, idx)
    return points[indices]

  def _normalize_points(self, points: np.ndarray) -> np.ndarray:
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points = points - center
    scale = np.abs(points).max()
    if scale > 1.0e-6:
      points = points / scale * 0.95
    return points

  def preprocess(self, sample: dict, idx: int):
    sample = sample.copy()
    points = np.asarray(sample['points'], dtype=np.float32)
    points = self._sample_points(points, idx)
    if self.normalize:
      points = self._normalize_points(points)
    points = np.ascontiguousarray(points, dtype=np.float32)
    # ScanObjectNN only provides XYZ. Construct Points without fake normals;
    # the ScanObjectNN configs use the position feature `P` exclusively.
    return Points(torch.from_numpy(points))


class ScanObjectNNDataset(torch.utils.data.Dataset):

  def __init__(self, h5_file: str, transform, take: int = -1):
    super().__init__()
    self.h5_file = os.path.abspath(h5_file)
    self.transform = transform
    self._h5 = None

    if not os.path.isfile(self.h5_file):
      raise FileNotFoundError('ScanObjectNN file not found: ' + self.h5_file)

    with h5py.File(self.h5_file, 'r') as h5:
      if 'data' not in h5 or 'label' not in h5:
        raise KeyError(
            'ScanObjectNN HDF5 must contain both `data` and `label`: ' +
            self.h5_file)
      if h5['data'].ndim != 3 or h5['data'].shape[-1] != 3:
        raise ValueError(
            'Expected ScanObjectNN data with shape [N, P, 3], got %s.' %
            (h5['data'].shape,))
      if h5['data'].shape[0] != h5['label'].shape[0]:
        raise ValueError('The number of ScanObjectNN samples and labels differs.')
      num_samples = int(h5['data'].shape[0])

    self.num_samples = num_samples if take < 1 else min(take, num_samples)

  def __len__(self):
    return self.num_samples

  def _get_h5(self):
    # Each DataLoader worker opens its own read-only handle. Keeping HDF5 handles
    # out of pickle avoids sharing an unsafe handle across worker processes.
    if self._h5 is None:
      self._h5 = h5py.File(self.h5_file, 'r')
    return self._h5

  def __getitem__(self, idx):
    h5 = self._get_h5()
    sample = {'points': np.asarray(h5['data'][idx], dtype=np.float32)}
    output = self.transform(sample, idx)
    output['label'] = int(np.asarray(h5['label'][idx]).reshape(-1)[0])
    output['filename'] = '%s:%d' % (os.path.basename(self.h5_file), idx)
    return output

  def __getstate__(self):
    state = self.__dict__.copy()
    state['_h5'] = None
    return state

  def __del__(self):
    h5 = getattr(self, '_h5', None)
    if h5 is not None:
      try:
        h5.close()
      except (AttributeError, TypeError):
        # Python may tear h5py internals down before dataset finalizers run.
        pass


def get_scanobjectnn_dataset(flags):
  transform = ScanObjectNNTransform(flags)
  dataset = ScanObjectNNDataset(
      flags.h5_file, transform, take=getattr(flags, 'take', -1))
  collate_batch = CollateBatch()
  return dataset, collate_batch
