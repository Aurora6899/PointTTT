import bisect
import os
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from ocnn.dataset import CollateBatch
from ocnn.octree import Points


# The category order and global part ids are the official ShapeNetPart HDF5
# convention used by PointNet/Point-BERT style evaluations.
SHAPENETPART_CATEGORIES: Tuple[str, ...] = (
    'Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
    'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard',
    'Table')

SHAPENETPART_PARTS: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (4, 5),
    (6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18),
    (19, 20, 21),
    (22, 23),
    (24, 25, 26, 27),
    (28, 29),
    (30, 31, 32, 33, 34, 35),
    (36, 37),
    (38, 39, 40),
    (41, 42, 43),
    (44, 45, 46),
    (47, 48, 49),
)


# This is the test-time augmentation recipe used by the official Pointcept
# v1.7 Utonia ShapeNetPart configs. The second vote applies RandomFlip along
# the x axis independently to every shape with probability 0.5.
SHAPENETPART_UTONIA_TTA: Tuple[Tuple[float, float], ...] = (
    (1.00, 0.0),
    (1.00, 0.5),
    (0.80, 0.0),
    (0.85, 0.0),
    (0.90, 0.0),
    (0.95, 0.0),
    (1.05, 0.0),
    (1.10, 0.0),
    (1.15, 0.0),
    (1.20, 0.0),
)


def build_shapenetpart_tta_points(
    points_list: Sequence[Points], scale: float, flip_probability: float,
    octree_bound: float) -> List[Points]:
  r'''Builds one Utonia-style ShapeNetPart test-time augmentation vote.

  O-CNN requires every coordinate to remain in ``[-1, 1]``. As in the
  regular ShapeNetPart transform, an overflowing shape is uniformly shrunk;
  no point is clipped or dropped, so predictions stay aligned across votes.
  '''
  if scale <= 0.0:
    raise ValueError('ShapeNetPart TTA scale must be positive.')
  if not (0.0 <= flip_probability <= 1.0):
    raise ValueError('ShapeNetPart TTA flip probability must be in [0, 1].')
  if not (0.0 < octree_bound < 1.0):
    raise ValueError('octree_bound must be in (0, 1).')

  augmented = []
  for points in points_list:
    xyz = points.points.clone() * float(scale)
    flip = flip_probability > 0.0 and np.random.rand() < flip_probability
    if flip:
      xyz[:, 0].mul_(-1.0)

    max_abs = float(xyz.abs().max().item())
    if max_abs > octree_bound:
      xyz.mul_(octree_bound / max_abs)

    normals = None if points.normals is None else points.normals.clone()
    if normals is not None and flip:
      normals[:, 0].mul_(-1.0)
    features = None if points.features is None else points.features.clone()
    labels = None if points.labels is None else points.labels.clone()
    augmented.append(Points(
        xyz, normals=normals, features=features, labels=labels))
  return augmented


def _read_h5_file_list(root: str, split: str) -> List[str]:
  split_file = os.path.join(root, split + '_hdf5_file_list.txt')
  if not os.path.isfile(split_file):
    raise FileNotFoundError('ShapeNetPart split list not found: ' + split_file)

  paths = []
  with open(split_file, 'r') as fid:
    for line in fid:
      filename = line.strip()
      if not filename:
        continue
      # Some releases store absolute/relative paths in the lists, while the
      # common Stanford HDF5 release stores basenames only.
      candidates = [filename, os.path.join(root, filename),
                    os.path.join(root, os.path.basename(filename))]
      path = next((p for p in candidates if os.path.isfile(p)), None)
      if path is None:
        raise FileNotFoundError(
            'ShapeNetPart HDF5 file %r from %s was not found.' %
            (filename, split_file))
      paths.append(os.path.abspath(path))

  if not paths:
    raise ValueError('ShapeNetPart split list is empty: ' + split_file)
  return paths


class ShapeNetPartTransform:
  r'''Converts one HDF5 sample to O-CNN points without CUDA FPS.

  The default path keeps all 2048 points. Training uses Point-BERT's
  per-axis ScaleAndTranslate augmentation: scale in [2/3, 3/2] and translation
  in [-0.2, 0.2]. A final *uniform* shrink is applied only when needed so that
  every point remains inside O-CNN's [-1, 1] octree domain. This keeps point
  coordinates and part labels perfectly aligned and never clips a point.
  '''

  def __init__(self, flags):
    self.split = str(flags.split).lower()
    self.num_points = int(getattr(flags, 'num_points', 2048))
    self.sampling = str(getattr(flags, 'sampling', 'none')).lower()
    self.normalize = bool(getattr(flags, 'normalize', True))
    self.distort = bool(flags.distort)
    self.scale_low = float(getattr(flags, 'scale_low', 2.0 / 3.0))
    self.scale_high = float(getattr(flags, 'scale_high', 3.0 / 2.0))
    self.translate_range = float(getattr(flags, 'translate_range', 0.2))
    self.octree_bound = float(getattr(flags, 'octree_bound', 0.999))

    if self.num_points < 1:
      raise ValueError('ShapeNetPart num_points must be positive.')
    if self.sampling not in ('none', 'random', 'first'):
      raise ValueError(
          'Unknown ShapeNetPart sampling mode %r; choose none, random, or first.'
          % self.sampling)
    if not (0.0 < self.scale_low <= self.scale_high):
      raise ValueError('Invalid ShapeNetPart augmentation scale range.')
    if not (0.0 < self.octree_bound < 1.0):
      raise ValueError('octree_bound must be in (0, 1).')

  def _sample_indices(self, num_input: int, idx: int) -> np.ndarray:
    if self.sampling == 'none':
      if self.num_points != num_input:
        raise ValueError(
            'sampling=none keeps all %d points, but num_points=%d.' %
            (num_input, self.num_points))
      return np.arange(num_input)
    if self.num_points > num_input:
      raise ValueError('Cannot sample %d points from a cloud with %d points.' %
                       (self.num_points, num_input))
    if self.sampling == 'first':
      return np.arange(self.num_points)
    if self.split in ('train', 'trainval'):
      return np.random.permutation(num_input)[:self.num_points]
    rng = np.random.RandomState(idx)
    return rng.permutation(num_input)[:self.num_points]

  @staticmethod
  def _normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    radius = np.sqrt(np.sum(xyz * xyz, axis=1)).max()
    if radius > 1.0e-12:
      xyz = xyz / radius
    return xyz

  def __call__(self, sample: Dict[str, np.ndarray], idx: int):
    xyz = np.asarray(sample['points'], dtype=np.float32)
    part = np.asarray(sample['parts'], dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
      raise ValueError('Expected ShapeNetPart points [N, 3], got %s.' %
                       (xyz.shape,))
    if xyz.shape[0] != part.shape[0]:
      raise ValueError('ShapeNetPart points and part labels are not aligned.')
    if not np.isfinite(xyz).all():
      raise ValueError('ShapeNetPart point cloud contains NaN or Inf values.')

    indices = self._sample_indices(xyz.shape[0], idx)
    xyz, part = xyz[indices], part[indices]
    if self.normalize:
      xyz = self._normalize(xyz)

    if self.distort:
      scale = np.random.uniform(
          self.scale_low, self.scale_high, size=(1, 3)).astype(np.float32)
      shift = np.random.uniform(
          -self.translate_range, self.translate_range,
          size=(1, 3)).astype(np.float32)
      xyz = xyz * scale + shift

    # O-CNN octrees require coordinates in [-1, 1]. Uniform shrinking retains
    # shape geometry and, unlike clipping, preserves all 2048 point labels.
    max_abs = float(np.abs(xyz).max())
    if max_abs > self.octree_bound:
      xyz = xyz * (self.octree_bound / max_abs)

    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    part = np.ascontiguousarray(part, dtype=np.int64)
    points = Points(torch.from_numpy(xyz), labels=torch.from_numpy(part))
    return {'points': points}


class ShapeNetPartDataset(torch.utils.data.Dataset):
  r'''Lazy reader for ``shapenet_part_seg_hdf5_data``.

  HDF5 handles are opened independently in each DataLoader worker. This avoids
  copying the complete 16881-shape dataset into every process.
  '''

  def __init__(self, root: str, split: str, transform,
               take: int = -1):
    super().__init__()
    self.root = os.path.abspath(root)
    self.split = split.lower()
    self.transform = transform
    if not os.path.isdir(self.root):
      raise FileNotFoundError('ShapeNetPart directory not found: ' + self.root)
    if self.split not in ('train', 'val', 'trainval', 'test'):
      raise ValueError(
          'Unknown ShapeNetPart split %r; choose train, val, trainval, or test.'
          % split)

    split_names: Sequence[str] = (
        ('train', 'val') if self.split == 'trainval' else (self.split,))
    self.files = []
    for split_name in split_names:
      self.files.extend(_read_h5_file_list(self.root, split_name))

    self.file_sizes = []
    for path in self.files:
      with h5py.File(path, 'r') as h5:
        for key in ('data', 'label', 'pid'):
          if key not in h5:
            raise KeyError('%s does not contain dataset %r.' % (path, key))
        size = int(h5['data'].shape[0])
        if h5['label'].shape[0] != size or h5['pid'].shape[0] != size:
          raise ValueError('Inconsistent sample counts in ' + path)
        self.file_sizes.append(size)

    self.cumulative_sizes = np.cumsum(self.file_sizes).tolist()
    total = self.cumulative_sizes[-1]
    self.length = total if take < 1 else min(int(take), total)
    self._handles: Dict[str, h5py.File] = {}
    self._handle_pid = None

  def __len__(self):
    return self.length

  def __getstate__(self):
    state = self.__dict__.copy()
    state['_handles'] = {}
    state['_handle_pid'] = None
    return state

  def _get_handle(self, path: str):
    pid = os.getpid()
    if self._handle_pid != pid:
      self.close()
      self._handle_pid = pid
    if path not in self._handles:
      self._handles[path] = h5py.File(path, 'r')
    return self._handles[path]

  def close(self):
    for handle in getattr(self, '_handles', {}).values():
      try:
        handle.close()
      except Exception:
        pass
    self._handles = {}

  def __del__(self):
    self.close()

  def __getitem__(self, idx):
    if idx < 0:
      idx += self.length
    if idx < 0 or idx >= self.length:
      raise IndexError(idx)

    file_idx = bisect.bisect_right(self.cumulative_sizes, idx)
    file_start = 0 if file_idx == 0 else self.cumulative_sizes[file_idx - 1]
    local_idx = idx - file_start
    path = self.files[file_idx]
    h5 = self._get_handle(path)

    xyz = np.asarray(h5['data'][local_idx], dtype=np.float32)
    category = int(np.asarray(h5['label'][local_idx]).reshape(-1)[0])
    parts = np.asarray(h5['pid'][local_idx], dtype=np.int64)
    if category < 0 or category >= len(SHAPENETPART_CATEGORIES):
      raise ValueError('Invalid ShapeNetPart category id %d in %s.' %
                       (category, path))
    valid_parts = SHAPENETPART_PARTS[category]
    if not np.isin(parts, valid_parts).all():
      raise ValueError(
          'Part labels do not match category %s in %s, sample %d.' %
          (SHAPENETPART_CATEGORIES[category], path, local_idx))

    output = self.transform({'points': xyz, 'parts': parts}, idx)
    output['label'] = category
    output['filename'] = '%s:%d' % (os.path.basename(path), local_idx)
    return output


def get_shapenetpart_dataset(flags):
  transform = ShapeNetPartTransform(flags)
  dataset = ShapeNetPartDataset(
      flags.location, flags.split, transform, take=getattr(flags, 'take', -1))
  return dataset, CollateBatch()
