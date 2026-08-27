import json
import math
import os
import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from ocnn.octree import Points


# Official Pointcept v1.7 / Utonia PartNetE category order.  Every category
# owns one additional ``other`` label, so the 103 named parts form 148 global
# output classes together with the 45 category-specific ``other`` labels.
PARTNETE_CATEGORIES: Tuple[str, ...] = (
    'Scissors', 'Lighter', 'Box', 'Camera', 'StorageFurniture', 'Safe',
    'Toilet', 'Chair', 'Oven', 'USB', 'Remote', 'Switch', 'Laptop', 'Phone',
    'Bottle', 'Mouse', 'Table', 'Keyboard', 'Eyeglasses', 'Faucet',
    'KitchenPot', 'Knife', 'Window', 'Pen', 'WashingMachine', 'Clock',
    'Refrigerator', 'Pliers', 'Microwave', 'Toaster', 'Printer', 'Kettle',
    'TrashCan', 'Door', 'Cart', 'Dishwasher', 'Suitcase', 'Dispenser',
    'Display', 'Bucket', 'Lamp', 'Globe', 'Stapler', 'CoffeeMachine',
    'FoldingChair')

PARTNETE_NUM_PARTS: Tuple[int, ...] = (
    4, 4, 2, 3, 4, 4, 4, 6, 3, 3, 2, 2, 6, 3, 2, 4, 7, 3, 3, 3, 3, 2,
    2, 3, 3, 2, 3, 2, 5, 3, 2, 4, 4, 4, 2, 3, 3, 3, 4, 2, 5, 2, 3, 5, 2)

PARTNETE_PART_OFFSETS: Tuple[int, ...] = tuple(
    np.concatenate(([0], np.cumsum(PARTNETE_NUM_PARTS))).tolist())
PARTNETE_NUM_CLASSES = int(sum(PARTNETE_NUM_PARTS))

# Pointcept's final PartNetE tester applies these ten augmentation branches.
# The second branch applies RandomFlip independently on x and y with p=0.5.
PARTNETE_UTONIA_TTA: Tuple[Tuple[float, float], ...] = (
    (1.00, 0.0), (1.00, 0.5), (0.80, 0.0), (0.85, 0.0), (0.90, 0.0),
    (0.95, 0.0), (1.05, 0.0), (1.10, 0.0), (1.15, 0.0), (1.20, 0.0))


def partnete_named_part_ids(category: int) -> Tuple[int, ...]:
  if category < 0 or category >= len(PARTNETE_CATEGORIES):
    raise ValueError('Invalid PartNetE category id: %d.' % category)
  start = PARTNETE_PART_OFFSETS[category]
  end = PARTNETE_PART_OFFSETS[category + 1]
  return tuple(range(start + 1, end))  # exclude category-specific ``other``


def load_partnete_meta(meta_path: str) -> Dict[str, Sequence[str]]:
  if not os.path.isfile(meta_path):
    raise FileNotFoundError('PartNetE metadata not found: ' + meta_path)
  with open(meta_path, 'r', encoding='utf-8') as handle:
    meta = json.load(handle)
  if set(meta) != set(PARTNETE_CATEGORIES):
    missing = sorted(set(PARTNETE_CATEGORIES) - set(meta))
    extra = sorted(set(meta) - set(PARTNETE_CATEGORIES))
    raise ValueError(
        'PartNetE metadata/category mismatch; missing=%r, extra=%r.' %
        (missing, extra))
  for category, num_parts in zip(PARTNETE_CATEGORIES, PARTNETE_NUM_PARTS):
    if len(meta[category]) != num_parts - 1:
      raise ValueError(
          'PartNetE category %s expects %d named parts, got %d.' %
          (category, num_parts - 1, len(meta[category])))
  return meta


def _load_object(path: str, category: int) -> Dict[str, np.ndarray]:
  arrays = {}
  specs = {
      'coord': (np.float32, 3),
      'normal': (np.float32, 3),
      'color': (np.float32, 3),
      'segment': (np.int64, None),
  }
  for key, (dtype, channels) in specs.items():
    filename = os.path.join(path, key + '.npy')
    if not os.path.isfile(filename):
      raise FileNotFoundError('Missing PartNetE asset: ' + filename)
    value = np.load(filename, allow_pickle=False)
    if key == 'segment':
      value = value.reshape(-1)
    elif value.ndim != 2 or value.shape[1] != channels:
      raise ValueError('%s must have shape [N, %d], got %s.' %
                       (filename, channels, value.shape))
    arrays[key] = np.asarray(value, dtype=dtype)

  npt = arrays['coord'].shape[0]
  if npt == 0 or any(value.shape[0] != npt for value in arrays.values()):
    raise ValueError('Unaligned or empty PartNetE object: ' + path)
  if not np.isfinite(arrays['coord']).all() or \
      not np.isfinite(arrays['normal']).all():
    raise ValueError('PartNetE coordinates/normals contain NaN or Inf: ' + path)

  # Raw PartSLIP labels use -1 for ``other`` and 0..K-1 for named parts.
  raw_segment = arrays['segment']
  max_named = PARTNETE_NUM_PARTS[category] - 2
  if raw_segment.min() < -1 or raw_segment.max() > max_named:
    raise ValueError(
        'Invalid local PartNetE labels [%d, %d] for category %s in %s.' %
        (raw_segment.min(), raw_segment.max(),
         PARTNETE_CATEGORIES[category], path))
  arrays['segment'] = np.ascontiguousarray(
      raw_segment + PARTNETE_PART_OFFSETS[category] + 1, dtype=np.int64)
  return arrays


def _select(data: Dict[str, np.ndarray], index: np.ndarray):
  npt = data['coord'].shape[0]
  return {
      key: np.ascontiguousarray(value[index])
      for key, value in data.items()
      if isinstance(value, np.ndarray) and value.shape[:1] == (npt,)
  }


def _center_shift(coord: np.ndarray, apply_z: bool):
  coord = np.asarray(coord, dtype=np.float32).copy()
  lo, hi = coord.min(0), coord.max(0)
  shift = np.asarray([
      (lo[0] + hi[0]) * 0.5,
      (lo[1] + hi[1]) * 0.5,
      lo[2] if apply_z else 0.0], dtype=np.float32)
  coord -= shift
  return coord


def _rotation_matrix(angle: float, axis: str):
  cosine, sine = math.cos(angle), math.sin(angle)
  if axis == 'x':
    matrix = [[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]]
  elif axis == 'y':
    matrix = [[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]]
  elif axis == 'z':
    matrix = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
  else:
    raise ValueError('Unknown rotation axis: ' + axis)
  return np.asarray(matrix, dtype=np.float32)


def _random_rotate(data, low, high, axis, probability=0.5, center=None):
  if random.random() > probability:
    return
  matrix = _rotation_matrix(np.random.uniform(low, high) * np.pi, axis)
  if center is None:
    lo, hi = data['coord'].min(0), data['coord'].max(0)
    center = (lo + hi) * 0.5
  center = np.asarray(center, dtype=np.float32)
  data['coord'] = (data['coord'] - center) @ matrix.T + center
  data['normal'] = data['normal'] @ matrix.T


def _fnv_hash(grid_coord: np.ndarray):
  hashed = np.full(
      grid_coord.shape[0], np.uint64(14695981039346656037), dtype=np.uint64)
  for axis in range(grid_coord.shape[1]):
    hashed *= np.uint64(1099511628211)
    hashed = np.bitwise_xor(
        hashed, grid_coord[:, axis].astype(np.uint64, copy=False))
  return hashed


def _grid_groups(coord: np.ndarray, grid_size: float):
  grid_coord = np.floor(coord / grid_size).astype(np.int64)
  grid_coord -= grid_coord.min(0)
  key = _fnv_hash(grid_coord)
  idx_sort = np.argsort(key)
  key_sort = key[idx_sort]
  _, inverse_sort, count = np.unique(
      key_sort, return_inverse=True, return_counts=True)
  starts = np.cumsum(np.insert(count, 0, 0)[:-1])
  inverse = np.empty_like(inverse_sort)
  inverse[idx_sort] = inverse_sort
  return idx_sort, starts, count, inverse


def _grid_sample_train(data, grid_size):
  idx_sort, starts, count, _ = _grid_groups(data['coord'], grid_size)
  offsets = np.random.randint(0, count.max(), count.size) % count
  return _select(data, idx_sort[starts + offsets])


def _crop_nearest(data, point_max):
  npt = data['coord'].shape[0]
  if point_max < 1 or npt <= point_max:
    return data
  center = data['coord'][np.random.randint(npt)]
  distance = np.sum(np.square(data['coord'] - center), axis=1)
  index = np.argpartition(distance, point_max - 1)[:point_max]
  return _select(data, index)


def _elastic_distortion(coord, distortion_params):
  if random.random() >= 0.95:
    return coord
  from scipy import interpolate, ndimage

  coord = coord.copy()
  for granularity, magnitude in distortion_params:
    blur_x = np.ones((3, 1, 1, 1), dtype=np.float32) / 3.0
    blur_y = np.ones((1, 3, 1, 1), dtype=np.float32) / 3.0
    blur_z = np.ones((1, 1, 3, 1), dtype=np.float32) / 3.0
    coord_min = coord.min(0)
    noise_dim = ((coord - coord_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)
    for _ in range(2):
      noise = ndimage.convolve(noise, blur_x, mode='constant', cval=0)
      noise = ndimage.convolve(noise, blur_y, mode='constant', cval=0)
      noise = ndimage.convolve(noise, blur_z, mode='constant', cval=0)
    axes = [
        np.linspace(lo - granularity,
                    lo + granularity * (size - 2), size)
        for lo, size in zip(coord_min, noise_dim)]
    interpolator = interpolate.RegularGridInterpolator(
        axes, noise, bounds_error=False, fill_value=0)
    coord += interpolator(coord).astype(np.float32) * float(magnitude)
  return coord


def _normalize_octree_coord(coord, scale_factor, octree_bound):
  coord = np.asarray(coord / scale_factor, dtype=np.float32)
  max_abs = float(np.abs(coord).max()) if coord.size else 0.0
  if max_abs > octree_bound:
    coord *= octree_bound / max_abs
  return np.ascontiguousarray(coord, dtype=np.float32)


def _make_points(data, scale_factor, octree_bound):
  coord = _center_shift(data['coord'], apply_z=False)
  coord = _normalize_octree_coord(coord, scale_factor, octree_bound)
  normal = np.ascontiguousarray(data['normal'], dtype=np.float32)
  color = np.ascontiguousarray(data['color'] / 255.0, dtype=np.float32)
  segment = np.ascontiguousarray(data['segment'], dtype=np.int64)
  return Points(
      torch.from_numpy(coord), normals=torch.from_numpy(normal),
      features=torch.from_numpy(color), labels=torch.from_numpy(segment))


class PartNetETrainTransform:

  def __init__(self, flags):
    self.grid_size = float(getattr(flags, 'voxel_size', 0.01))
    self.point_max = int(getattr(flags, 'max_npt', 102400))
    self.scale_factor = float(getattr(flags, 'scale_factor', 3.4))
    self.octree_bound = float(getattr(flags, 'octree_bound', 0.999))
    self.distort = bool(flags.distort)

  def __call__(self, source):
    data = {key: value.copy() for key, value in source.items()}
    data['coord'] = _center_shift(data['coord'], apply_z=True)
    if self.distort:
      npt = data['coord'].shape[0]
      keep = np.random.choice(npt, int(npt * 0.8), replace=False)
      data = _select(data, keep)
      _random_rotate(data, -1.0, 1.0, 'z', center=(0, 0, 0))
      _random_rotate(data, -1.0 / 64.0, 1.0 / 64.0, 'x')
      _random_rotate(data, -1.0 / 64.0, 1.0 / 64.0, 'y')
      data['coord'] *= np.random.uniform(0.9, 1.1)
      for axis in (0, 1):
        if np.random.rand() < 0.5:
          data['coord'][:, axis] *= -1.0
          data['normal'][:, axis] *= -1.0
      jitter = np.clip(
          np.random.randn(*data['coord'].shape) * 0.005, -0.02, 0.02)
      data['coord'] += jitter.astype(np.float32)
      data['coord'] = _elastic_distortion(
          data['coord'], ((0.2, 0.4), (0.8, 1.6)))

    data = _grid_sample_train(data, self.grid_size)
    data = _crop_nearest(data, self.point_max)
    return _make_points(data, self.scale_factor, self.octree_bound)


def partnete_test_augmentations(enable_tta: bool):
  return PARTNETE_UTONIA_TTA if enable_tta else ((1.0, 0.0),)


def build_partnete_test_variant(source, flags, scale, flip_probability,
                                exhaustive=False):
  r'''Builds Pointcept-style voxel fragments for one PartNetE test vote.

  Periodic validation uses one representative per voxel plus an inverse map.
  The final Utonia pass uses GridSample(test), which enumerates every voxel
  occupancy layer and therefore predicts every original point.
  '''
  if scale <= 0.0 or not 0.0 <= flip_probability <= 1.0:
    raise ValueError('Invalid PartNetE test augmentation parameters.')
  grid_size = float(getattr(flags, 'voxel_size', 0.01))
  fragment_max = int(getattr(flags, 'fragment_max_npt', 102400))
  data = {key: value.copy() for key, value in source.items()
          if isinstance(value, np.ndarray)}
  data['coord'] = _center_shift(data['coord'], apply_z=True)
  data['coord'] *= float(scale)
  for axis in (0, 1):
    if flip_probability > 0.0 and np.random.rand() < flip_probability:
      data['coord'][:, axis] *= -1.0
      data['normal'][:, axis] *= -1.0

  idx_sort, starts, count, inverse = _grid_groups(data['coord'], grid_size)
  if exhaustive:
    layers = [idx_sort[starts + layer % count]
              for layer in range(int(count.max()))]
    representative = None
    inverse = None
  else:
    representative = idx_sort[starts]
    layers = [representative]

  fragments = []
  for layer in layers:
    if fragment_max > 0 and layer.size > fragment_max:
      fragments.extend(
          layer[start:start + fragment_max]
          for start in range(0, layer.size, fragment_max))
    else:
      fragments.append(layer)
  return data, fragments, representative, inverse


def make_partnete_points(data, index, flags):
  selected = _select(data, np.asarray(index, dtype=np.int64))
  return _make_points(
      selected, float(getattr(flags, 'scale_factor', 3.4)),
      float(getattr(flags, 'octree_bound', 0.999)))


class PartNetEDataset(torch.utils.data.Dataset):

  def __init__(self, root: str, split: str, meta_path: str, transform=None,
               test_mode=False, take: int = -1):
    super().__init__()
    self.root = os.path.abspath(root)
    self.split = str(split)
    self.transform = transform
    self.test_mode = bool(test_mode)
    self.meta = load_partnete_meta(meta_path)
    if self.split not in ('few_shot', 'test'):
      raise ValueError('PartNetE split must be few_shot or test.')
    split_root = os.path.join(self.root, self.split)
    if not os.path.isdir(split_root):
      raise FileNotFoundError('PartNetE split not found: ' + split_root)

    self.objects = []
    for category, category_name in enumerate(PARTNETE_CATEGORIES):
      category_root = os.path.join(split_root, category_name)
      if not os.path.isdir(category_root):
        raise FileNotFoundError('PartNetE category not found: ' + category_root)
      for object_id in sorted(os.listdir(category_root)):
        object_path = os.path.join(category_root, object_id)
        if os.path.isdir(object_path):
          self.objects.append((category, category_name, object_id, object_path))
    if take > 0:
      self.objects = self.objects[:int(take)]
    if not self.objects:
      raise ValueError('PartNetE split is empty: ' + split_root)

  def __len__(self):
    return len(self.objects)

  def __getitem__(self, idx):
    category, category_name, object_id, path = self.objects[idx]
    data = _load_object(path, category)
    filename = category_name + '_' + object_id
    if self.test_mode:
      data.update({
          'category': category,
          'category_name': category_name,
          'filename': filename,
      })
      return data
    points = self.transform(data)
    return {'points': points, 'label': category, 'filename': filename}


class PartNetECollate:

  def __init__(self, test_mode=False):
    self.test_mode = bool(test_mode)

  def __call__(self, batch):
    if self.test_mode:
      if len(batch) != 1:
        raise ValueError('PartNetE full-object evaluation requires batch_size=1.')
      output = batch[0]
      output['_partnete_test'] = True
      return output
    return {key: [sample[key] for sample in batch]
            if key != 'label' else torch.as_tensor(
                [sample[key] for sample in batch], dtype=torch.long)
            for key in batch[0].keys()}


def get_partnete_dataset(flags):
  test_mode = bool(getattr(flags, 'test_mode', False))
  transform = None if test_mode else PartNetETrainTransform(flags)
  dataset = PartNetEDataset(
      flags.location, flags.split, flags.meta_path, transform, test_mode,
      take=getattr(flags, 'take', -1))
  return dataset, PartNetECollate(test_mode)
