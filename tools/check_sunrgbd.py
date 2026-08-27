#!/usr/bin/env python
"""Read-only validation for MMDetection3D-formatted SUN RGB-D data."""

import argparse
import pickle
from collections import Counter
from pathlib import Path


CLASSES = (
    'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
    'night_stand', 'bookshelf', 'bathtub')
EXPECTED_SPLIT_SIZES = {'train': 5285, 'val': 5050}
POINT_BYTES = 100000 * 6 * 4


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate SUN RGB-D paths, splits and point shapes.')
    parser.add_argument(
        '--data-root', default='data/sunrgbd', type=Path)
    parser.add_argument(
        '--train-info', default='sunrgbd_infos_train（复件）.pkl',
        help='training info filename relative to data root')
    parser.add_argument(
        '--val-info', default='sunrgbd_infos_val.pkl',
        help='validation info filename relative to data root')
    parser.add_argument(
        '--full', action='store_true',
        help='check every point file size (default: first 32 per split)')
    return parser.parse_args()


def load_infos(path):
    # MMDetection3D info files are trusted pickle artifacts produced during
    # preprocessing. Do not run this checker on an untrusted downloaded pkl.
    with path.open('rb') as stream:
        infos = pickle.load(stream)
    if not isinstance(infos, list):
        raise TypeError(f'{path}: expected list, got {type(infos).__name__}')
    return infos


def check_split(root, split, info_name, full):
    info_path = root / info_name
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    infos = load_infos(info_path)
    expected = EXPECTED_SPLIT_SIZES[split]
    if len(infos) != expected:
        raise ValueError(
            f'{split}: expected {expected} scenes, found {len(infos)}')

    point_paths = []
    class_counts = Counter()
    for index, info in enumerate(infos):
        required = {'point_cloud', 'pts_path', 'annos'}
        missing = required.difference(info)
        if missing:
            raise KeyError(f'{split}[{index}] missing keys: {sorted(missing)}')
        if info['point_cloud'].get('num_features') != 6:
            raise ValueError(f'{split}[{index}] does not declare 6 features')
        point_paths.append(root / info['pts_path'])
        class_counts.update(map(str, info['annos'].get('name', [])))

    paths_to_check = point_paths if full else point_paths[:32]
    for point_path in paths_to_check:
        if not point_path.is_file():
            raise FileNotFoundError(point_path)
        if point_path.stat().st_size != POINT_BYTES:
            raise ValueError(
                f'{point_path}: expected {POINT_BYTES} bytes '
                '(100000 x 6 float32)')

    target_boxes = sum(class_counts[name] for name in CLASSES)
    scene_ids = {path.stem for path in point_paths}
    print(
        f'{split}: {len(infos)} scenes, {target_boxes} target boxes, '
        f'{len(paths_to_check)} point files checked')
    return scene_ids


def main():
    args = parse_args()
    root = args.data_root.resolve()
    train_ids = check_split(root, 'train', args.train_info, args.full)
    val_ids = check_split(root, 'val', args.val_info, args.full)
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise ValueError(
            f'train/val leakage: {len(overlap)} duplicated scene ids')
    print(
        f'OK: official 5285/5050 split, 10 OctFormer classes, '
        f'{len(train_ids) + len(val_ids)} distinct scenes')


if __name__ == '__main__':
    main()
