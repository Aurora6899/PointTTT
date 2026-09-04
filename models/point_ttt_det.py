"""PointTTT backbone adapter for the OctFormer/FCAF3D detection pipeline.

This module is intentionally not imported from ``models/__init__.py``.  It is
loaded only by ``detection.py``, keeping MMDetection3D and MinkowskiEngine
optional for the existing classification and segmentation entry points.
"""

import torch
import torch.distributed as dist
import ocnn
import MinkowskiEngine as ME
from mmcv.utils import print_log
from ocnn.octree import Octree, Points

from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head
from mmdet3d.models.builder import BACKBONES
from mmdet3d.models.detectors import Base3DDetector

from .point_ttt import PointTTT


@BACKBONES.register_module()
class PointTTTDetectionBackbone(PointTTT):
    """Detection-only registry adapter around the current PointTTT backbone."""

    def __init__(self, *args, partition_by_batch=True, **kwargs):
        # FCAF3D trains with several scenes per GPU.  Scene-aware partitioning
        # is required because a flat TTT chunk must never span two point clouds.
        super().__init__(
            *args, partition_by_batch=partition_by_batch, **kwargs)


@DETECTORS.register_module()
class PointTTTSingleStage3DDetector(Base3DDetector):
    """Octree-to-Minkowski adapter with the official FCAF3D detection head."""

    def __init__(self, backbone, head, voxel_size, octree_depth,
                 octree_feature='F', train_cfg=None, test_cfg=None,
                 init_cfg=None, pretrained=None):
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        head.update(train_cfg=train_cfg)
        head.update(test_cfg=test_cfg)
        self.head = build_head(head)
        self.voxel_size = voxel_size
        self.octree_depth = octree_depth
        self.scale_factor = 2.0 / (2 ** octree_depth * voxel_size)
        self.octree_feature = octree_feature
        self.init_weights()
        self._report_parameter_count()

    def _report_parameter_count(self):
        """Print parameter statistics once in a distributed launch."""
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        backbone_params = sum(
            parameter.numel() for parameter in self.backbone.parameters())
        head_params = sum(
            parameter.numel() for parameter in self.head.parameters())
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.parameters()
            if parameter.requires_grad)
        print_log(
            'PointTTT SUN RGB-D parameter count: '
            f'backbone={backbone_params / 1e6:.2f}M, '
            f'FCAF3D head={head_params / 1e6:.2f}M, '
            f'total={total_params / 1e6:.2f}M, '
            f'trainable={trainable_params / 1e6:.2f}M',
            logger='mmdet')

    def build_octree(self, raw_points):
        """Build one octree per scene and merge them without mixing samples."""
        octrees = []
        for points in raw_points:
            xyz = points[:, :3] * self.scale_factor
            color = points[:, 3:]
            point_cloud = Points(xyz, features=color)
            point_cloud.clip(min=-1.0, max=1.0)

            octree = Octree(self.octree_depth, device=xyz.device)
            octree.build_octree(point_cloud)
            octrees.append(octree)

        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        features = ocnn.modules.InputFeature(
            self.octree_feature, nempty=True)(octree)
        return features, octree

    def extract_feat(self, points):
        features, octree = self.build_octree(points)
        feature_dict = self.backbone(
            features, octree, self.octree_depth)
        feature_list = list(feature_dict.values())
        depths = list(feature_dict.keys())

        sparse_features = []
        coordinate_manager = None
        metric_scale = self.scale_factor * self.voxel_size
        for stage, (feature, depth) in enumerate(zip(feature_list, depths)):
            x, y, z, batch = octree.xyzb(depth, nempty=True)
            xyz = torch.stack([x, y, z], dim=-1)
            xyz = (xyz / (2 ** (depth - 1)) - 1) / metric_scale
            coordinates = torch.cat(
                [batch.unsqueeze(1), xyz], dim=1).int()
            sparse = ME.SparseTensor(
                coordinates=coordinates,
                features=feature,
                tensor_stride=2 ** (stage + 3),
                coordinate_manager=coordinate_manager)
            sparse_features.append(sparse)
            if coordinate_manager is None:
                coordinate_manager = sparse.coordinate_manager
        return sparse_features

    def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
        features = self.extract_feat(points)
        return self.head.forward_train(
            features, gt_bboxes_3d, gt_labels_3d, img_metas)

    def simple_test(self, points, img_metas, *args, **kwargs):
        features = self.extract_feat(points)
        bbox_list = self.head.forward_test(features, img_metas)
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

    def aug_test(self, points, img_metas, **kwargs):
        raise NotImplementedError(
            'The official OctFormer SUN RGB-D protocol uses single-scale '
            'evaluation without test-time augmentation.')
