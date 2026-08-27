# PointTTT + FCAF3D on SUN RGB-D.
#
# Dataset protocol, augmentation, head, losses and metrics follow the official
# OctFormer configuration.  For two GPUs we retain the official micro-batch of
# 8 scenes/GPU and accumulate 2 iterations: 2 x 8 x 2 = effective batch 32,
# equal to OctFormer's 4 x 8 training batch without doubling GPU memory.

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/pointttt_sunrgbd'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

model = dict(
    type='PointTTTSingleStage3DDetector',
    voxel_size=0.01,
    octree_depth=12,
    backbone=dict(
        type='PointTTTDetectionBackbone',
        in_channels=3,
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        drop_path=0.5,
        stem_down=3,
        partition_by_batch=True),
    head=dict(
        type='FCAF3DHead',
        in_channels=(96, 192, 384, 384),
        out_channels=128,
        voxel_size=0.01,
        pts_prune_threshold=100000,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        n_classes=10,
        n_reg_outs=8,
        bbox_loss=dict(type='RotatedIoU3DLoss')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))

n_points = 100000
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
# The canonical-named train pkl in the current dataset overlaps validation by
# 2200 scenes.  This existing copy contains OctFormer's official 5051--10335
# training split and is deliberately used without overwriting user data.
train_ann_file = data_root + 'sunrgbd_infos_train.pkl'
val_ann_file = data_root + 'sunrgbd_infos_val.pkl'
class_names = (
    'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
    'night_stand', 'bookshelf', 'bathtub')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(type='RandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=train_ann_file,
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        samples_per_gpu=8),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        samples_per_gpu=8))

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2,
    grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)
evaluation = dict(interval=1, iou_thr=(0.25, 0.5))
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# This entry is informational in distributed mode; torch.distributed.launch
# sets the actual world size.  It also makes config dumps self-describing.
gpu_ids = range(2)
