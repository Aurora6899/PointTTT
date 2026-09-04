
import os

import ocnn

import datasets
import models

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def point_ttt_seg_large(in_channels, out_channels, nempty=True, **kwargs):
  return models.PointTTTSeg(
      in_channels, out_channels,
      channels=[192, 384, 768, 768],
      num_blocks=[2, 2, 18, 2],
      drop_path=0.5, nempty=nempty,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5],
      **kwargs)


def point_ttt_seg_base(in_channels, out_channels, nempty=True, **kwargs):
  return models.PointTTTSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 18, 2],
      drop_path=0.5, nempty=nempty,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5],
      **kwargs)


def point_ttt_seg_small(in_channels, out_channels, nempty=True, **kwargs):
  return models.PointTTTSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 9, 2],
      drop_path=0.5, nempty=nempty,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5],
      **kwargs)


def point_ttt_cls(in_channels, out_channels, nempty, **kwargs):
  return models.PointTTTCls(
      in_channels, out_channels,
      channels=[192],
      num_blocks=[2],
      drop_path=0.3, nempty=nempty,
      stem_down=2, head_drop=0.5,
      **kwargs)


def get_segmentation_model(flags):
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
      'partition_by_batch': bool(getattr(flags, 'partition_by_batch', False)),
      'ttt_base_lr': float(getattr(flags, 'ttt_base_lr', 1.0)),
      'ttt_update_train': bool(getattr(flags, 'ttt_update_train', True)),
      'ttt_update_test': bool(getattr(flags, 'ttt_update_test', True)),
      'ttt_patch_size': int(getattr(flags, 'ttt_patch_size', 64)),
      'ttt_num_heads': int(getattr(flags, 'ttt_num_heads', 24)),
      'ttt_layer_type': str(getattr(flags, 'ttt_layer_type', 'linear')),
      'pointttt_hierarchical_enabled': bool(getattr(
          flags, 'pointttt_hierarchical_enabled', False)),
      'pointttt_hierarchical_stages': list(getattr(
          flags, 'pointttt_hierarchical_stages', [])),
      'pointttt_hierarchical_block_interval': int(getattr(
          flags, 'pointttt_hierarchical_block_interval', 0)),
      'pointttt_global_chunk_size': int(getattr(
          flags, 'pointttt_global_chunk_size', 128)),
      'pointttt_summary_tokens': int(getattr(
          flags, 'pointttt_summary_tokens', 1)),
      'pointttt_global_bidirectional': bool(getattr(
          flags, 'pointttt_global_bidirectional', True)),
      'pointttt_global_gate_init': float(getattr(
          flags, 'pointttt_global_gate_init', 0.0)),
  }
  networks = {
      # 'octsegformer': octsegformer,
      # 'octsegformer_large': octsegformer_large,
      # 'octsegformer_small': octsegformer_small,
      'pointttt_seg': point_ttt_seg_base,
      'pointttt_seg_large': point_ttt_seg_large,
      'pointttt_seg_small': point_ttt_seg_small,
  }

  model_name = flags.name.lower()
  if model_name not in networks:
    raise ValueError(f'Unknown segmentation model: {flags.name}')
  return networks[model_name](**params)


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.models.LeNet(
        flags.channel, flags.nout, flags.stages, flags.nempty)
  elif flags.name.lower() == 'hrnet':
    model = ocnn.models.HRNet(
        flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
  elif flags.name.lower() == 'pointttt_cls':
    model = point_ttt_cls(
        flags.channel, flags.nout, flags.nempty,
        ttt_base_lr=float(getattr(flags, 'ttt_base_lr', 1.0)),
        ttt_update_train=bool(getattr(flags, 'ttt_update_train', True)),
        ttt_update_test=bool(getattr(flags, 'ttt_update_test', True)))
  else:
    raise ValueError(f'Unknown classification model: {flags.name}')
  return model


def get_classification_dataset(flags):
  name = flags.name.lower()
  if name == 'modelnet40':
    return datasets.get_modelnet40_dataset(flags)
  elif name == 'scanobjectnn':
    return datasets.get_scanobjectnn_dataset(flags)
  elif name == 'shapenet55':
    return datasets.get_shapenet55_dataset(flags)
  else:
    raise ValueError('Unknown classification dataset: ' + flags.name)


def get_segmentation_dataset(flags):
  if flags.name.lower() in ('shapenetpart', 'shapenet'):
    return datasets.get_shapenetpart_dataset(flags)
  elif flags.name.lower() in ('partnete', 'partnet_e'):
    return datasets.get_partnete_dataset(flags)
  elif flags.name.lower() == 'scannet':
    return datasets.get_scannet_dataset(flags)
  elif flags.name.lower() == 's3dis':
    return datasets.get_s3dis_dataset(flags)
  elif flags.name.lower() in ('semantickitti', 'semantic_kitti'):
    return datasets.get_semantickitti_dataset(flags)
  elif flags.name.lower() == 'kitti':
    return datasets.get_kitti_dataset(flags)
  else:
    raise ValueError('Unknown segmentation dataset: ' + flags.name)
