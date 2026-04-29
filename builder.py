
import ocnn
import torch
import datasets
import models

import os

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'



def PointTTTSeg_base(in_channels, out_channels, **kwargs):
  return models.PointTTTSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[6, 9, 18, 6],
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def PointTTTSeg_small(in_channels, out_channels, **kwargs):
  return models.PointTTTSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 9, 2],
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def PointTTT_cls(in_channels, out_channels, nempty, **kwargs):
  return models.PointTTTCls(
      in_channels, out_channels,
      channels=[192],
      num_blocks=[2],
      drop_path=0.3, nempty=nempty,
      stem_down=2, head_drop=0.5)


def get_segmentation_model(flags):
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
  }
  networks = {
      'pointttt_seg': PointTTTSeg_base,
      'pointttt_seg_small': PointTTTSeg_small,
  }

  return networks[flags.name.lower()](**params)


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.models.LeNet(
        flags.channel, flags.nout, flags.stages, flags.nempty)
  elif flags.name.lower() == 'hrnet':
    model = ocnn.models.HRNet(
        flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
  elif flags.name.lower() == 'pointttt_cls':
    model = PointTTT_cls(flags.channel, flags.nout, flags.nempty)
  else:
    raise ValueError
  return model


def get_segmentation_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_seg_dataset(flags)
  elif flags.name.lower() == 'scannet':
    return datasets.get_scannet_dataset(flags)
  else:
    raise ValueError
