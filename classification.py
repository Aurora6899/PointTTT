
import torch
import torch.nn.functional as F
import ocnn

import os

from thsolver import Solver, get_config
import builder


class ClsSolver(Solver):

  @classmethod
  def update_configs(cls):
    flags = get_config()
    flags.defrost()
    flags.MODEL.ttt_base_lr = 1.0
    flags.MODEL.ttt_update_train = True
    flags.MODEL.ttt_update_test = True
    flags.freeze()

  def get_model(self, flags):
    return builder.get_classification_model(flags)

  def config_model(self):
    r'''Builds the model without dumping its complete module tree to stdout.'''
    flags = self.FLAGS.MODEL
    model = self.get_model(flags)
    model_name = model.__class__.__name__
    model.cuda(device=self.device)
    if self.world_size > 1:
      if flags.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = torch.nn.parallel.DistributedDataParallel(
          module=model, device_ids=[self.device],
          output_device=self.device, broadcast_buffers=False,
          find_unused_parameters=flags.find_unused_parameters)
    if self.is_master:
      total = sum(parameter.numel() for parameter in model.parameters())
      trainable = sum(
          parameter.numel() for parameter in model.parameters()
          if parameter.requires_grad)
      print(
          f'Model configured: {model_name} | parameters: {total / 1e6:.2f}M '
          f'| trainable: {trainable / 1e6:.2f}M')
    self.model = model

  def get_dataset(self, flags):
    return builder.get_classification_dataset(flags)

  @staticmethod
  def _extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
      raise TypeError('The pretrained checkpoint must contain a state dict.')
    for key in ('model_dict', 'state_dict', 'model'):
      value = checkpoint.get(key)
      if isinstance(value, dict):
        return value
    return checkpoint

  def load_pretrained_backbone(self, checkpoint_path):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
      raise FileNotFoundError(
          'Pretrained checkpoint not found: ' + checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = self._extract_model_state(checkpoint)
    model = self.model.module if self.world_size > 1 else self.model
    target_keys = set(model.backbone.state_dict().keys())
    backbone_dict = {}

    for key, value in state_dict.items():
      clean_key = key
      while clean_key.startswith('module.'):
        clean_key = clean_key[len('module.'):]
      if clean_key.startswith('backbone.'):
        backbone_key = clean_key[len('backbone.'):]
      elif clean_key in target_keys:
        # Also accept checkpoints saved directly from model.backbone.
        backbone_key = clean_key
      else:
        continue
      if backbone_key in target_keys:
        backbone_dict[backbone_key] = value

    if not backbone_dict:
      raise RuntimeError(
          'No PointTTT backbone weights were found in: ' + checkpoint_path)

    missing = sorted(target_keys.difference(backbone_dict))
    if missing:
      preview = ', '.join(missing[:5])
      raise RuntimeError(
          'The pretrained backbone is incomplete: %d keys are missing (%s%s).'
          % (len(missing), preview, ' ...' if len(missing) > 5 else ''))

    # Load only the backbone. The target dataset's classification head remains
    # freshly initialized (55 classes for ShapeNet55, 15 for ScanObjectNN).
    model.backbone.load_state_dict(backbone_dict, strict=True)
    if self.is_master:
      print('Load pretrained backbone: %s (%d tensors)' %
            (checkpoint_path, len(backbone_dict)))
      print('Keep the target classification head randomly initialized.')

  def _has_finetune_checkpoint(self):
    if self.FLAGS.SOLVER.ckpt:
      return True
    if not os.path.isdir(self.ckpt_dir):
      return False
    return any(filename.endswith('solver.tar')
               for filename in os.listdir(self.ckpt_dir))

  def load_checkpoint(self):
    pretrained = getattr(self.FLAGS.SOLVER, 'pretrained', '')
    # An explicit or automatically discovered fine-tuning checkpoint has
    # priority, so interrupted fine-tuning resumes with its optimizer state.
    if self._has_finetune_checkpoint() or not pretrained:
      return super().load_checkpoint()
    self.load_pretrained_backbone(pretrained)

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def forward(self, batch):

    # print('batch:', batch.keys())
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    # print('octree:', len(octree.points))

    # for i in range(len(octree.points)):
    # print(type(octree.points[0]))
    data = self.get_input_feature(octree)


    # num = random.randint(0, len(data)-1)
    # # print('data:', data[num])
    logits = self.model(data, octree, octree.depth)
    log_softmax = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_softmax, label)
    pred = torch.argmax(logits, dim=1)
    # Keep the metric accumulators in floating point. thsolver gathers and
    # averages every tracked tensor across GPUs, and torch.mean is undefined
    # for integer tensors in distributed validation.
    correct = pred.eq(label).sum().float()
    total = label.new_tensor(label.numel(), dtype=torch.float32)
    accu = correct / total.clamp_min(1)
    return loss, accu, correct, total

  def train_step(self, batch):
    loss, accu, _, _ = self.forward(batch)
    return {'train/loss': loss, 'train/accu': accu}

  def test_step(self, batch):
    with torch.no_grad():
      loss, _, correct, total = self.forward(batch)
    return {'test/loss': loss, 'test/correct': correct,
            'test/total': total}

  def result_callback(self, avg_tracker, epoch):
    # Point-BERT reports overall classification accuracy: the number of correct
    # predictions divided by the number of test shapes. Do not average per-batch
    # accuracies because the final batch can be smaller than the others.
    correct = avg_tracker.value.pop('test/correct')
    total = avg_tracker.value.pop('test/total')
    avg_tracker.num.pop('test/correct')
    avg_tracker.num.pop('test/total')
    accu = correct.float() / total.clamp_min(1).float()
    avg_tracker.update({'test/accu': accu})
    if self.is_master:
      print('[TEST] acc = %.4f' % (accu.item() * 100.0))


if __name__ == "__main__":
  ClsSolver.main()
