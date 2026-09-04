import os
import math
import time
import glob
from contextlib import nullcontext
import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver, get_config
from thsolver.tracker import AverageTracker

import builder
from datasets.shapenetpart import (
    SHAPENETPART_CATEGORIES, SHAPENETPART_PARTS,
    SHAPENETPART_UTONIA_TTA, build_shapenetpart_tta_points)
from datasets.partnete import (
    PARTNETE_CATEGORIES, PARTNETE_NUM_CLASSES, build_partnete_test_variant,
    make_partnete_points, partnete_named_part_ids,
    partnete_test_augmentations)
from datasets.s3dis import (
    S3DIS_CATEGORIES, build_s3dis_test_variant, make_s3dis_points,
    s3dis_test_augmentations, split_s3dis_fragments)
from datasets.semantickitti import (
    SEMANTICKITTI_CATEGORIES, SEMANTICKITTI_CE_WEIGHTS,
    SEMANTICKITTI_LEARNING_MAP_INV, SemanticKITTIDistributedEvalSampler,
    build_semantickitti_test_variant, make_semantickitti_points,
    semantickitti_test_augmentations, split_semantickitti_fragments)
from losses import lovasz_softmax_loss

# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
# torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def shapenetpart_metric_sums(logit, label, categories, batch_npt):
  r'''Returns additive ShapeNetPart statistics for distributed evaluation.

  Prediction is restricted to the valid part ids of each object category, as
  in the standard Point-BERT/PointNet ShapeNetPart evaluation. For a valid part
  absent from both prediction and ground truth, its IoU is defined as 1.
  '''
  if logit.ndim != 2 or logit.shape[1] != 50:
    raise ValueError('ShapeNetPart logits must have shape [N, 50].')
  label = label.reshape(-1)
  categories = torch.as_tensor(categories, device=logit.device).long().reshape(-1)
  batch_npt = torch.as_tensor(batch_npt, device=logit.device).long().reshape(-1)
  if label.numel() != logit.shape[0]:
    raise ValueError('ShapeNetPart logits and labels have different lengths.')
  if categories.numel() != batch_npt.numel():
    raise ValueError('One ShapeNetPart category is required for every shape.')
  if int(batch_npt.sum().item()) != label.numel():
    raise ValueError('ShapeNetPart batch point counts do not match labels.')

  dtype, device = logit.dtype, logit.device
  category_iou_sum = torch.zeros(
      len(SHAPENETPART_CATEGORIES), dtype=dtype, device=device)
  category_shape_count = torch.zeros_like(category_iou_sum)
  instance_iou_sum = torch.zeros((), dtype=dtype, device=device)
  instance_count = torch.zeros((), dtype=dtype, device=device)
  correct = torch.zeros((), dtype=dtype, device=device)
  point_count = torch.zeros((), dtype=dtype, device=device)

  start = 0
  for category_tensor, npt_tensor in zip(categories, batch_npt):
    category = int(category_tensor.item())
    npt = int(npt_tensor.item())
    if category < 0 or category >= len(SHAPENETPART_CATEGORIES):
      raise ValueError('Invalid ShapeNetPart category id: %d.' % category)

    end = start + npt
    shape_logit, shape_label = logit[start:end], label[start:end]
    valid_parts = torch.tensor(
        SHAPENETPART_PARTS[category], dtype=torch.long, device=device)
    pred = valid_parts[shape_logit[:, valid_parts].argmax(dim=1)]

    correct = correct + pred.eq(shape_label).sum().to(dtype)
    point_count = point_count + float(npt)
    part_ious = []
    for part in valid_parts:
      pred_part, label_part = pred.eq(part), shape_label.eq(part)
      union = torch.logical_or(pred_part, label_part).sum()
      if union.item() == 0:
        part_iou = logit.new_tensor(1.0)
      else:
        intersection = torch.logical_and(pred_part, label_part).sum()
        part_iou = intersection.to(dtype) / union.to(dtype)
      part_ious.append(part_iou)

    shape_iou = torch.stack(part_ious).mean()
    instance_iou_sum = instance_iou_sum + shape_iou
    instance_count = instance_count + 1.0
    category_iou_sum[category] = category_iou_sum[category] + shape_iou
    category_shape_count[category] = category_shape_count[category] + 1.0
    start = end

  return {
      'instance_iou_sum': instance_iou_sum,
      'instance_count': instance_count,
      'category_iou_sum': category_iou_sum,
      'category_shape_count': category_shape_count,
      'correct': correct,
      'point_count': point_count,
  }


def partnete_metric_sums_from_prediction(prediction, label, category):
  r'''Returns Pointcept PartNetE part-wise IoU sums for one full object.

  The official evaluator uses the global 148-way argmax, excludes each
  category's ``other`` part, and skips a named part when it is absent from the
  ground truth.  The final metric is the macro mean of the per-part averages.
  '''
  prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
  label = np.asarray(label, dtype=np.int64).reshape(-1)
  category = int(category)
  if prediction.shape != label.shape:
    raise ValueError('PartNetE prediction and label lengths do not match.')
  if category < 0 or category >= len(PARTNETE_CATEGORIES):
    raise ValueError('Invalid PartNetE category id: %d.' % category)

  iou_sum = np.zeros(PARTNETE_NUM_CLASSES, dtype=np.float64)
  iou_count = np.zeros(PARTNETE_NUM_CLASSES, dtype=np.float64)
  for part_id in partnete_named_part_ids(category):
    label_part = label == part_id
    if not label_part.any():
      continue
    pred_part = prediction == part_id
    union = np.logical_or(pred_part, label_part).sum()
    intersection = np.logical_and(pred_part, label_part).sum()
    iou_sum[part_id] = float(intersection) / (float(union) + 1.0e-10)
    iou_count[part_id] = 1.0
  return {
      'part_iou_sum': iou_sum,
      'part_iou_count': iou_count,
      'correct': float((prediction == label).sum()),
      'point_count': float(label.size),
  }


class SegSolver(Solver):

  @classmethod
  def update_configs(cls):
    r'''Registers PointTTT options so YAML/CLI overrides reach the backbone.'''
    flags = get_config()
    flags.defrost()
    flags.MODEL.partition_by_batch = False
    flags.MODEL.ttt_base_lr = 1.0
    flags.MODEL.ttt_update_train = True
    flags.MODEL.ttt_update_test = True
    flags.MODEL.ttt_patch_size = 64
    flags.MODEL.ttt_num_heads = 24
    flags.MODEL.ttt_layer_type = 'linear'
    # Hierarchical PointTTT is opt-in.  Classification configs and small-object
    # segmentation therefore keep exactly the historical local-only graph.
    flags.MODEL.pointttt_hierarchical_enabled = False
    flags.MODEL.pointttt_hierarchical_stages = []
    # 0 selects only the final block of each requested stage.  A positive
    # value additionally selects every N-th block (and always the final one).
    flags.MODEL.pointttt_hierarchical_block_interval = 0
    flags.MODEL.pointttt_global_chunk_size = 128
    flags.MODEL.pointttt_summary_tokens = 1
    flags.MODEL.pointttt_global_bidirectional = True
    # Zero gives an exact local-PointTTT function at initialization while the
    # gate learns how strongly the new global memory should contribute.
    flags.MODEL.pointttt_global_gate_init = 0.0
    # Optional local/earlier-phase PointTTT weights.  This is deliberately
    # separate from SOLVER.ckpt: solver checkpoints still perform exact resume
    # with optimizer and scheduler state and always take priority.
    flags.SOLVER.pointttt_pretrained = ''
    flags.freeze()

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # TTA is activated only by the final-test entry points below. Periodic
    # validation and ordinary ``SOLVER.run test`` therefore remain single-pass.
    self._shapenetpart_final_tta_active = False
    self._partnete_final_tta_active = False
    self._semantickitti_final_tta_active = False

  def is_shapenetpart(self):
    return self.FLAGS.DATA.test.name.lower() in ('shapenetpart', 'shapenet')

  def is_partnete(self):
    return self.FLAGS.DATA.test.name.lower() in ('partnete', 'partnet_e')

  def is_s3dis(self):
    return self.FLAGS.DATA.test.name.lower() == 's3dis'

  def is_semantickitti(self):
    return self.FLAGS.DATA.test.name.lower() in (
        'semantickitti', 'semantic_kitti')

  def get_model(self, flags):
    return builder.get_segmentation_model(flags)

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
    return builder.get_segmentation_dataset(flags)

  def get_dataloader(self, flags):
    name = flags.name.lower()
    semantic_test = name in ('semantickitti', 'semantic_kitti') and bool(
        getattr(flags, 'test_mode', False))
    if not (semantic_test and self.world_size > 1):
      return super().get_dataloader(flags)

    dataset, collate_fn = self.get_dataset(flags)
    rank = torch.distributed.get_rank()
    sampler = SemanticKITTIDistributedEvalSampler(
        dataset, self.world_size, rank)
    return torch.utils.data.DataLoader(
        dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
        sampler=sampler, collate_fn=collate_fn, pin_memory=flags.pin_memory)

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def process_batch(self, batch, flags):
    def points2octree(points):
      # ``points`` has already been moved to CUDA. Creating the Octree on its
      # default CPU device makes neighbor construction mix CPU/CUDA tensors.
      octree = ocnn.octree.Octree(
          flags.depth, flags.full_depth, device=points.device)
      octree.build_octree(points)
      return octree

    if 'octree' in batch:
      batch['octree'] = batch['octree'].cuda(non_blocking=True)
      batch['points'] = batch['points'].cuda(non_blocking=True)
    else:
      points = [pts.cuda(non_blocking=True) for pts in batch['points']]
      octrees = [points2octree(pts) for pts in points]
      octree = ocnn.octree.merge_octrees(octrees)
      octree.construct_all_neigh()
      batch['points'] = ocnn.octree.merge_points(points)
      batch['octree'] = octree
    return batch

  def model_forward_all(self, batch):
    octree, points = batch['octree'], batch['points']
    data = self.get_input_feature(octree)
    query_pts = torch.cat([points.points, points.batch_id], dim=1)
    logit = self.model(data, octree, octree.depth, query_pts)
    return logit, points.labels

  def model_forward(self, batch):
    # Most existing datasets discard ignored points before computing losses and
    # metrics. SemanticKITTI full-scan inference uses ``model_forward_all`` so
    # unlabeled test points are still predicted and written to the submission.
    logit, labels = self.model_forward_all(batch)
    label_mask = labels > self.FLAGS.LOSS.mask  # filter labels
    return logit[label_mask], labels[label_mask]

  def config_optimizer(self):
    flags = self.FLAGS.SOLVER
    if flags.type.lower() == 'adamw_attn':
      base_lr = flags.lr * self.world_size
      transformer_lr_scale = 0.1
      parameters = [
          {"params": [p for n, p in self.model.named_parameters()
                      if "blocks" not in n and p.requires_grad], },
          {"params": [p for n, p in self.model.named_parameters()
                      if "blocks" in n and p.requires_grad],
           "lr": base_lr * transformer_lr_scale, }, ]
      self.optimizer = torch.optim.AdamW(
          parameters, lr=base_lr, weight_decay=flags.weight_decay)
    else:
      super().config_optimizer()

  def config_lr_scheduler(self):
    flags = self.FLAGS.SOLVER
    if (self.is_semantickitti() or self.is_partnete()) and \
        flags.lr_type.lower() == 'onecycle':
      accumulation = max(1, int(getattr(flags, 'accumulation_steps', 1)))
      updates_per_epoch = int(math.ceil(len(self.train_loader) / accumulation))
      total_steps = updates_per_epoch * int(flags.max_epoch)
      max_lr = [group['lr'] for group in self.optimizer.param_groups]
      self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
          self.optimizer, max_lr=max_lr, total_steps=total_steps,
          pct_start=float(getattr(flags, 'pct_start', 0.04)),
          anneal_strategy='cos',
          div_factor=float(getattr(flags, 'div_factor', 10.0)),
          final_div_factor=float(getattr(flags, 'final_div_factor', 100.0)))
    else:
      super().config_lr_scheduler()

  def semantickitti_train_epoch(self, epoch):
    r'''SemanticKITTI epoch with optional gradient accumulation and OneCycle.'''
    self.model.train()
    if self.world_size > 1:
      self.train_loader.sampler.set_epoch(epoch)
      if hasattr(self.train_loader.sampler, 'reset_sampler'):
        self.train_loader.sampler.reset_sampler()

    flags = self.FLAGS.SOLVER
    accumulation = max(1, int(getattr(flags, 'accumulation_steps', 1)))
    num_iter = len(self.train_loader)
    tick = time.time()
    train_tracker = AverageTracker()
    self.optimizer.zero_grad()

    for it in tqdm(
        range(num_iter), ncols=80, leave=False, disable=self.disable_tqdm):
      batch = next(self.train_iter)
      batch['iter_num'] = it
      batch['epoch'] = epoch
      data_time = time.time() - tick

      group_start = (it // accumulation) * accumulation
      group_end = min(group_start + accumulation, num_iter)
      group_size = group_end - group_start
      update_now = (it + 1) == group_end
      sync_context = nullcontext()
      if self.world_size > 1 and not update_now:
        sync_context = self.model.no_sync()
      with sync_context:
        output = self.train_step(batch)
        (output['train/loss'] / float(group_size)).backward()

      if update_now:
        if flags.clip_grad > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), flags.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

      batch_time = time.time() - tick
      tick = time.time()
      output.update({
          'time/data': torch.tensor(data_time),
          'time/batch': torch.tensor(batch_time),
      })
      train_tracker.update(output)

      if it % 50 == 0 and flags.empty_cache:
        torch.cuda.empty_cache()
      if self.is_master and flags.log_per_iter > 0 and \
          it % flags.log_per_iter == 0:
        train_tracker.log(
            epoch, msg_tag='- ', notes='iter: %d' % it, print_time=False)

    if self.world_size > 1:
      train_tracker.average_all_gather()
    if self.is_master:
      train_tracker.log(epoch, self.summary_writer)

  def save_semantickitti_recovery_checkpoint(self, epoch):
    r'''Atomically saves a full epoch-boundary SemanticKITTI checkpoint.

    Validation is intentionally infrequent in the Pointcept protocol, so its
    cadence must not control whether training can be resumed.  The solver
    archive contains the model, optimizer, OneCycle scheduler and completed
    epoch.  Temporary files are atomically renamed only after ``torch.save``
    succeeds, preventing an interrupted write from appearing as the latest
    valid checkpoint.
    '''
    if not self.is_master:
      return

    os.makedirs(self.ckpt_dir, exist_ok=True)
    model_dict = (self.model.module.state_dict() if self.world_size > 1
                  else self.model.state_dict())
    stem = os.path.join(self.ckpt_dir, '%05d' % epoch)
    model_path, solver_path = stem + '.model.pth', stem + '.solver.tar'
    suffix = '.tmp.%d' % os.getpid()
    model_tmp, solver_tmp = model_path + suffix, solver_path + suffix
    state = {
        'model_dict': model_dict,
        'epoch': epoch,
        'optimizer_dict': self.optimizer.state_dict(),
        'scheduler_dict': self.scheduler.state_dict(),
        'best_val': self.best_val,
        'checkpoint_type': 'semantickitti_epoch_recovery',
    }

    try:
      torch.save(model_dict, model_tmp)
      torch.save(state, solver_tmp)
      # Publish the solver archive last: automatic resume only discovers the
      # final ``*.solver.tar`` filename, never either temporary file.
      os.replace(model_tmp, model_path)
      os.replace(solver_tmp, solver_path)
    finally:
      for temporary in (model_tmp, solver_tmp):
        if os.path.exists(temporary):
          os.remove(temporary)

    keep = max(1, int(self.FLAGS.SOLVER.ckpt_num))
    solver_files = sorted(glob.glob(
        os.path.join(self.ckpt_dir, '[0-9][0-9][0-9][0-9][0-9].solver.tar')))
    for old_solver in solver_files[:-keep]:
      old_stem = old_solver[:-len('.solver.tar')]
      old_model = old_stem + '.model.pth'
      os.remove(old_solver)
      if os.path.isfile(old_model):
        os.remove(old_model)

    tqdm.write(
        '=> Saved SemanticKITTI recovery checkpoint at epoch %d: %s' %
        (epoch, solver_path))

  def load_checkpoint(self):
    r'''Loads the regular solver checkpoint and restores validation history.'''
    pretrained = str(getattr(
        self.FLAGS.SOLVER, 'pointttt_pretrained', '')).strip()
    explicit_ckpt = str(getattr(self.FLAGS.SOLVER, 'ckpt', '')).strip()
    solver_files = sorted(glob.glob(
        os.path.join(self.ckpt_dir, '*.solver.tar')))
    if pretrained and not explicit_ckpt and not solver_files:
      self.load_pointttt_pretrained(pretrained)
      return

    if not (self.is_semantickitti() or self.is_partnete()):
      return super().load_checkpoint()

    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      solver_files = sorted(glob.glob(
          os.path.join(self.ckpt_dir, '*.solver.tar')))
      ckpt = solver_files[-1] if solver_files else ''
    if not ckpt:
      return

    trained_dict = torch.load(
        ckpt, map_location=torch.device('cuda', self.device))
    if ckpt.endswith('.solver.tar'):
      model_dict = trained_dict['model_dict']
      self.start_epoch = int(trained_dict['epoch']) + 1
      if self.optimizer is not None:
        self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
      if self.scheduler is not None:
        self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
      if 'best_val' in trained_dict:
        self.best_val = trained_dict['best_val']
    else:
      model_dict = trained_dict
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)

    if self.is_master:
      tqdm.write('Load the checkpoint: %s' % ckpt)
      tqdm.write('The start_epoch is %d' % self.start_epoch)
      scheduler_step = getattr(self.scheduler, 'last_epoch', -1)
      tqdm.write(
          '=> %s resume state: completed_epoch=%d, '
          'next_epoch=%d, scheduler_step=%d' %
          ('PartNetE' if self.is_partnete() else 'SemanticKITTI',
           self.start_epoch - 1, self.start_epoch, scheduler_step))

  def load_pointttt_pretrained(self, checkpoint):
    r'''Warm-starts a hierarchical phase from local/earlier PointTTT weights.

    Only newly introduced ``hierarchical_pointttt`` tensors may be missing.
    This prevents ``strict=False`` from silently hiding a broken backbone or
    segmentation head while allowing Local -> Stage 3 -> Stage 2+3 curricula.
    '''
    if not bool(getattr(
        self.FLAGS.MODEL, 'pointttt_hierarchical_enabled', False)):
      raise RuntimeError(
          'SOLVER.pointttt_pretrained requires Hierarchical PointTTT to be enabled.')
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(checkpoint):
      raise FileNotFoundError(
          'PointTTT pretrained checkpoint not found: ' + checkpoint)

    trained = torch.load(
        checkpoint, map_location=torch.device('cuda', self.device))
    state_dict = trained.get('model_dict', trained) if isinstance(
        trained, dict) else trained
    state_dict = {
        key[len('module.'):] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }
    model = self.model.module if self.world_size > 1 else self.model
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if '.hierarchical_pointttt.' not in key]
    if invalid_missing or incompatible.unexpected_keys:
      raise RuntimeError(
          'Invalid PointTTT warm-start checkpoint: non-hierarchical missing '
          f'keys={invalid_missing[:8]}, unexpected '
          f'keys={incompatible.unexpected_keys[:8]}')
    if not incompatible.missing_keys:
      raise RuntimeError(
          'SOLVER.pointttt_pretrained did not initialize a new hierarchical '
          'phase; use SOLVER.ckpt for an exact resume instead.')
    if self.is_master:
      tqdm.write(
          '=> Warm-start Hierarchical PointTTT from %s; initialized %d new '
          'hierarchical tensors.' %
          (checkpoint, len(incompatible.missing_keys)))

  def reset_semantickitti_validation_iterator(self):
    r'''Restarts the finite validation split independently on every rank.'''
    if hasattr(self.test_loader.sampler, 'reset_sampler'):
      self.test_loader.sampler.reset_sampler()
    self.test_iter = iter(self.test_loader)
    self.total_forward_time = 0
    self.num_iterations = 0
    self.total_memory_usage = 0

  def semantickitti_validation_recorded(self, epoch):
    r'''Returns a rank-consistent flag for an already completed validation.'''
    recorded = False
    if self.is_master and self.log_file and os.path.isfile(self.log_file):
      marker = 'Epoch: %d,' % epoch
      with open(self.log_file, 'r') as handle:
        recorded = any(
            marker in line and 'test/mIoU:' in line for line in handle)
    if self.world_size > 1:
      flag = torch.tensor(
          int(recorded), dtype=torch.int32,
          device=torch.device('cuda', self.device))
      torch.distributed.broadcast(flag, src=0)
      recorded = bool(flag.item())
    return recorded

  def _train_semantickitti_onecycle(self):
    self.manual_seed()
    self.config_model()
    self.config_dataloader()
    self.config_optimizer()
    self.config_lr_scheduler()
    self.configure_log()
    self.load_checkpoint()

    # A recovery checkpoint is published before validation. If validation was
    # interrupted (for example the exhausted-iterator bug at epoch 10), resume
    # by completing that missing pass before advancing to the next epoch.
    completed_epoch = self.start_epoch - 1
    test_every = int(self.FLAGS.SOLVER.test_every_epoch)
    if completed_epoch > 0 and completed_epoch % test_every == 0 and \
        not self.semantickitti_validation_recorded(completed_epoch):
      if self.is_master:
        tqdm.write(
            '=> Resuming missing SemanticKITTI validation at epoch %d' %
            completed_epoch)
      self.reset_semantickitti_validation_iterator()
      self.test_epoch(completed_epoch)
      if self.world_size > 1:
        torch.distributed.barrier()

    for epoch in tqdm(
        range(self.start_epoch, self.FLAGS.SOLVER.max_epoch + 1),
        ncols=80, disable=self.disable_tqdm):
      self.semantickitti_train_epoch(epoch)
      if self.is_master:
        self.summary_writer.add_scalar(
            'train/lr', self.scheduler.get_last_lr()[0], epoch)

      # Recovery checkpointing is independent from validation.  With the
      # default save_every_epoch=1, an interruption loses at most the current
      # unfinished epoch even though validation runs only at epoch 50.
      save_every = max(
          1, int(getattr(self.FLAGS.SOLVER, 'save_every_epoch', 1)))
      if epoch % save_every == 0 or \
          epoch == self.FLAGS.SOLVER.max_epoch:
        self.save_semantickitti_recovery_checkpoint(epoch)
      if self.world_size > 1:
        torch.distributed.barrier()

      if epoch % self.FLAGS.SOLVER.test_every_epoch == 0:
        # ``thsolver`` creates ``test_iter`` only once in
        # ``config_dataloader``.  A finite SemanticKITTI validation sampler is
        # exhausted after the first validation (for example at epoch 5), so a
        # later validation would otherwise raise StopIteration immediately.
        # Restart each rank's deterministic, non-padding validation partition
        # before every periodic pass.
        self.reset_semantickitti_validation_iterator()
        self.test_epoch(epoch)

    if self.world_size > 1:
      torch.distributed.barrier()

  def partnete_train_epoch(self, epoch):
    r'''PartNetE epoch with effective-batch gradient accumulation.'''
    self.model.train()
    if self.world_size > 1:
      self.train_loader.sampler.set_epoch(epoch)
      if hasattr(self.train_loader.sampler, 'reset_sampler'):
        self.train_loader.sampler.reset_sampler()

    flags = self.FLAGS.SOLVER
    accumulation = max(1, int(getattr(flags, 'accumulation_steps', 1)))
    num_iter = len(self.train_loader)
    tick = time.time()
    train_tracker = AverageTracker()
    self.optimizer.zero_grad()

    for it in tqdm(
        range(num_iter), ncols=80, leave=False, disable=self.disable_tqdm):
      batch = next(self.train_iter)
      batch['iter_num'] = it
      batch['epoch'] = epoch
      data_time = time.time() - tick

      group_start = (it // accumulation) * accumulation
      group_end = min(group_start + accumulation, num_iter)
      group_size = group_end - group_start
      update_now = (it + 1) == group_end
      sync_context = nullcontext()
      if self.world_size > 1 and not update_now:
        sync_context = self.model.no_sync()
      with sync_context:
        output = self.train_step(batch)
        (output['train/loss'] / float(group_size)).backward()

      if update_now:
        if flags.clip_grad > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), flags.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

      batch_time = time.time() - tick
      tick = time.time()
      output.update({
          'time/data': torch.tensor(data_time),
          'time/batch': torch.tensor(batch_time),
      })
      train_tracker.update(output)
      if it % 50 == 0 and flags.empty_cache:
        torch.cuda.empty_cache()
      if self.is_master and flags.log_per_iter > 0 and \
          it % flags.log_per_iter == 0:
        train_tracker.log(
            epoch, msg_tag='- ', notes='iter: %d' % it, print_time=False)

    if self.world_size > 1:
      train_tracker.average_all_gather()
    if self.is_master:
      train_tracker.log(epoch, self.summary_writer)

  def save_partnete_recovery_checkpoint(self, epoch):
    r'''Saves model/optimizer/OneCycle state independently of validation.'''
    if not self.is_master:
      return
    os.makedirs(self.ckpt_dir, exist_ok=True)
    model_dict = (self.model.module.state_dict() if self.world_size > 1
                  else self.model.state_dict())
    stem = os.path.join(self.ckpt_dir, '%05d' % epoch)
    model_path, solver_path = stem + '.model.pth', stem + '.solver.tar'
    suffix = '.tmp.%d' % os.getpid()
    model_tmp, solver_tmp = model_path + suffix, solver_path + suffix
    state = {
        'model_dict': model_dict,
        'epoch': epoch,
        'optimizer_dict': self.optimizer.state_dict(),
        'scheduler_dict': self.scheduler.state_dict(),
        'best_val': self.best_val,
        'checkpoint_type': 'partnete_epoch_recovery',
    }
    try:
      torch.save(model_dict, model_tmp)
      torch.save(state, solver_tmp)
      os.replace(model_tmp, model_path)
      os.replace(solver_tmp, solver_path)
    finally:
      for temporary in (model_tmp, solver_tmp):
        if os.path.exists(temporary):
          os.remove(temporary)

    keep = max(1, int(self.FLAGS.SOLVER.ckpt_num))
    solver_files = sorted(glob.glob(
        os.path.join(self.ckpt_dir, '[0-9][0-9][0-9][0-9][0-9].solver.tar')))
    for old_solver in solver_files[:-keep]:
      old_stem = old_solver[:-len('.solver.tar')]
      old_model = old_stem + '.model.pth'
      os.remove(old_solver)
      if os.path.isfile(old_model):
        os.remove(old_model)
    tqdm.write(
        '=> Saved PartNetE recovery checkpoint at epoch %d: %s' %
        (epoch, solver_path))

  def reset_partnete_validation_iterator(self):
    if hasattr(self.test_loader.sampler, 'reset_sampler'):
      self.test_loader.sampler.reset_sampler()
    self.test_iter = iter(self.test_loader)
    self.total_forward_time = 0
    self.num_iterations = 0
    self.total_memory_usage = 0

  def partnete_validation_recorded(self, epoch):
    r'''Checks whether an epoch-boundary PartNetE validation completed.'''
    recorded = False
    if self.is_master and self.log_file and os.path.isfile(self.log_file):
      marker = 'Epoch: %d,' % epoch
      with open(self.log_file, 'r') as handle:
        recorded = any(
            marker in line and 'test/mIoU_part:' in line for line in handle)
    if self.world_size > 1:
      flag = torch.tensor(
          int(recorded), dtype=torch.int32,
          device=torch.device('cuda', self.device))
      torch.distributed.broadcast(flag, src=0)
      recorded = bool(flag.item())
    return recorded

  def _train_partnete_onecycle(self):
    self.manual_seed()
    self.config_model()
    self.config_dataloader()
    self.config_optimizer()
    self.config_lr_scheduler()
    self.configure_log()
    self.load_checkpoint()

    # The recovery checkpoint is published before validation. If that
    # validation was interrupted, finish it before advancing to the next
    # training epoch so checkpoint selection never silently skips a boundary.
    completed_epoch = self.start_epoch - 1
    test_every = int(self.FLAGS.SOLVER.test_every_epoch)
    if completed_epoch > 0 and completed_epoch % test_every == 0 and \
        not self.partnete_validation_recorded(completed_epoch):
      if self.is_master:
        tqdm.write(
            '=> Resuming missing PartNetE validation at epoch %d' %
            completed_epoch)
      self.reset_partnete_validation_iterator()
      self.test_epoch(completed_epoch)
      if self.world_size > 1:
        torch.distributed.barrier()

    for epoch in tqdm(
        range(self.start_epoch, self.FLAGS.SOLVER.max_epoch + 1),
        ncols=80, disable=self.disable_tqdm):
      self.partnete_train_epoch(epoch)
      if self.is_master:
        self.summary_writer.add_scalar(
            'train/lr', self.scheduler.get_last_lr()[0], epoch)

      save_every = max(
          1, int(getattr(self.FLAGS.SOLVER, 'save_every_epoch', 10)))
      if epoch % save_every == 0 or epoch == self.FLAGS.SOLVER.max_epoch:
        self.save_partnete_recovery_checkpoint(epoch)
      if self.world_size > 1:
        torch.distributed.barrier()

      if epoch % self.FLAGS.SOLVER.test_every_epoch == 0:
        self.reset_partnete_validation_iterator()
        self.test_epoch(epoch)
      if self.world_size > 1:
        torch.distributed.barrier()

  def train_step(self, batch):
    batch = self.process_batch(batch, self.FLAGS.DATA.train)
    logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)
    accu = self.accuracy(logit, label)
    return {'train/loss': loss, 'train/accu': accu}

  def test_step(self, batch):
    if self.is_partnete() and batch.get('_partnete_test', False):
      return self.partnete_test_step(batch)

    if self.is_semantickitti() and batch.get('_semantickitti_test', False):
      return self.semantickitti_test_step(batch)

    if self.is_s3dis() and batch.get('_s3dis_test', False):
      return self.s3dis_test_step(batch)

    if self.is_shapenetpart() and self._shapenetpart_final_tta_active:
      return self.shapenetpart_tta_test_step(batch)

    batch = self.process_batch(batch, self.FLAGS.DATA.test)
    with torch.no_grad():
      logit, label = self.model_forward(batch)
    loss = self.loss_function(logit, label)

    if self.is_shapenetpart():
      stats = shapenetpart_metric_sums(
          logit, label, batch['label'], batch['points'].batch_npt)
      return self.shapenetpart_metric_output(stats, loss)

    accu = self.accuracy(logit, label)
    num_class = self.FLAGS.LOSS.num_class
    IoU, insc, union = self.IoU_per_shape(logit, label, num_class)

    names = ['test/loss', 'test/accu', 'test/mIoU'] + \
            ['test/intsc_%d' % i for i in range(num_class)] + \
            ['test/union_%d' % i for i in range(num_class)]
    tensors = [loss, accu, IoU] + insc + union
    return dict(zip(names, tensors))

  @staticmethod
  def shapenetpart_metric_output(stats, loss):
    output = {
        'test/loss': loss,
        'test/_instance_iou_sum': stats['instance_iou_sum'],
        'test/_instance_count': stats['instance_count'],
        'test/_correct': stats['correct'],
        'test/_point_count': stats['point_count'],
    }
    for category_id in range(len(SHAPENETPART_CATEGORIES)):
      output['test/_category_iou_sum_%d' % category_id] = \
          stats['category_iou_sum'][category_id]
      output['test/_category_count_%d' % category_id] = \
          stats['category_shape_count'][category_id]
    return output

  def shapenetpart_tta_test_step(self, batch):
    r'''Fuses the ten Utonia ShapeNetPart votes by summing softmax scores.'''
    flags = self.FLAGS.DATA.test
    source_points = batch['points']
    categories = batch['label']
    probability_sum, reference_label, batch_npt = None, None, None

    for scale, flip_probability in SHAPENETPART_UTONIA_TTA:
      points = build_shapenetpart_tta_points(
          source_points, scale, flip_probability,
          float(getattr(flags, 'octree_bound', 0.999)))
      variant = self.process_batch({'points': points}, flags)
      with torch.no_grad():
        logit, label = self.model_forward(variant)
        probability = torch.softmax(logit, dim=1)

      if probability_sum is None:
        probability_sum = probability
        reference_label = label
        batch_npt = variant['points'].batch_npt
      else:
        if not torch.equal(label, reference_label):
          raise RuntimeError(
              'ShapeNetPart labels changed between TTA prediction branches.')
        probability_sum.add_(probability)

    probability = probability_sum / float(len(SHAPENETPART_UTONIA_TTA))
    # This loss is diagnostic only. Predictions and IoUs use the summed
    # probabilities exactly as the Utonia tester does.
    loss = torch.nn.functional.nll_loss(
        probability.clamp_min(1.0e-12).log(), reference_label.long())
    stats = shapenetpart_metric_sums(
        probability_sum, reference_label, categories, batch_npt)
    return self.shapenetpart_metric_output(stats, loss)

  @staticmethod
  def partnete_metric_output(stats, loss, device):
    output = {
        'test/loss': torch.tensor(float(loss), device=device),
        'test/_partnete_correct': torch.tensor(
            stats['correct'], dtype=torch.float32, device=device),
        'test/_partnete_point_count': torch.tensor(
            stats['point_count'], dtype=torch.float32, device=device),
    }
    for part_id in range(PARTNETE_NUM_CLASSES):
      output['test/_partnete_iou_sum_%d' % part_id] = torch.tensor(
          stats['part_iou_sum'][part_id], dtype=torch.float32, device=device)
      output['test/_partnete_iou_count_%d' % part_id] = torch.tensor(
          stats['part_iou_count'][part_id], dtype=torch.float32, device=device)
    return output

  def partnete_test_step(self, sample):
    r'''Full-resolution PartNetE inference and optional Utonia ten-vote TTA.'''
    flags = self.FLAGS.DATA.test
    num_class = PARTNETE_NUM_CLASSES
    source = {
        key: np.asarray(sample[key])
        for key in ('coord', 'normal', 'color', 'segment')
    }
    label = np.asarray(source['segment'], dtype=np.int64).reshape(-1)
    category = int(sample['category'])
    npt = label.size
    vote_sum = np.zeros((npt, num_class), dtype=np.float32)
    vote_count = np.zeros(npt, dtype=np.int32)
    final_tta = self._partnete_final_tta_active and bool(
        getattr(flags, 'tta', False))
    augmentations = partnete_test_augmentations(final_tta)
    output_device = torch.device('cuda', torch.cuda.current_device())

    for scale, flip_probability in augmentations:
      data, fragments, representative, inverse = build_partnete_test_variant(
          source, flags, scale, flip_probability, exhaustive=final_tta)

      if representative is not None:
        representative_prob = np.zeros(
            (representative.size, num_class), dtype=np.float32)
        representative_position = np.full(npt, -1, dtype=np.int64)
        representative_position[representative] = np.arange(
            representative.size, dtype=np.int64)

      for index in fragments:
        points = make_partnete_points(data, index, flags)
        fragment_batch = self.process_batch({'points': [points]}, flags)
        with torch.no_grad():
          logit, _ = self.model_forward_all(fragment_batch)
          probability = torch.softmax(logit, dim=1).cpu().numpy()
        output_device = logit.device
        if representative is None:
          vote_sum[index] += probability
          vote_count[index] += 1
        else:
          position = representative_position[index]
          if (position < 0).any():
            raise RuntimeError('Invalid PartNetE representative mapping.')
          representative_prob[position] = probability

      if representative is not None:
        vote_sum += representative_prob[inverse]
        vote_count += 1

    if not np.all(vote_count > 0):
      raise RuntimeError('PartNetE inference did not cover every input point.')
    probability = vote_sum / vote_count[:, None]
    prediction = probability.argmax(axis=1)
    true_probability = probability[np.arange(npt), label]
    loss = float(-np.log(np.maximum(true_probability, 1.0e-12)).mean())
    stats = partnete_metric_sums_from_prediction(
        prediction, label, category)
    return self.partnete_metric_output(stats, loss, output_device)

  def save_best_checkpoint(self, tracker, epoch):
    # The final TTA pass evaluates the selected checkpoint and must not rewrite
    # best_model.pth or participate in checkpoint selection.
    if self._shapenetpart_final_tta_active or \
        self._partnete_final_tta_active or \
        self._semantickitti_final_tta_active:
      return
    super().save_best_checkpoint(tracker, epoch)

  def load_shapenetpart_final_checkpoint(self, checkpoint=None):
    checkpoint = checkpoint or os.path.join(self.logdir, 'best_model.pth')
    if not os.path.isfile(checkpoint):
      raise FileNotFoundError(
          'ShapeNetPart final TTA checkpoint not found: ' + checkpoint)
    trained_dict = torch.load(
        checkpoint, map_location=torch.device('cuda', self.device))
    model_dict = (trained_dict['model_dict']
                  if checkpoint.endswith('.solver.tar') else trained_dict)
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)
    if self.is_master:
      tqdm.write('=> Loaded best model for ShapeNetPart 10-vote TTA: ' +
                 checkpoint)

  def run_shapenetpart_final_tta(self, checkpoint=None):
    if not self.is_shapenetpart():
      raise RuntimeError('The final ten-vote TTA is only for ShapeNetPart.')
    if not bool(getattr(self.FLAGS.DATA.test, 'tta', False)):
      raise RuntimeError(
          'Set DATA.test.tta True to run ShapeNetPart ten-vote TTA.')

    self.load_shapenetpart_final_checkpoint(checkpoint)
    if self.world_size > 1:
      torch.distributed.barrier()

    # Restart the deterministic sampler so this pass is independent of all
    # periodic validation epochs.
    if hasattr(self.test_loader.sampler, 'reset_sampler'):
      self.test_loader.sampler.reset_sampler()
    self.test_iter = iter(self.test_loader)
    self.total_forward_time = 0
    self.num_iterations = 0
    self.total_memory_usage = 0

    old_log_file = self.log_file
    self.log_file = os.path.join(self.logdir, 'final_tta_log.csv')
    self._shapenetpart_final_tta_active = True
    try:
      if self.is_master:
        tqdm.write(
            '=> Starting final ShapeNetPart Utonia 10-vote TTA evaluation')
      self.test_epoch(epoch=self.FLAGS.SOLVER.max_epoch + 1)
    finally:
      self._shapenetpart_final_tta_active = False
      self.log_file = old_log_file

    if self.world_size > 1:
      torch.distributed.barrier()

  def load_partnete_final_checkpoint(self, checkpoint=None):
    checkpoint = checkpoint or os.path.join(self.logdir, 'best_model.pth')
    if not os.path.isfile(checkpoint):
      raise FileNotFoundError(
          'PartNetE final TTA checkpoint not found: ' + checkpoint)
    trained_dict = torch.load(
        checkpoint, map_location=torch.device('cuda', self.device))
    model_dict = (trained_dict['model_dict']
                  if checkpoint.endswith('.solver.tar') else trained_dict)
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)
    if self.is_master:
      tqdm.write('=> Loaded best model for PartNetE 10-vote TTA: ' + checkpoint)

  def run_partnete_final_tta(self, checkpoint=None):
    if not self.is_partnete():
      raise RuntimeError('The final ten-vote TTA is only for PartNetE.')
    if not bool(getattr(self.FLAGS.DATA.test, 'tta', False)):
      raise RuntimeError('Set DATA.test.tta True to run PartNetE ten-vote TTA.')
    self.load_partnete_final_checkpoint(checkpoint)
    if self.world_size > 1:
      torch.distributed.barrier()
    self.reset_partnete_validation_iterator()

    old_log_file = self.log_file
    self.log_file = os.path.join(self.logdir, 'final_tta_log.csv')
    self._partnete_final_tta_active = True
    try:
      if self.is_master:
        tqdm.write('=> Starting final PartNetE Utonia 10-vote TTA evaluation')
      self.test_epoch(epoch=self.FLAGS.SOLVER.max_epoch + 1)
    finally:
      self._partnete_final_tta_active = False
      self.log_file = old_log_file
    if self.world_size > 1:
      torch.distributed.barrier()

  def load_semantickitti_final_checkpoint(self, checkpoint=None):
    checkpoint = checkpoint or os.path.join(self.logdir, 'best_model.pth')
    if not os.path.isfile(checkpoint):
      raise FileNotFoundError(
          'SemanticKITTI final checkpoint not found: ' + checkpoint)
    trained_dict = torch.load(
        checkpoint, map_location=torch.device('cuda', self.device))
    model_dict = (trained_dict['model_dict']
                  if checkpoint.endswith('.solver.tar') else trained_dict)
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)
    if self.is_master:
      tqdm.write(
          '=> Loaded best model for SemanticKITTI final evaluation: ' +
          checkpoint)

  def run_semantickitti_final_tta(self, checkpoint=None):
    if not self.is_semantickitti():
      raise RuntimeError(
          'The SemanticKITTI final TTA is only for SemanticKITTI.')
    if not bool(getattr(self.FLAGS.DATA.test, 'tta', False)):
      raise RuntimeError(
          'Set DATA.test.tta True to run SemanticKITTI four-vote TTA.')
    self.load_semantickitti_final_checkpoint(checkpoint)
    if self.world_size > 1:
      torch.distributed.barrier()
    if hasattr(self.test_loader.sampler, 'reset_sampler'):
      self.test_loader.sampler.reset_sampler()
    self.test_iter = iter(self.test_loader)
    self.total_forward_time = 0
    self.num_iterations = 0
    self.total_memory_usage = 0

    old_log_file = self.log_file
    self.log_file = os.path.join(self.logdir, 'final_tta_log.csv')
    self._semantickitti_final_tta_active = True
    try:
      if self.is_master:
        tqdm.write(
            '=> Starting final SemanticKITTI 4-rotation TTA evaluation')
      self.test_epoch(epoch=self.FLAGS.SOLVER.max_epoch + 1)
    finally:
      self._semantickitti_final_tta_active = False
      self.log_file = old_log_file
    if self.world_size > 1:
      torch.distributed.barrier()

  def train(self):
    onecycle = (self.is_semantickitti() or self.is_partnete()) and \
        self.FLAGS.SOLVER.lr_type.lower() == 'onecycle'
    if onecycle and self.is_semantickitti():
      self._train_semantickitti_onecycle()
    elif onecycle and self.is_partnete():
      self._train_partnete_onecycle()
    else:
      super().train()
    enabled = bool(getattr(self.FLAGS.SOLVER, 'final_test_best', False))
    enabled = enabled and bool(getattr(self.FLAGS.DATA.test, 'tta', False))
    if self.is_shapenetpart() and enabled:
      self.run_shapenetpart_final_tta()
    elif self.is_partnete() and enabled:
      self.run_partnete_final_tta()
    elif self.is_semantickitti() and enabled:
      self.run_semantickitti_final_tta()

  def test_tta(self):
    r'''Explicit final TTA entry point for an already trained checkpoint.'''
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    checkpoint = self.FLAGS.SOLVER.ckpt or None
    if self.is_shapenetpart():
      self.run_shapenetpart_final_tta(checkpoint)
    elif self.is_partnete():
      self.run_partnete_final_tta(checkpoint)
    elif self.is_semantickitti():
      self.run_semantickitti_final_tta(checkpoint)
    else:
      raise RuntimeError(
          'SOLVER.run test_tta supports ShapeNetPart, PartNetE, and '
          'SemanticKITTI.')

  def submit(self):
    r'''Generates SemanticKITTI test-set .label submission files.'''
    if not self.is_semantickitti():
      raise RuntimeError('SOLVER.run submit is only for SemanticKITTI.')
    self.manual_seed()
    self.config_model()
    self.configure_log(set_writer=False)
    self.config_dataloader(disable_train_data=True)
    checkpoint = self.FLAGS.SOLVER.ckpt or None
    if bool(getattr(self.FLAGS.DATA.test, 'tta', False)):
      self.run_semantickitti_final_tta(checkpoint)
    else:
      self.load_semantickitti_final_checkpoint(checkpoint)
      self.test_epoch(epoch=0)

  def s3dis_test_step(self, sample):
    r'''Fragment inference and full-resolution evaluation for one S3DIS room.'''
    flags = self.FLAGS.DATA.test
    num_class = self.FLAGS.LOSS.num_class
    raw_label = np.asarray(sample['segment'], dtype=np.int64).reshape(-1)
    raw_npt = raw_label.size
    vote_sum = np.zeros((raw_npt, num_class), dtype=np.float32)
    augmentations = s3dis_test_augmentations(bool(getattr(flags, 'tta', False)))
    loss_sum, loss_npt = 0.0, 0
    output_device = torch.device('cuda', torch.cuda.current_device())

    for scale, flip in augmentations:
      sampled, inverse = build_s3dis_test_variant(sample, flags, scale, flip)
      fragments = split_s3dis_fragments(
          sampled['coord'], int(getattr(flags, 'fragment_max_npt', 120000)),
          float(getattr(flags, 'scale_factor', 10.24)),
          float(getattr(flags, 'octree_bound', 0.999)))
      voxel_prob = np.zeros(
          (sampled['coord'].shape[0], num_class), dtype=np.float32)
      covered = np.zeros(sampled['coord'].shape[0], dtype=np.bool_)

      for index in fragments:
        points = make_s3dis_points(sampled, index, flags)
        fragment_batch = self.process_batch(
            {'points': [points]}, self.FLAGS.DATA.test)
        with torch.no_grad():
          logit, label = self.model_forward(fragment_batch)
          loss = self.loss_function(logit, label)
          prob = torch.softmax(logit, dim=1).cpu().numpy()
        voxel_prob[index] = prob
        covered[index] = True
        loss_sum += float(loss.item()) * int(label.numel())
        loss_npt += int(label.numel())
        output_device = logit.device

      if not covered.all():
        raise RuntimeError('S3DIS fragments did not cover the complete room.')
      # Project voxel predictions to all original room points in bounded chunks
      # to avoid a second full [N, C] temporary allocation.
      project_step = 1000000
      for start in range(0, raw_npt, project_step):
        end = min(start + project_step, raw_npt)
        vote_sum[start:end] += voxel_prob[inverse[start:end]]

    pred = vote_sum.argmax(axis=1)
    correct_label = raw_label[pred == raw_label]
    intersection = np.bincount(correct_label, minlength=num_class)
    prediction = np.bincount(pred, minlength=num_class)
    target = np.bincount(raw_label, minlength=num_class)
    union = prediction + target - intersection
    loss_value = loss_sum / max(loss_npt, 1)

    return {
        'test/loss': torch.tensor(loss_value, device=output_device),
        'test/_s3dis_intersection': torch.as_tensor(
            intersection, dtype=torch.float32, device=output_device),
        'test/_s3dis_union': torch.as_tensor(
            union, dtype=torch.float32, device=output_device),
        'test/_s3dis_target': torch.as_tensor(
            target, dtype=torch.float32, device=output_device),
    }

  def _save_semantickitti_submission(self, sample, pred, covered):
    flags = self.FLAGS.DATA.test
    submission_root = str(getattr(flags, 'submission_dir', '')).strip()
    if not submission_root:
      submission_root = os.path.join(self.logdir, 'submission')
    output_dir = os.path.join(
        submission_root, 'sequences', sample['sequence'], 'predictions')
    os.makedirs(output_dir, exist_ok=True)

    # Points outside Pointcept's official clipping range are written as the
    # official unlabeled id 0. Every input point still receives one uint32
    # entry, as required by the SemanticKITTI submission format.
    raw_prediction = np.zeros(pred.shape[0], dtype=np.uint32)
    for learning_id, raw_id in SEMANTICKITTI_LEARNING_MAP_INV.items():
      mask = np.logical_and(covered, pred == learning_id)
      raw_prediction[mask] = np.uint32(raw_id)
    output_path = os.path.join(output_dir, sample['frame'] + '.label')
    raw_prediction.tofile(output_path)
    return output_path

  def semantickitti_test_step(self, sample):
    r'''Full-scan SemanticKITTI validation, TTA, and test submission.'''
    flags = self.FLAGS.DATA.test
    num_class = int(self.FLAGS.LOSS.num_class)
    raw_label = np.asarray(sample['segment'], dtype=np.int64).reshape(-1)
    raw_npt = raw_label.size
    vote_sum = np.zeros((raw_npt, num_class), dtype=np.float32)
    vote_count = np.zeros(raw_npt, dtype=np.int32)
    enable_tta = self._semantickitti_final_tta_active and bool(
        getattr(flags, 'tta', False))
    augmentations = semantickitti_test_augmentations(enable_tta)
    loss_sum, loss_npt = 0.0, 0
    output_device = torch.device('cuda', torch.cuda.current_device())

    for angle in augmentations:
      sampled, inverse, valid_index = build_semantickitti_test_variant(
          sample, flags, angle)
      fragments = split_semantickitti_fragments(
          sampled['coord'], int(getattr(flags, 'fragment_max_npt', 120000)))
      voxel_prob = np.zeros(
          (sampled['coord'].shape[0], num_class), dtype=np.float32)
      voxel_covered = np.zeros(sampled['coord'].shape[0], dtype=np.bool_)

      for index in fragments:
        points = make_semantickitti_points(sampled, index, flags)
        fragment_batch = self.process_batch(
            {'points': [points]}, self.FLAGS.DATA.test)
        with torch.no_grad():
          logit, label = self.model_forward_all(fragment_batch)
          probability = torch.softmax(logit, dim=1)
        voxel_prob[index] = probability.cpu().numpy()
        voxel_covered[index] = True
        valid_label = label > self.FLAGS.LOSS.mask
        if valid_label.any():
          loss = self.loss_function(logit[valid_label], label[valid_label])
          valid_npt = int(valid_label.sum().item())
          loss_sum += float(loss.item()) * valid_npt
          loss_npt += valid_npt
        output_device = logit.device

      if not voxel_covered.all():
        raise RuntimeError(
            'SemanticKITTI fragments did not cover the complete scan.')
      vote_sum[valid_index] += voxel_prob[inverse]
      vote_count[valid_index] += 1

    covered = vote_count > 0
    if not covered.any():
      raise RuntimeError('SemanticKITTI inference did not cover any points.')
    vote_sum[covered] /= vote_count[covered, None]
    pred = vote_sum.argmax(axis=1)
    loss_value = loss_sum / max(loss_npt, 1)

    if not bool(sample['has_label']):
      self._save_semantickitti_submission(sample, pred, covered)
      return {
          'test/loss': torch.tensor(0.0, device=output_device),
          'test/_semantickitti_submitted': torch.tensor(
              1.0, device=output_device),
      }

    evaluate = np.logical_and(covered, raw_label >= 0)
    pred_eval, label_eval = pred[evaluate], raw_label[evaluate]
    correct_label = label_eval[pred_eval == label_eval]
    intersection = np.bincount(correct_label, minlength=num_class)
    prediction = np.bincount(pred_eval, minlength=num_class)
    target = np.bincount(label_eval, minlength=num_class)
    union = prediction + target - intersection
    output = {'test/loss': torch.tensor(loss_value, device=output_device)}
    # thsolver 1.1.4 reduces a vector to a scalar in its distributed tracker.
    # Keep one scalar carrier per class so two-GPU metric aggregation remains
    # correct without changing thsolver or any existing dataset.
    for category_id in range(num_class):
      output['test/_semantickitti_intersection_%d' % category_id] = \
          torch.tensor(float(intersection[category_id]), device=output_device)
      output['test/_semantickitti_union_%d' % category_id] = \
          torch.tensor(float(union[category_id]), device=output_device)
      output['test/_semantickitti_target_%d' % category_id] = \
          torch.tensor(float(target[category_id]), device=output_device)
    return output

  def eval_step(self, batch):
    batch = self.process_batch(batch, self.FLAGS.DATA.test)
    with torch.no_grad():
      logit, _ = self.model_forward(batch)
    prob = torch.nn.functional.softmax(logit, dim=1)

    # split predictions
    inbox_masks = batch['inbox_mask']
    npts = batch['points'].batch_npt.tolist()
    probs = torch.split(prob, npts)

    # merge predictions
    batch_size = len(inbox_masks)
    for i in range(batch_size):
      # The point cloud may be clipped when doing data augmentation. The
      # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
      # contains the prediction for all points.
      prob = probs[i].cpu()
      inbox_mask = inbox_masks[i].to(prob.device)
      prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
      prob_all_pts[inbox_mask] = prob

      # Aggregate predictions across different epochs
      filename = batch['filename'][i]
      self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

      # Save the prediction results in the last epoch
      if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
        full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')
        curr_folder = os.path.dirname(full_filename)
        if not os.path.exists(curr_folder): os.makedirs(curr_folder)
        np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

  def result_callback(self, avg_tracker, epoch):
    r''' Calculate the part mIoU for PartNet and ScanNet.
    '''

    if self.is_semantickitti():
      self.semantickitti_result_callback(avg_tracker, epoch)
      return

    if self.is_s3dis():
      self.s3dis_result_callback(avg_tracker, epoch)
      return

    if self.is_partnete():
      self.partnete_result_callback(avg_tracker, epoch)
      return

    if self.is_shapenetpart():
      self.shapenetpart_result_callback(avg_tracker, epoch)
      return

    iou_part = 0.0
    avg = avg_tracker.average()

    # Labels smaller than `mask` is ignored. The points with the label 0 in
    # PartNet are background points, i.e., unlabeled points
    mask = self.FLAGS.LOSS.mask + 1
    num_class = self.FLAGS.LOSS.num_class
    for i in range(mask, num_class):
      instc_i = avg['test/intsc_%d' % i]
      union_i = avg['test/union_%d' % i]
      iou_part += instc_i / (union_i + 1.0e-10)
    iou_part = iou_part / (num_class - mask)

    avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
    tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

  def shapenetpart_result_callback(self, avg_tracker, epoch):
    r'''Computes Point-BERT ShapeNetPart mIoUI and mIoUC metrics.'''
    avg = avg_tracker.average()
    eps = 1.0e-12
    miou_i = avg['test/_instance_iou_sum'] / (
        avg['test/_instance_count'] + eps)
    point_accu = avg['test/_correct'] / (avg['test/_point_count'] + eps)

    category_ious = []
    for category_id in range(len(SHAPENETPART_CATEGORIES)):
      iou_sum = avg['test/_category_iou_sum_%d' % category_id]
      count = avg['test/_category_count_%d' % category_id]
      if count <= 0:
        raise RuntimeError(
            'No test shapes found for ShapeNetPart category %s.' %
            SHAPENETPART_CATEGORIES[category_id])
      category_ious.append(iou_sum / count)
    miou_c = sum(category_ious) / len(category_ious)

    # The underscore-prefixed values are only distributed sum/count carriers;
    # remove them so that log.csv and TensorBoard contain benchmark metrics.
    internal_keys = [key for key in avg_tracker.value
                     if key.startswith('test/_')]
    for key in internal_keys:
      del avg_tracker.value[key]
      del avg_tracker.num[key]

    sample_tensor = next(iter(avg_tracker.value.values()))
    device = sample_tensor.device if isinstance(sample_tensor, torch.Tensor) \
        else torch.device('cpu')
    final_metrics = {
        'test/accu': torch.tensor(point_accu, device=device),
        'test/mIoUC': torch.tensor(miou_c, device=device),
        'test/mIoUI': torch.tensor(miou_i, device=device),
    }
    for name, iou in zip(SHAPENETPART_CATEGORIES, category_ious):
      final_metrics['test/IoU_' + name] = torch.tensor(iou, device=device)
    avg_tracker.update(final_metrics)

    tqdm.write(
        '=> Epoch: %d, test/accu: %.6f, test/mIoUC: %.6f, '
        'test/mIoUI: %.6f' % (epoch, point_accu, miou_c, miou_i))
    category_msg = ', '.join(
        '%s: %.4f' % (name, iou)
        for name, iou in zip(SHAPENETPART_CATEGORIES, category_ious))
    tqdm.write('=> ShapeNetPart category IoUs: ' + category_msg)

  def partnete_result_callback(self, avg_tracker, epoch):
    r'''Computes Pointcept's official PartNetE named-part macro mIoU.'''
    avg = avg_tracker.average()
    sample = next(iter(avg_tracker.value.values()))
    device = sample.device if isinstance(sample, torch.Tensor) \
        else torch.device('cpu')
    iou_sum = torch.stack([
        torch.as_tensor(
            avg['test/_partnete_iou_sum_%d' % part_id], device=device)
        for part_id in range(PARTNETE_NUM_CLASSES)])
    iou_count = torch.stack([
        torch.as_tensor(
            avg['test/_partnete_iou_count_%d' % part_id], device=device)
        for part_id in range(PARTNETE_NUM_CLASSES)])
    valid = iou_count > 0
    if not valid.any():
      raise RuntimeError('PartNetE evaluation found no named ground-truth part.')
    part_iou = iou_sum[valid] / iou_count[valid].clamp_min(1.0e-10)
    part_miou = part_iou.mean()
    accuracy = torch.as_tensor(
        avg['test/_partnete_correct'], device=device) / torch.as_tensor(
            avg['test/_partnete_point_count'], device=device).clamp_min(1.0e-10)

    internal_keys = [
        key for key in avg_tracker.value if key.startswith('test/_partnete_')]
    for key in internal_keys:
      del avg_tracker.value[key]
      del avg_tracker.num[key]
    avg_tracker.update({
        'test/accu': accuracy,
        'test/mIoU_part': part_miou,
    })
    tqdm.write(
        '=> Epoch: %d, test/accu: %.6f, test/mIoU_part: %.6f '
        '(103 named parts, other excluded)' %
        (epoch, accuracy.item(), part_miou.item()))

  def s3dis_result_callback(self, avg_tracker, epoch):
    r'''Computes the mIoU/mAcc/allAcc metrics used by Pointcept PTv3.'''
    intersection = avg_tracker.value.pop('test/_s3dis_intersection')
    union = avg_tracker.value.pop('test/_s3dis_union')
    target = avg_tracker.value.pop('test/_s3dis_target')
    avg_tracker.num.pop('test/_s3dis_intersection')
    avg_tracker.num.pop('test/_s3dis_union')
    avg_tracker.num.pop('test/_s3dis_target')

    eps = 1.0e-10
    class_iou = intersection / (union + eps)
    class_accu = intersection / (target + eps)
    miou = class_iou.mean()
    macc = class_accu.mean()
    all_accu = intersection.sum() / (target.sum() + eps)
    metrics = {
        'test/mIoU': miou,
        'test/mAcc': macc,
        'test/allAcc': all_accu,
    }
    for category_id, name in enumerate(S3DIS_CATEGORIES):
      metrics['test/IoU_' + name] = class_iou[category_id]
      metrics['test/Acc_' + name] = class_accu[category_id]
    avg_tracker.update(metrics)

    tqdm.write(
        '=> Epoch: %d, test/mIoU: %.6f, test/mAcc: %.6f, '
        'test/allAcc: %.6f' %
        (epoch, miou.item(), macc.item(), all_accu.item()))
    category_msg = ', '.join(
        '%s: %.4f/%.4f' %
        (name, class_iou[i].item(), class_accu[i].item())
        for i, name in enumerate(S3DIS_CATEGORIES))
    tqdm.write('=> S3DIS class IoU/Acc: ' + category_msg)

  def semantickitti_result_callback(self, avg_tracker, epoch):
    r'''Computes the official 19-class SemanticKITTI validation metrics.'''
    submitted_key = 'test/_semantickitti_submitted'
    if submitted_key in avg_tracker.value:
      avg_tracker.value.pop(submitted_key)
      avg_tracker.num.pop(submitted_key)
      tqdm.write(
          '=> SemanticKITTI submission files were written to: %s' %
          str(getattr(
              self.FLAGS.DATA.test, 'submission_dir', '') or
              os.path.join(self.logdir, 'submission')))
      return

    device = next(iter(avg_tracker.value.values())).device
    statistics = {'intersection': [], 'union': [], 'target': []}
    for statistic in statistics:
      for category_id in range(len(SEMANTICKITTI_CATEGORIES)):
        key = 'test/_semantickitti_%s_%d' % (statistic, category_id)
        if key not in avg_tracker.value:
          raise RuntimeError(
              'SemanticKITTI metric accumulator is missing: ' + key)
        statistics[statistic].append(avg_tracker.value.pop(key))
        avg_tracker.num.pop(key)
    intersection = torch.stack(statistics['intersection']).to(device)
    union = torch.stack(statistics['union']).to(device)
    target = torch.stack(statistics['target']).to(device)

    eps = 1.0e-10
    class_iou = intersection / (union + eps)
    class_accu = intersection / (target + eps)
    miou = class_iou.mean()
    macc = class_accu.mean()
    all_accu = intersection.sum() / (target.sum() + eps)
    metrics = {
        'test/mIoU': miou,
        'test/mAcc': macc,
        'test/allAcc': all_accu,
    }
    for category_id, name in enumerate(SEMANTICKITTI_CATEGORIES):
      metrics['test/IoU_' + name] = class_iou[category_id]
      metrics['test/Acc_' + name] = class_accu[category_id]
    avg_tracker.update(metrics)

    tqdm.write(
        '=> Epoch: %d, test/mIoU: %.6f, test/mAcc: %.6f, '
        'test/allAcc: %.6f' %
        (epoch, miou.item(), macc.item(), all_accu.item()))
    category_msg = ', '.join(
        '%s: %.4f/%.4f' %
        (name, class_iou[i].item(), class_accu[i].item())
        for i, name in enumerate(SEMANTICKITTI_CATEGORIES))
    tqdm.write('=> SemanticKITTI class IoU/Acc: ' + category_msg)

  def loss_function(self, logit, label):
    if self.is_semantickitti():
      class_weight = logit.new_tensor(SEMANTICKITTI_CE_WEIGHTS)
      criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
      criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logit, label.long())
    if self.is_s3dis() or self.is_semantickitti() or self.is_partnete():
      ce_weight = float(getattr(self.FLAGS.LOSS, 'ce_weight', 1.0))
      lovasz_weight = float(getattr(self.FLAGS.LOSS, 'lovasz_weight', 0.0))
      loss = ce_weight * loss
      if lovasz_weight > 0:
        loss = loss + lovasz_weight * lovasz_softmax_loss(
            logit, label.long(), ignore_index=self.FLAGS.LOSS.mask)
    return loss

  def accuracy(self, logit, label):
    pred = logit.argmax(dim=1)
    accu = pred.eq(label).float().mean()
    return accu

  def IoU_per_shape(self, logit, label, class_num):
    pred = logit.argmax(dim=1)

    IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk, lk = pred.eq(k), label.eq(k)
      intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
      union[k] = torch.sum(torch.logical_or(pk, lk).float())

      valid = torch.sum(lk.any()) > 0
      valid_part_num += valid.item()
      IoU += valid * intsc[k] / (union[k] + esp)

    # Calculate the shape IoU for ShapeNet
    IoU /= valid_part_num + esp
    return IoU, intsc, union


if __name__ == "__main__":
  SegSolver.main()
