import math
from typing import Tuple

import torch
import torch.nn as nn

from .ttt import TTTConfig, TTTLinear


class PointTTTGeometrySummary(nn.Module):
  r'''Compresses one local PointTTT chunk into geometry-aware tokens.

  The summary contains feature mean/max statistics and normalized octree
  centroid, scale and occupancy.  It therefore preserves scene layout and
  density information that a feature-only pooling operation would discard.
  '''

  def __init__(self, dim: int, num_tokens: int = 1):
    super().__init__()
    if num_tokens < 1:
      raise ValueError('pointttt_summary_tokens must be positive.')
    self.dim = dim
    self.num_tokens = num_tokens
    self.proj = nn.Sequential(
        nn.Linear(dim * 2 + 7, dim),
        nn.GELU(),
        nn.Linear(dim, dim * num_tokens),
    )
    self.token_embedding = nn.Parameter(torch.zeros(num_tokens, dim))
    self.norm = nn.LayerNorm(dim)
    nn.init.trunc_normal_(self.token_embedding, std=0.02)

  def forward(
      self, features: torch.Tensor, xyz: torch.Tensor,
      chunk_size: int, depth: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r'''Returns ``[num_chunks, num_tokens, C]`` and chunk point counts.'''
    num_points, dim = features.shape
    if num_points == 0:
      return features.new_empty((0, self.num_tokens, dim)), torch.empty(
          0, dtype=torch.long, device=features.device)

    num_chunks = math.ceil(num_points / chunk_size)
    padded_points = num_chunks * chunk_size
    pad_len = padded_points - num_points
    if pad_len:
      features = torch.cat(
          [features, features.new_zeros((pad_len, dim))], dim=0)
      xyz = torch.cat(
          [xyz, xyz.new_zeros((pad_len, 3))], dim=0)

    chunks = features.view(num_chunks, chunk_size, dim)
    xyz_chunks = xyz.view(num_chunks, chunk_size, 3).float()
    counts = torch.full(
        (num_chunks,), chunk_size, dtype=torch.long, device=features.device)
    if pad_len:
      counts[-1] = chunk_size - pad_len
    valid = torch.arange(
        chunk_size, device=features.device).unsqueeze(0) < counts.unsqueeze(1)
    valid_f = valid.unsqueeze(-1).to(chunks.dtype)

    denominator = counts.clamp_min(1).to(chunks.dtype).view(-1, 1)
    feature_mean = (chunks * valid_f).sum(dim=1) / denominator
    feature_max = chunks.masked_fill(
        ~valid.unsqueeze(-1), torch.finfo(chunks.dtype).min).amax(dim=1)

    # Octree xyz is in [0, 2**depth - 1].  Normalizing to [-1, 1] makes the
    # summary comparable across the different PointTTT backbone stages.
    xyz_denominator = float(max((1 << int(depth)) - 1, 1))
    xyz_chunks = xyz_chunks.mul(2.0 / xyz_denominator).sub(1.0)
    valid_xyz = valid.unsqueeze(-1).to(xyz_chunks.dtype)
    count_xyz = counts.clamp_min(1).float().view(-1, 1)
    centroid = (xyz_chunks * valid_xyz).sum(dim=1) / count_xyz
    relative_xyz = (xyz_chunks - centroid.unsqueeze(1)) * valid_xyz
    scale = torch.sqrt(
        relative_xyz.square().sum(dim=1) / count_xyz + 1.0e-6)
    occupancy = (counts.float() / float(chunk_size)).unsqueeze(1)

    geometry = torch.cat([centroid, scale, occupancy], dim=-1)
    statistics = torch.cat([
        feature_mean, feature_max, geometry.to(feature_mean.dtype)], dim=-1)
    summary = self.proj(statistics).view(
        num_chunks, self.num_tokens, self.dim)
    summary = summary + self.token_embedding.unsqueeze(0)
    return self.norm(summary), counts


class PointTTTGlobalMemory(nn.Module):
  r'''Bidirectional PointTTT-Linear memory over one complete scene summary.

  Unlike local PointTTT, the chunk dimension is kept in the sequence axis.
  ``TTTLinear`` consequently carries its fast weights through all internal
  global mini-batches via its existing scan implementation.
  '''

  def __init__(
      self, dim: int, num_heads: int, chunk_size: int,
      bidirectional: bool = True, ttt_base_lr: float = 1.0,
      ttt_update_train: bool = True, ttt_update_test: bool = True):
    super().__init__()
    if chunk_size < 1:
      raise ValueError('pointttt_global_chunk_size must be positive.')
    if dim % num_heads != 0:
      raise ValueError(
          f'PointTTT global dim {dim} must be divisible by {num_heads} heads.')
    if (dim // num_heads) % 2 != 0:
      raise ValueError(
          'PointTTT global head dimension must be even for rotary embedding.')

    self.dim = dim
    self.num_heads = num_heads
    self.chunk_size = chunk_size
    self.bidirectional = bidirectional
    self.config = TTTConfig(
        hidden_size=dim,
        intermediate_size=dim * 4,
        num_hidden_layers=2 if bidirectional else 1,
        num_attention_heads=num_heads,
        ttt_layer_type='linear',
        ttt_base_lr=ttt_base_lr,
        ttt_update_train=ttt_update_train,
        ttt_update_test=ttt_update_test,
        mini_batch_size=chunk_size,
        use_cache=False,
        share_qk=True,
        use_gate=True,
        pre_conv=True,
        tie_word_embeddings=False,
    )
    self.pointttt_forward = TTTLinear(self.config, layer_idx=0)
    if bidirectional:
      self.pointttt_backward = TTTLinear(self.config, layer_idx=1)
      self.gate_backward = nn.Parameter(torch.tensor(0.1))
    else:
      self.pointttt_backward = None
      self.register_parameter('gate_backward', None)
    self.gate_forward = nn.Parameter(torch.tensor(0.1))
    self.out_proj = nn.Linear(dim, dim)

  @staticmethod
  def _position_ids(x: torch.Tensor):
    return torch.arange(
        x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)

  def forward(self, summary: torch.Tensor):
    if summary.ndim != 3 or summary.shape[0] != 1:
      raise ValueError('PointTTT global memory expects shape [1, L, C].')
    position_ids = self._position_ids(summary)
    forward = self.pointttt_forward(
        hidden_states=summary, attention_mask=None,
        position_ids=position_ids, cache_params=None)
    output = torch.tanh(self.gate_forward) * forward

    if self.pointttt_backward is not None:
      reversed_summary = torch.flip(summary, dims=[1])
      backward = self.pointttt_backward(
          hidden_states=reversed_summary, attention_mask=None,
          position_ids=position_ids, cache_params=None)
      backward = torch.flip(backward, dims=[1])
      output = output + torch.tanh(self.gate_backward) * backward
    return self.out_proj(output)


class HierarchicalPointTTTLayer(nn.Module):
  r'''Adds scene-level PointTTT memory to an existing local PointTTT output.'''

  def __init__(
      self, dim: int, local_chunk_size: int, num_heads: int,
      global_chunk_size: int = 128, summary_tokens: int = 1,
      global_bidirectional: bool = True, global_gate_init: float = 0.0,
      nempty: bool = True, ttt_base_lr: float = 1.0,
      ttt_update_train: bool = True, ttt_update_test: bool = True):
    super().__init__()
    self.dim = dim
    self.local_chunk_size = local_chunk_size
    self.summary_tokens = summary_tokens
    self.nempty = nempty
    self.geometry_summary = PointTTTGeometrySummary(dim, summary_tokens)
    self.global_memory = PointTTTGlobalMemory(
        dim=dim, num_heads=num_heads, chunk_size=global_chunk_size,
        bidirectional=global_bidirectional, ttt_base_lr=ttt_base_lr,
        ttt_update_train=ttt_update_train,
        ttt_update_test=ttt_update_test)
    self.global_proj = nn.Linear(dim, dim)
    self.global_gate = nn.Parameter(torch.tensor(float(global_gate_init)))

  def _forward_scene(
      self, features: torch.Tensor, xyz: torch.Tensor, depth: int):
    summaries, counts = self.geometry_summary(
        features, xyz, self.local_chunk_size, depth)
    num_chunks = summaries.shape[0]
    summary_sequence = summaries.reshape(
        1, num_chunks * self.summary_tokens, self.dim)

    # Keeping batch size one is essential: TTTLinear.scan then propagates the
    # final fast weight of one global mini-batch into the next mini-batch.
    global_tokens = self.global_memory(summary_sequence)
    global_chunks = global_tokens.view(
        num_chunks, self.summary_tokens, self.dim).mean(dim=1)
    return torch.repeat_interleave(global_chunks, counts, dim=0)

  def forward(self, local_features: torch.Tensor, octree, depth: int):
    if local_features.numel() == 0:
      return local_features
    x, y, z, batch_id = octree.xyzb(depth, nempty=self.nempty)
    if batch_id.numel() != local_features.shape[0]:
      raise RuntimeError(
          f'PointTTT geometry/data mismatch at depth {depth}: '
          f'{batch_id.numel()} coordinates for {local_features.shape[0]} features.')
    xyz = torch.stack([x, y, z], dim=-1).to(local_features.device)
    batch_id = batch_id.to(local_features.device).long()

    global_context = torch.empty_like(local_features)
    for scene_id in torch.unique(batch_id, sorted=True):
      indices = torch.nonzero(
          batch_id == scene_id, as_tuple=False).flatten()
      scene_features = local_features.index_select(0, indices)
      scene_xyz = xyz.index_select(0, indices)
      scene_context = self._forward_scene(scene_features, scene_xyz, depth)
      global_context = global_context.index_copy(0, indices, scene_context)

    global_delta = self.global_proj(global_context)
    return local_features + torch.tanh(self.global_gate) * global_delta

  def extra_repr(self):
    return (
        f'dim={self.dim}, local_chunk_size={self.local_chunk_size}, '
        f'summary_tokens={self.summary_tokens}, nempty={self.nempty}')
