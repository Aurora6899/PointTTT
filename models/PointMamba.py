import torch
import ocnn
import dwconv
from ocnn.octree import Octree
from typing import Optional, List
from torch.utils.checkpoint import checkpoint
from typing import Optional
# 导入 Mamba2 类，实现见下方
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from torch.cuda.amp import autocast
import copy
import random  # 添加random导入

# ===== 🎯 全局配置区域 - 只需要修改这里！ =====
# 多序列化配置
MULTI_SERIALIZATION_CONFIG = {
    'enabled': False,                    # 是否启用多序列化,运行分割的代码时修改为False才不会报错
    'strategy': 'random',               # 策略：'sequential', 'random', 'adaptive'
    'methods': ['z_order', 'trans_z', 'hilbert', 'trans_hilbert'],  # 可用方法
    'debug': False                      # 是否打印调试信息
}

# 便捷的策略切换函数
def switch_strategy(strategy):
    """快速切换策略：'sequential', 'random', 'adaptive'"""
    assert strategy in ['sequential', 'random', 'adaptive'], f"Invalid strategy: {strategy}"
    MULTI_SERIALIZATION_CONFIG['strategy'] = strategy
    print(f"🔄 Strategy switched to: {strategy}")

def enable_debug():
    """启用调试模式"""
    MULTI_SERIALIZATION_CONFIG['debug'] = True
    print("🐛 Debug mode enabled")

def disable_debug():
    """禁用调试模式"""
    MULTI_SERIALIZATION_CONFIG['debug'] = False
    print("✅ Debug mode disabled")
# =============================================

# 启用多序列化功能
MULTI_SERIALIZATION_AVAILABLE = MULTI_SERIALIZATION_CONFIG['enabled']

try:
    from .multi_serialization import multi_xyz2key, multi_key2xyz
    if MULTI_SERIALIZATION_CONFIG['debug']:
        print("🎯 Multi-serialization module loaded successfully!")
        print(f"📋 Available methods: {MULTI_SERIALIZATION_CONFIG['methods']}")
        print(f"🚀 Strategy: {MULTI_SERIALIZATION_CONFIG['strategy']}")
except ImportError as e:
    print(f"Warning: Multi-serialization not available: {e}")
    MULTI_SERIALIZATION_AVAILABLE = False
    multi_xyz2key = None
    multi_key2xyz = None



class OctreeT(Octree):

    def __init__(self, octree: Octree, patch_size: int = 24, dilation: int = 4,
                 nempty: bool = True, max_depth: Optional[int] = None,
                 start_depth: Optional[int] = None, **kwargs):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)

        self.patch_size = patch_size
        self.dilation = dilation  # TODO dilation as a list
        self.nempty = nempty
        self.max_depth = max_depth or self.depth
        self.start_depth = start_depth or self.full_depth
        self.invalid_mask_value = -1e3
        assert self.start_depth > 1

        self.block_num = patch_size * dilation
        self.nnum_t = self.nnum_nempty if nempty else self.nnum
        self.nnum_a = ((self.nnum_t / self.block_num).ceil() * self.block_num).int()

        num = self.max_depth + 1
        self.batch_idx = [None] * num
        self.patch_mask = [None] * num
        self.dilate_mask = [None] * num
        self.rel_pos = [None] * num
        self.dilate_pos = [None] * num
        self.build_t()

    def build_t(self):
        for d in range(self.start_depth, self.max_depth + 1):
            self.build_batch_idx(d)
            self.build_attn_mask(d)
            self.build_rel_pos(d)

    def build_batch_idx(self, depth: int):
        batch = self.batch_id(depth, self.nempty)
        self.batch_idx[depth] = self.patch_partition(batch, depth, self.batch_size)

    def build_attn_mask(self, depth: int):
        batch = self.batch_idx[depth]
        mask = batch.view(-1, self.patch_size)
        self.patch_mask[depth] = self._calc_attn_mask(mask)

        mask = batch.view(-1, self.patch_size, self.dilation)
        mask = mask.transpose(1, 2).reshape(-1, self.patch_size)
        self.dilate_mask[depth] = self._calc_attn_mask(mask)

    def _calc_attn_mask(self, mask: torch.Tensor):
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, self.invalid_mask_value)
        return attn_mask

    def build_rel_pos(self, depth: int):
        key = self.key(depth, self.nempty)
        key = self.patch_partition(key, depth)
        x, y, z, _ = ocnn.octree.key2xyz(key, depth)
        xyz = torch.stack([x, y, z], dim=1)

        xyz = xyz.view(-1, self.patch_size, 3)
        self.rel_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

        xyz = xyz.view(-1, self.patch_size, self.dilation, 3)
        xyz = xyz.transpose(1, 2).reshape(-1, self.patch_size, 3)
        self.dilate_pos[depth] = xyz.unsqueeze(2) - xyz.unsqueeze(1)

    def patch_partition(self, data: torch.Tensor, depth: int, fill_value=0):
        num = self.nnum_a[depth] - self.nnum_t[depth]
        tail = data.new_full((num,) + data.shape[1:], fill_value)
        return torch.cat([data, tail], dim=0)

    def patch_reverse(self, data: torch.Tensor, depth: int):
        return data[:self.nnum_t[depth]]

class MLP(torch.nn.Module):

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, activation=torch.nn.GELU,
                 drop: float = 0.0, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = torch.nn.Linear(self.in_features, self.hidden_features)
        self.act = activation()
        self.fc2 = torch.nn.Linear(self.hidden_features, self.out_features)
        self.drop = torch.nn.Dropout(drop, inplace=True)

    def forward(self, data: torch.Tensor):
        data = self.fc1(data)
        data = self.act(data)
        data = self.drop(data)
        data = self.fc2(data)
        data = self.drop(data)
        return data

class OctreeDWConvBn(torch.nn.Module):

    def __init__(self, in_channels: int, kernel_size: List[int] = [3],
                 stride: int = 1, nempty: bool = False):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out

class RPE(torch.nn.Module):#相对位置编码
    def __init__(self, patch_size: int, num_heads: int, dilation: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.pos_bnd = self.get_pos_bnd(patch_size)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def get_pos_bnd(self, patch_size: int):
        return int(0.8 * patch_size * self.dilation ** 0.5)

    def xyz2idx(self, xyz: torch.Tensor):
        mul = torch.arange(3, device=xyz.device) * self.rpe_num
        xyz = xyz.clamp(-self.pos_bnd, self.pos_bnd)
        idx = xyz + (self.pos_bnd + mul)
        return idx

    def forward(self, xyz):
        idx = self.xyz2idx(xyz)
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

    def extra_repr(self) -> str:
        return 'num_heads={}, pos_bnd={}, dilation={}'.format(
            self.num_heads, self.pos_bnd, self.dilation)  # noqa

    def extra_repr(self) -> str:
        return 'dim={}, patch_size={}, num_heads={}, dilation={}'.format(
            self.dim)  # noqa



class PointMambaBlock(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, 
                 **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        
        # 使用OctreeMamba_5（不再传递参数，使用全局配置）
        self.mamba = OctreeMamba_5(
            dim=dim, 
            proj_drop=proj_drop,
            partition_by_batch=kwargs.get('partition_by_batch', False),
        )
        
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.mamba(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data


class PointMambaStage(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 use_checkpoint: bool = True, num_blocks: int = 2,
                 pim_block=PointMambaBlock,
                 **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        
        # 使用全局配置检查是否需要禁用checkpoint
        config = MULTI_SERIALIZATION_CONFIG
        if MULTI_SERIALIZATION_AVAILABLE and config['enabled'] and config['strategy'] != 'z_order':
            if config['debug']:
                print(f"[Warning] Disabling checkpoint due to multi-serialization (strategy: {config['strategy']})")
            self.use_checkpoint = False
        else:
            self.use_checkpoint = use_checkpoint
            
        self.interval = interval
        self.num_norms = (num_blocks - 1) // self.interval

        self.blocks = torch.nn.ModuleList([pim_block(
            dim=dim,
            proj_drop=proj_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            nempty=nempty, 
            activation=activation,
            **kwargs,
        ) for i in range(num_blocks)])

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(self.blocks[i], data, octree, depth, use_reentrant=False)
            else:
                data = self.blocks[i](data, octree, depth)
        return data


class PatchEmbed(torch.nn.Module):
    def __init__(self, in_channels: int = 3, dim: int = 96, num_down: int = 2,
                 nempty: bool = True, **kwargs):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2 ** i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            in_channels if i == 0 else channels[i], channels[i], kernel_size=[3],
            stride=1, nempty=nempty) for i in range(self.num_stages)])
        self.downsamples = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i + 1], kernel_size=[2], stride=2, nempty=nempty)
            for i in range(self.num_stages)])
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: List[int] = [2], nempty: bool = True):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(in_channels, out_channels, kernel_size,
                                       stride=2, nempty=nempty, use_bias=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class PointMamba(torch.nn.Module):
    def __init__(self, in_channels: int,
                 channels: List[int] = [96, 192, 384, 384],
                 num_blocks: List[int] = [2, 2, 18, 2],
                 drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, 
                 **kwargs):
        super().__init__()
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        self.layers = torch.nn.ModuleList([PointMambaStage(
            dim=channels[i],
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i + 1])],
            nempty=nempty, 
            num_blocks=num_blocks[i],
            **kwargs,
        ) for i in range(self.num_stages)])
        
        self.downsamples = torch.nn.ModuleList([Downsample(
            channels[i], channels[i + 1], kernel_size=[2],
            nempty=nempty) for i in range(self.num_stages - 1)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        depth = depth - self.stem_down
        octree = OctreeT(octree, patch_size=24, dilation=4, nempty=self.nempty,
                         max_depth=depth, start_depth=depth - self.num_stages + 1)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)
        return features


import torch
import torch.nn as nn
from .ttt import TTTConfig, TTTLinear

class BiTTTLayer(nn.Module):
    """
    基于论文双向TTT思路修改的实现，支持前向-后向数据整合
    核心改动：新增反向TTT分支、序列反转算子、门控融合机制
    """
    def __init__(self,
                 dim: int,
                 patch_size: int,
                 num_heads: int,
                 proj_drop: float = 0.0,
                 dilation: int = 1,
                 use_rpe: bool = True,
                 partition_by_batch: bool = False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe
        # Keep the historical flat sequence as the default so existing
        # classification/segmentation experiments are bit-for-bit unchanged.
        # Detection enables this option to prevent TTT chunks from crossing
        # point-cloud boundaries in a multi-sample batch.
        self.partition_by_batch = partition_by_batch

        # 1. 构建共享的TTT配置（前向/反向分支共享核心参数）
        self.config = TTTConfig(
            #vocab_size=32000,
            hidden_size=dim,
            intermediate_size=dim * 4,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            ttt_layer_type="mlp",  # 论文中双向TTT默认用MLP类型
            ttt_base_lr=1,
            mini_batch_size=patch_size,
            use_cache=False,
            share_qk=True,
            use_gate=True,  # 启用内置门控（与外部融合门控区分）
            pre_conv=True,
            tie_word_embeddings=False,
        )

        # 2. 实例化前向/反向TTT层（共享配置，独立参数）
        self.ttt_forward = TTTLinear(self.config, layer_idx=0)  # 前向分支（处理过去上下文）
        self.ttt_backward = TTTLinear(self.config, layer_idx=1)  # 反向分支（处理未来上下文）

        # 3. 门控融合参数（可学习，控制前向/反向特征的权重）
        self.gate_forward = nn.Parameter(torch.tensor(0.1))  # 初始值设小，避免破坏预训练特征
        self.gate_backward = nn.Parameter(torch.tensor(0.1))

        # 4. 输出投影与dropout（保持接口兼容）
        self.out_proj = nn.Linear(dim, dim)  # 从双向融合特征映射回原维度
        self.proj_drop = nn.Dropout(proj_drop)

        # 打印配置（调试用）
        try:
            import json
            print("--- 双向TTT配置 ---")
            print(json.dumps(self.config.to_dict(), indent=2))
            print("-------------------")
        except Exception:
            pass

    @torch.no_grad()
    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device):
        """生成位置索引（前向用正常顺序，反向用反转顺序）"""
        return torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def _run_bidirectional(self, x: torch.Tensor):
        """Run both TTT directions on a dense ``[B, L, C]`` sequence."""
        B_seq, seq_len, _ = x.shape
        # --------------------------
        # 第一步：前向TTT处理（捕捉过去）
        # --------------------------
        pos_forward = self._build_position_ids(B_seq, seq_len, x.device)
        out_forward = self.ttt_forward(
            hidden_states=x,
            attention_mask=None,
            position_ids=pos_forward,
            cache_params=None,
        )  # [B_seq, K, C]：前向输出（含过去依赖）

        # --------------------------
        # 第二步：反向TTT处理（捕捉未来）
        # --------------------------
        # 1. 序列反转：将未来上下文转为"伪过去"（核心操作）
        x_rev = torch.flip(x, dims=[1])  # 沿时间维度反转 [B_seq, K, C] → [B_seq, K, C]
        pos_backward = self._build_position_ids(B_seq, seq_len, x.device)
        
        # 2. 反向TTT处理反转序列
        out_backward_rev = self.ttt_backward(
            hidden_states=x_rev,
            attention_mask=None,
            position_ids=pos_backward,
            cache_params=None,
        )  # [B_seq, K, C]：反向输出（基于反转序列）
        
        # 3. 反转回原时间顺序：让反向特征与原序列对齐
        out_backward = torch.flip(out_backward_rev, dims=[1])  # [B_seq, K, C]：含未来依赖

        # --------------------------
        # 第三步：门控融合+残差连接（无缝整合）
        # --------------------------
        # 1. 门控机制：平衡前向/反向特征权重（论文公式11-12）
        gate_f = torch.tanh(self.gate_forward)  # 前向权重（-1~1）
        gate_b = torch.tanh(self.gate_backward)  # 反向权重（-1~1）
        fused = gate_f * out_forward + gate_b * out_backward  # [B_seq, K, C]

        # 2. 残差连接：保留原始特征（避免丢失基础信息）
        fused_with_residual = fused + x  # 与输入x残差融合

        # 3. 输出投影与dropout
        out = self.out_proj(fused_with_residual)  # 映射回原维度
        out = self.proj_drop(out)
        return out

    def _forward_flat(self, data: torch.Tensor):
        """Historical implementation used by all pre-existing tasks."""
        N, C = data.shape
        K = self.patch_size
        pad_len = (-N) % K
        if pad_len > 0:
            pad_idx = torch.arange(pad_len, device=data.device) % N
            pad = data.index_select(0, pad_idx).clone()
            data_padded = torch.cat([data, pad], dim=0)
        else:
            data_padded = data

        B_seq = data_padded.shape[0] // K
        out = self._run_bidirectional(data_padded.view(B_seq, K, C))

        # 恢复原始长度（去除填充）
        out = out.reshape(B_seq * K, C)  # [N + pad_len, C]
        if pad_len > 0:
            out = out[:-pad_len]  # [N, C]
        return out

    def _forward_by_batch(self, data: torch.Tensor, octree, depth: int):
        """Process every point cloud independently without padded tail tokens.

        Full chunks from all scenes are evaluated together for efficiency. Tail
        chunks are grouped only when they have the same length, so neither TTT
        direction can observe nodes belonging to a different scene.
        """
        batch_id = octree.batch_id(depth, nempty=True).long()
        if batch_id.numel() != data.shape[0]:
            raise RuntimeError(
                f'Octree/data size mismatch at depth {depth}: '
                f'{batch_id.numel()} batch ids for {data.shape[0]} features')

        K = self.patch_size
        chunk_groups = {}
        index_groups = {}
        for scene_id in torch.unique(batch_id, sorted=True):
            indices = torch.nonzero(batch_id == scene_id, as_tuple=False).flatten()
            count = indices.numel()
            if count == 0:
                continue
            num_full = count // K
            if num_full:
                full_indices = indices[:num_full * K].view(num_full, K)
                index_groups.setdefault(K, []).append(full_indices)
                chunk_groups.setdefault(K, []).append(
                    data.index_select(0, full_indices.reshape(-1)).view(num_full, K, -1))
            tail_len = count - num_full * K
            if tail_len:
                tail_indices = indices[num_full * K:].view(1, tail_len)
                index_groups.setdefault(tail_len, []).append(tail_indices)
                chunk_groups.setdefault(tail_len, []).append(
                    data.index_select(0, tail_indices.reshape(-1)).view(1, tail_len, -1))

        output = torch.empty_like(data)
        for seq_len, chunks in chunk_groups.items():
            x = torch.cat(chunks, dim=0)
            indices = torch.cat(index_groups[seq_len], dim=0).reshape(-1)
            values = self._run_bidirectional(x).reshape(-1, data.shape[1])
            output = output.index_copy(0, indices, values)
        return output

    def forward(self, data: torch.Tensor, octree, depth: int):
        """Apply bidirectional TTT with an optional per-scene batch boundary."""
        if data.numel() == 0:
            return data
        if self.partition_by_batch:
            return self._forward_by_batch(data, octree, depth)
        return self._forward_flat(data)

    def extra_repr(self) -> str:
        return (f"(双向TTT) dim={self.dim}, patch_size={self.patch_size}, "
                f"num_heads={self.num_heads}, dilation={self.dilation}, "
                f"gate_forward={self.gate_forward.item():.3f}, gate_backward={self.gate_backward.item():.3f}")


class BiTTTLayer_star(nn.Module):
    """
    基于论文双向TTT思路修改的实现，支持前向-后向数据整合
    核心改动：新增反向TTT分支、序列反转算子、门控融合机制
    """
    def __init__(self,
                 dim: int,
                 patch_size: int,
                 num_heads: int,
                 proj_drop: float = 0.0,
                 dilation: int = 1,
                 use_rpe: bool = True):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dilation = dilation
        self.use_rpe = use_rpe

        # 1. 构建共享的TTT配置（前向/反向分支共享核心参数）
        self.config = TTTConfig(
            #vocab_size=32000,
            hidden_size=dim,
            intermediate_size=dim * 4,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            ttt_layer_type="mlp",  # 论文中双向TTT默认用MLP类型
            ttt_base_lr=1,
            mini_batch_size=patch_size,
            use_cache=False,
            share_qk=True,
            use_gate=True,  # 启用内置门控（与外部融合门控区分）
            pre_conv=True,
            tie_word_embeddings=False,
        )

        # 2. 实例化前向/反向TTT层（共享配置，独立参数）
        self.ttt_forward = TTTLinear(self.config, layer_idx=0)  # 前向分支（处理过去上下文）
        self.ttt_backward = TTTLinear(self.config, layer_idx=1)  # 反向分支（处理未来上下文）

        # 3. 门控融合参数（可学习，控制前向/反向特征的权重）
        self.gate_forward = nn.Parameter(torch.tensor(0.1))  # 初始值设小，避免破坏预训练特征
        self.gate_backward = nn.Parameter(torch.tensor(0.1))

        # 4. 输出投影：用 Star 替代原始的线性投影 self.out_proj = nn.Linear(dim, dim)
        #    Star 会保持输入/输出形状为 [B_seq, K, dim]，因此不需要在 forward 中做形状改动
        self.out_proj = Star(dim, drop=proj_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 打印配置（调试用）
        try:
            import json
            print("--- 双向TTT配置 ---")
            print(json.dumps(self.config.to_dict(), indent=2))
            print("-------------------")
        except Exception:
            pass

    @torch.no_grad()
    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device):
        """生成位置索引（前向用正常顺序，反向用反转顺序）"""
        return torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def forward(self, data: torch.Tensor, octree, depth: int):
        """
        双向TTT前向传播：
        1. 前向TTT处理原序列（捕捉过去依赖）
        2. 反向TTT处理反转序列（捕捉未来依赖）
        3. 门控融合+残差连接（无缝整合双向信息）
        """
        if data.numel() == 0:
            return data
        N, C = data.shape
        K = self.patch_size  # 每个patch的长度
        pad_len = (-N) % K
        # 填充到patch_size的倍数（保持与原实现兼容）
        if pad_len > 0:
            pad_idx = torch.arange(pad_len, device=data.device) % N
            pad = data.index_select(0, pad_idx).clone()
            data_padded = torch.cat([data, pad], dim=0)  # [N + pad_len, C]
        else:
            data_padded = data
        B_seq = data_padded.shape[0] // K  # 序列批次大小
        x = data_padded.view(B_seq, K, C)  # [B_seq, K, C]：原序列（前向输入）

        # --------------------------
        # 第一步：前向TTT处理（捕捉过去）
        # --------------------------
        pos_forward = self._build_position_ids(B_seq, K, x.device)  # [B_seq, K]
        out_forward = self.ttt_forward(
            hidden_states=x,
            attention_mask=None,
            position_ids=pos_forward,
            cache_params=None,
        )  # [B_seq, K, C]：前向输出（含过去依赖）

        # --------------------------
        # 第二步：反向TTT处理（捕捉未来）
        # --------------------------
        # 1. 序列反转：将未来上下文转为"伪过去"（核心操作）
        x_rev = torch.flip(x, dims=[1])  # 沿时间维度反转 [B_seq, K, C] → [B_seq, K, C]
        pos_backward = self._build_position_ids(B_seq, K, x.device)  # 反向位置索引（仍用正常顺序，因序列已反转）
        
        # 2. 反向TTT处理反转序列
        out_backward_rev = self.ttt_backward(
            hidden_states=x_rev,
            attention_mask=None,
            position_ids=pos_backward,
            cache_params=None,
        )  # [B_seq, K, C]：反向输出（基于反转序列）
        
        # 3. 反转回原时间顺序：让反向特征与原序列对齐
        out_backward = torch.flip(out_backward_rev, dims=[1])  # [B_seq, K, C]：含未来依赖

        # --------------------------
        # 第三步：门控融合+残差连接（无缝整合）
        # --------------------------
        # 1. 门控机制：平衡前向/反向特征权重（论文公式11-12）
        gate_f = torch.tanh(self.gate_forward)  # 前向权重（-1~1）
        gate_b = torch.tanh(self.gate_backward)  # 反向权重（-1~1）
        fused = gate_f * out_forward + gate_b * out_backward  # [B_seq, K, C]

        # 2. 残差连接：保留原始特征（避免丢失基础信息）
        fused_with_residual = fused + x  # 与输入x残差融合

        # 3. 输出投影与dropout
        out = self.out_proj(fused_with_residual)  # 使用 Star 替代 Linear，输入/输出为 [B_seq, K, C]
        out = self.proj_drop(out)

        # 恢复原始长度（去除填充）
        out = out.reshape(B_seq * K, C)  # [N + pad_len, C]
        if pad_len > 0:
            out = out[:-pad_len]  # [N, C]
        return out

    def extra_repr(self) -> str:
        return (f"(双向TTT) dim={self.dim}, patch_size={self.patch_size}, "
                f"num_heads={self.num_heads}, dilation={self.dilation}, "
                f"gate_forward={self.gate_forward.item():.3f}, gate_backward={self.gate_backward.item():.3f}")


class OctreeAdaptiveNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.pixel_norm = nn.LayerNorm(dim)
        self.window_norm = nn.LayerNorm(dim)
        
    def forward(self, x, depth):
        # 类型转换
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        B, C, L = x.shape
        assert C == self.dim
        if L == 0:
            return x
        
        # 动态确定patch_size
        if depth in (3, 4, 5):
            patch_size = 64
        elif depth in (6, 7, 8, 9):
            patch_size = 24
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # ========== 统一填充处理 ==========
        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            pad_idx = torch.arange(pad_len, device=x.device) % L
            borrowed_points = x.index_select(2, pad_idx)
            x_padded = torch.cat([x, borrowed_points], dim=2)
            L_padded = L + pad_len
        else:
            x_padded = x
            L_padded = L
        
        # ========== 第一阶段：像素级处理 ==========
        num_patches = L_padded // patch_size
        
        # 像素级分块处理
        x_div = x_padded.reshape(B, C, num_patches, patch_size)
        x_div = x_div.permute(0, 3, 1, 2).contiguous()
        x_div = x_div.view(B * patch_size, C, num_patches)
        x_flat = x_div.transpose(1, 2)
        
        # 像素级归一化
        x_norm = self.pixel_norm(x_flat)
        
        # 重组输出（保持填充长度）
        x_out = x_norm.reshape(B, patch_size, num_patches, C)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        x_out = x_out.reshape(B, C, L_padded)
        
        # 添加残差（使用填充后的输入）
        pixel_output = x_out + x_padded
        
        # ========== 第二阶段：窗口级处理 ==========
        # 直接使用像素级输出（保持填充长度）
        num_patches_win = L_padded // patch_size
        
        # 动态池化/上采样
        pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        unpool = nn.Upsample(scale_factor=patch_size, mode='nearest')
        
        # 窗口级处理
        x_div_win = pool(pixel_output)
        x_flat_win = x_div_win.transpose(1, 2)
        x_norm_win = self.window_norm(x_flat_win)
        x_out_win = x_norm_win.transpose(1, 2)
        x_out_win = unpool(x_out_win)
        # 添加残差（使用像素级输出）
        window_output = x_out_win + pixel_output
        # ========== 最终裁剪 ==========
        if L_padded != L:
            window_output = window_output[:, :, :L]
        return window_output


'''
3.2 创建新的多序列化OctreeTTT类
在PointMamba.py文件中添加新的类：
'''
# 在文件末尾添加新的类

# 序列化方法性能评估器
class SerializationPerformanceEvaluator(nn.Module):
    """评估不同序列化方法的性能"""
    def __init__(self, dim, num_methods):
        super().__init__()
        self.performance_history = {}
        
    def evaluate_locality_preservation(self, data, octree, depth, method):
        """评估空间局部性保持能力"""
        try:
            key = octree.key(depth, octree.nempty)
            if key.numel() == 0 or key.numel() != data.shape[0]:
                return 0.5  # 默认中性分数
                
            from ocnn.octree.shuffled_key import key2xyz
            x, y, z, b = key2xyz(key, depth)
            xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
            
            # 重新排序后的1D序列
            if method != 'z_order' and MULTI_SERIALIZATION_AVAILABLE:
                new_key = multi_xyz2key(x, y, z, b, depth, method)
                _, sort_idx = torch.sort(new_key)
            else:
                _, sort_idx = torch.sort(key)
            
            # 计算局部性保持分数
            locality_score = self._compute_locality_score(xyz, sort_idx)
            return locality_score.item()
            
        except Exception as e:
            return 0.5  # 出错时返回中性分数
    
    def _compute_locality_score(self, xyz, sort_idx):
        """计算序列化后的局部性保持分数"""
        if len(xyz) < 2:
            return torch.tensor(0.5)
            
        # 在1D序列中相邻的点在3D空间中的平均距离
        sorted_xyz = xyz[sort_idx]
        consecutive_distances = torch.norm(
            sorted_xyz[1:] - sorted_xyz[:-1], dim=1
        )
        
        # 与随机排序的对比
        random_idx = torch.randperm(len(xyz), device=xyz.device)
        random_xyz = xyz[random_idx]
        random_distances = torch.norm(
            random_xyz[1:] - random_xyz[:-1], dim=1
        )
        
        # 分数越高表示局部性越好
        locality_score = random_distances.mean() / (consecutive_distances.mean() + 1e-6)
        return torch.clamp(locality_score, 0.0, 1.0)

# 智能自适应选择网络
class AdaptiveSerializationSelector(nn.Module):
    """智能序列化方法选择器"""
    def __init__(self, feature_dim, num_methods):
        super().__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 方法选择头
        self.method_selector = nn.Linear(64, num_methods)
        
        # 性能预测头（用于自监督学习）
        self.performance_predictor = nn.Linear(64, num_methods)
        
        # 温度参数（用于Gumbel Softmax）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features, training=True):
        h = self.feature_extractor(features)
        
        # 方法选择逻辑
        method_logits = self.method_selector(h)
        
        if training:
            # 训练时使用softmax采样保持探索性
            probs = F.softmax(method_logits / self.temperature, dim=-1)
            method_idx = torch.multinomial(probs, 1).item()
        else:
            # 推理时使用确定性选择
            method_idx = method_logits.argmax().item()
        
        # 性能预测（用于损失计算）
        performance_pred = self.performance_predictor(h)
        
        return method_idx, method_logits, performance_pred

# 改进版的MultiSerializationBiTTTLayer
class MultiSerializationBiTTTLayer(BiTTTLayer):
    def __init__(self, dim, patch_size=64, num_heads=24, proj_drop=0.0,
                 partition_by_batch=False):
        super().__init__(dim, patch_size, num_heads, proj_drop,
                         partition_by_batch=partition_by_batch)
        
        # 使用全局配置
        config = MULTI_SERIALIZATION_CONFIG
        
        if MULTI_SERIALIZATION_AVAILABLE and config['enabled']:
            # 确保z_order在方法列表中
            methods = config['methods'].copy()
            if 'z_order' not in methods:
                methods = ['z_order'] + methods
            
            self.serialization_methods = methods
            self.selection_strategy = config['strategy']
            
            if config['debug']:
                print(f"🚀 Multi-serialization initialized with strategy: {config['strategy']}")
                print(f"📋 Available methods: {methods}")
        else:
            self.serialization_methods = ['z_order']
            self.selection_strategy = 'sequential'
            if config['debug']:
                print("Multi-serialization disabled, using z_order only")
        
        # 全局计数器，用于sequential模式
        self.global_call_count = 0
        
        # 深度特定的计数器
        self.depth_call_counts = {}
        
        # 自适应网络组件
        if MULTI_SERIALIZATION_AVAILABLE and self.selection_strategy == 'adaptive':
            # 特征维度：八叉树结构(8) + 数据特征(5) + 上下文(2) = 15
            feature_dim = 15
            
            self.adaptive_selector = AdaptiveSerializationSelector(
                feature_dim, len(self.serialization_methods)
            )
            
            self.performance_evaluator = SerializationPerformanceEvaluator(
                dim, len(self.serialization_methods)
            )
            
            # 性能历史记录
            self.performance_history = []
            self.update_frequency = 20  # 每N次调用更新一次网络
            self.call_count = 0
            
            if config['debug']:
                print(f"🧠 Adaptive selector initialized with {len(self.serialization_methods)} methods")
    
    def extract_comprehensive_features(self, data, octree, depth):
        """提取基于八叉树结构的综合特征"""
        features = []
        
        try:
            # 1. 基础八叉树信息
            features.extend([
                depth,
                octree.nnum[depth] if depth < len(octree.nnum) else 0,
                octree.nnum_nempty[depth] if (octree.nempty and depth < len(octree.nnum_nempty)) else 0,
            ])
            
            # 2. 空间几何特征
            key = octree.key(depth, octree.nempty)
            if key.numel() > 0:
                from ocnn.octree.shuffled_key import key2xyz
                x, y, z, b = key2xyz(key, depth)
                xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
                
                # 空间分布统计
                center = xyz.mean(0)
                distances = torch.norm(xyz - center, dim=1)
                
                features.extend([
                    xyz.std(0).mean().item(),              # 空间分布方差
                    (xyz.max(0)[0] - xyz.min(0)[0]).mean().item(),  # 空间跨度
                    distances.mean().item(),               # 到中心平均距离
                    distances.std().item(),                # 距离方差
                    xyz.shape[0] / (8 ** depth),           # 节点密度
                ])
            else:
                features.extend([0.0] * 5)
                
            # 3. 数据特征统计
            features.extend([
                data.mean().item(),                         # 特征均值
                data.std().item(),                          # 特征方差
                (data.max() - data.min()).item(),           # 特征动态范围
                (data > data.mean()).float().mean().item(), # 激活稀疏度
                float(data.shape[0]),                       # 数据量
            ])
            
            # 4. 上下文信息
            features.extend([
                float(depth / 10.0),                        # 归一化深度
                float(data.shape[1] / 512.0),               # 归一化特征维度
            ])
            
        except Exception as e:
            # 如果特征提取失败，返回默认特征
            features = [0.0] * 15
            
        return torch.tensor(features, device=data.device, dtype=torch.float32)

    def select_serialization_method(self, data, octree, depth):
        """智能选择序列化方法 - 保留所有策略"""
        if not MULTI_SERIALIZATION_AVAILABLE:
            return 'z_order'
        
        # === 保留原有的sequential和random策略 ===
        if self.selection_strategy == 'sequential':
            method_idx = self.global_call_count % len(self.serialization_methods)
            method = self.serialization_methods[method_idx]
            self.global_call_count += 1
            return method
            
        elif self.selection_strategy == 'sequential_by_depth':
            if depth not in self.depth_call_counts:
                self.depth_call_counts[depth] = 0
            method_idx = self.depth_call_counts[depth] % len(self.serialization_methods)
            method = self.serialization_methods[method_idx]
            self.depth_call_counts[depth] += 1
            return method
            
        elif self.selection_strategy == 'random':
            method = random.choice(self.serialization_methods)
            return method
            
        elif self.selection_strategy == 'random_seeded':
            old_state = random.getstate()
            seed = depth * 1000 + (self.global_call_count % 1000)
            random.seed(seed)
            method = random.choice(self.serialization_methods)
            random.setstate(old_state)
            self.global_call_count += 1
            return method
        
        # === 新的自适应策略实现 ===
        elif self.selection_strategy == 'adaptive' and hasattr(self, 'adaptive_selector'):
            # 提取基于八叉树的综合特征
            features = self.extract_comprehensive_features(data, octree, depth)
            
            # 使用自适应选择器（启用梯度）
            method_idx, method_logits, performance_pred = self.adaptive_selector(
                features, training=self.training
            )
            
            selected_method = self.serialization_methods[method_idx]
            
            # 记录性能用于自监督学习
            if self.training:
                self.record_performance(selected_method, data, octree, depth, performance_pred, method_logits)
            
            return selected_method
        
        # 默认返回z_order
        return 'z_order'
    
    def record_performance(self, method, data, octree, depth, performance_pred, method_logits):
        """记录性能用于自监督学习"""
        self.call_count += 1
        
        if self.call_count % self.update_frequency == 0:
            # 评估实际性能
            actual_performance = self.performance_evaluator.evaluate_locality_preservation(
                data, octree, depth, method
            )
            
            # 记录历史
            self.performance_history.append({
                'method': method,
                'depth': depth,
                'predicted': performance_pred.detach().cpu(),
                'actual': actual_performance,
                'logits': method_logits.detach().cpu(),
            })
            
            # 保持历史记录在合理范围内
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
    
    def compute_adaptive_loss(self):
        """计算自适应学习损失"""
        if len(self.performance_history) < 10:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        losses = []
        for record in self.performance_history[-50:]:  # 使用最近的记录
            try:
                method_idx = self.serialization_methods.index(record['method'])
                predicted = record['predicted'][method_idx]
                actual = torch.tensor(record['actual'], device=predicted.device)
                
                # MSE损失用于性能预测
                perf_loss = F.mse_loss(predicted, actual)
                
                # 基于实际性能调整选择概率的损失
                target_probs = torch.zeros_like(record['logits'])
                target_probs[method_idx] = record['actual']  # 实际性能作为目标概率
                target_probs = F.softmax(target_probs, dim=-1)
                
                pred_probs = F.log_softmax(record['logits'], dim=-1)
                selection_loss = F.kl_div(pred_probs, target_probs, reduction='batchmean')
                
                total_loss = perf_loss + 0.1 * selection_loss
                losses.append(total_loss)
                
            except Exception as e:
                continue
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
    
    def forward(self, data: torch.Tensor, octree, depth: int):
        # 使用改进的选择方法（支持所有策略）
        selected_method = self.select_serialization_method(data, octree, depth)
        
        config = MULTI_SERIALIZATION_CONFIG
        if config.get('debug', False):
            print(f"[Multi-Serialization] Depth {depth}: Using {selected_method} (Strategy: {self.selection_strategy})")
        
        if selected_method == 'z_order':
            # z_order直接使用原始处理
            result = super().forward(data, octree, depth)
        
        elif selected_method in ['trans_z', 'hilbert', 'trans_hilbert'] and MULTI_SERIALIZATION_AVAILABLE:
            # 其他方法需要重排序
            try:
                key = octree.key(depth, octree.nempty)
                if key.numel() > 0 and key.numel() == data.shape[0]:
                    from ocnn.octree.shuffled_key import key2xyz
                    x, y, z, b = key2xyz(key, depth)
                    
                    # 使用新的序列化方法重新编码
                    new_key = multi_xyz2key(x, y, z, b, depth, selected_method)
                    
                    # 重排序
                    _, sort_indices = torch.sort(new_key)
                    _, original_indices = torch.sort(key)
                    
                    reorder_map = torch.empty_like(sort_indices)
                    reorder_map[original_indices] = sort_indices
                    
                    reordered_data = data[reorder_map]
                    result = super().forward(reordered_data, octree, depth)
                    
                    # 恢复原序列
                    inverse_map = torch.empty_like(reorder_map)
                    inverse_map[reorder_map] = torch.arange(len(reorder_map), device=reorder_map.device)
                    result = result[inverse_map]
                else:
                    result = super().forward(data, octree, depth)
                    
            except Exception as e:
                if config.get('debug', False):
                    print(f"Warning: Multi-serialization failed at depth {depth} with method {selected_method}: {e}")
                result = super().forward(data, octree, depth)
        else:
            # 默认处理
            result = super().forward(data, octree, depth)
        
        # 如果是训练模式且使用adaptive策略，计算自适应损失
        if (self.training and self.selection_strategy == 'adaptive' and 
            hasattr(self, 'adaptive_selector') and len(self.performance_history) > 10):
            
            # 计算自适应损失并反向传播（仅用于更新自适应组件）
            adaptive_loss = self.compute_adaptive_loss()
            if adaptive_loss.requires_grad and adaptive_loss.item() > 0:
                # 使用retain_graph=True以免影响主要的梯度流
                try:
                    adaptive_loss.backward(retain_graph=True)
                except Exception as e:
                    if config.get('debug', False):
                        print(f"Adaptive loss backward failed: {e}")
        
        return result


class OctreeMamba_5(nn.Module):
    """支持多序列化策略的OctreeMamba"""
    
    def __init__(self, dim: int, proj_drop: float = 0.0, 
                 ttt_patch_size: int = 64, ttt_num_heads: int = 24,
                 partition_by_batch: bool = False):
        super().__init__()
        self.dim = dim
        
        # 使用多序列化的TTT层（不再传递参数，使用全局配置）
        self.octree_ttt = MultiSerializationBiTTTLayer(
            dim=dim,
            patch_size=ttt_patch_size,
            num_heads=ttt_num_heads,
            proj_drop=proj_drop,
            partition_by_batch=partition_by_batch,
        )
        
        # 保持原有的其他组件
        self.bi_scale_mamba = OctreeAdaptiveNorm(dim=dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree, depth: int):
        N, C = data.shape
        
        # 1. 使用多序列化TTT处理
        data_ttt = self.octree_ttt(data, octree, depth)
        
        # 2. 后续处理保持不变
        data_mamba = data_ttt.unsqueeze(0).permute(0, 2, 1)
        data_mamba = self.bi_scale_mamba(data_mamba, depth)
        data = data_mamba.permute(0, 2, 1).squeeze(0)
        
        # 3. 最终投影和dropout
        data = self.proj(data)
        data = self.proj_drop(data)
        
        return data
    

# 已修改的 BiTTTLayer（out_proj 替换为 Star），保留原始注释与逻辑
import torch
import torch.nn as nn

class Star(nn.Module):
    """
    Star 模块：用于替代线性投影的非线性映射（保持输入/输出形状为 [B, T, C]）
    说明：
    - 输入/输出形状均为 [B, T, C]（B: patch batch, T: patch length, C: channels）
    - 内部将维度置为 [B, C, T] 用于 1D 卷积运算，最后转回 [B, T, C]
    - depthwise conv 的 padding=1 保持 T 不变，保证残差相加不出错
    """
    def __init__(self, ninp, drop=0.):
        super().__init__()
        # point-wise projections
        self.fc1 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.fc2 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.g   = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # depthwise convolutions: padding=1 保持序列长度不变
        self.dwconv1 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=ninp, bias=True, padding_mode='zeros')
        self.dwconv2 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=ninp, bias=True, padding_mode='zeros')

        self.drop = nn.Dropout(drop)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C]  (与输入形状一致，便于残差连接)
        """
        residual = x  # [B, T, C]
        # 转为 conv 格式 [B, C, T]
        x = x.permute(0, 2, 1)

        # 局部卷积建模
        x = self.dwconv1(x)  # [B, C, T]
        # 两路 1x1 投影与逐元素乘法（star/门控交互）
        x1 = self.fc1(x)     # [B, C, T]
        x2 = self.fc2(x)     # [B, C, T]
        x = self.act(x1) * x2
        # 门控映射 + depthwise conv
        x = self.dwconv2(self.g(x))  # [B, C, T]

        # 回到 [B, T, C]
        x = x.permute(0, 2, 1)

        # 残差相加（输入输出保持相同形状）
        out = residual + x
        return out
