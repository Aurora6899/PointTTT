import random
from typing import List

import dwconv
import ocnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ocnn.octree import Octree
from torch.utils.checkpoint import checkpoint

from .point_ttt_hierarchical import HierarchicalPointTTTLayer
from .ttt import TTTConfig, TTTLinear, TTTMLP



MULTI_SERIALIZATION_CONFIG = {
    'enabled': False,
    'strategy': 'random',
    'methods': ['z_order', 'trans_z', 'hilbert', 'trans_hilbert'],
    'debug': False
}


MULTI_SERIALIZATION_AVAILABLE = MULTI_SERIALIZATION_CONFIG['enabled']

try:
    from .multi_serialization import multi_xyz2key
    if MULTI_SERIALIZATION_CONFIG['debug']:
        print("🎯 Multi-serialization module loaded successfully!")
        print(f"📋 Available methods: {MULTI_SERIALIZATION_CONFIG['methods']}")
        print(f"🚀 Strategy: {MULTI_SERIALIZATION_CONFIG['strategy']}")
except ImportError as e:
    print(f"Warning: Multi-serialization not available: {e}")
    MULTI_SERIALIZATION_AVAILABLE = False
    multi_xyz2key = None


def _remap_state_dict_prefix(state_dict, old_prefix, new_prefix):
    """Move legacy checkpoint keys to their current module prefix."""
    for old_key in [key for key in state_dict if key.startswith(old_prefix)]:
        new_key = new_prefix + old_key[len(old_prefix):]
        if new_key not in state_dict:
            state_dict[new_key] = state_dict[old_key]
        state_dict.pop(old_key)



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

class PointTTTBlock(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        

        self.point_ttt = OctreeTTT(
            dim=dim, 
            proj_drop=proj_drop,
            nempty=nempty,
            partition_by_batch=kwargs.get('partition_by_batch', False),
            ttt_base_lr=kwargs.get('ttt_base_lr', 1.0),
            ttt_update_train=kwargs.get('ttt_update_train', True),
            ttt_update_test=kwargs.get('ttt_update_test', True),
            ttt_patch_size=kwargs.get('ttt_patch_size', 64),
            ttt_num_heads=kwargs.get('ttt_num_heads', 24),
            ttt_layer_type=kwargs.get('ttt_layer_type', 'linear'),
            pointttt_hierarchical_enabled=kwargs.get(
                'pointttt_hierarchical_active', False),
            pointttt_global_chunk_size=kwargs.get(
                'pointttt_global_chunk_size', 128),
            pointttt_summary_tokens=kwargs.get(
                'pointttt_summary_tokens', 1),
            pointttt_global_bidirectional=kwargs.get(
                'pointttt_global_bidirectional', True),
            pointttt_global_gate_init=kwargs.get(
                'pointttt_global_gate_init', 0.0),
        )
        
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints created before the PointTTT rename used this module key.
        legacy_name = 'mamba.'
        _remap_state_dict_prefix(
            state_dict, prefix + legacy_name, prefix + 'point_ttt.')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.cpe(data, octree, depth) + data
        ttt_output = self.point_ttt(self.norm1(data), octree, depth)
        data = data + self.drop_path(ttt_output, octree, depth)
        return data


class PointTTTStage(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 use_checkpoint: bool = True, num_blocks: int = 2,
                 block_class=PointTTTBlock,
                 **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        

        config = MULTI_SERIALIZATION_CONFIG
        if MULTI_SERIALIZATION_AVAILABLE and config['enabled'] and config['strategy'] != 'z_order':
            if config['debug']:
                print(f"[Warning] Disabling checkpoint due to multi-serialization (strategy: {config['strategy']})")
            self.use_checkpoint = False
        else:
            self.use_checkpoint = use_checkpoint
            
        stage_idx = int(kwargs.get('pointttt_stage_idx', -1))
        hierarchical_enabled = bool(
            kwargs.get('pointttt_hierarchical_enabled', False))
        hierarchical_stages = tuple(int(stage) for stage in kwargs.get(
            'pointttt_hierarchical_stages', []))
        hierarchical_interval = int(kwargs.get(
            'pointttt_hierarchical_block_interval', 0))
        blocks = []
        for i in range(num_blocks):
            if hierarchical_interval > 0:
                selected_block = ((i + 1) % hierarchical_interval == 0 or
                                  i == num_blocks - 1)
            else:
                selected_block = i == num_blocks - 1
            block_kwargs = dict(kwargs)
            block_kwargs['pointttt_hierarchical_active'] = (
                hierarchical_enabled and stage_idx in hierarchical_stages and
                selected_block)
            blocks.append(block_class(
                dim=dim,
                proj_drop=proj_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                nempty=nempty,
                **block_kwargs,
            ))
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
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


class PointTTT(torch.nn.Module):
    def __init__(self, in_channels: int,
                 channels: List[int] = [96, 192, 384, 384],
                 num_blocks: List[int] = [2, 2, 18, 2],
                 drop_path: float = 0.5,
                 nempty: bool = True, stem_down: int = 2, 
                 **kwargs):
        super().__init__()
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        layers = []
        for i in range(self.num_stages):
            stage_kwargs = dict(kwargs)
            stage_kwargs['pointttt_stage_idx'] = i
            layers.append(PointTTTStage(
                dim=channels[i],
                drop_path=drop_ratio[
                    sum(num_blocks[:i]):sum(num_blocks[:i + 1])],
                nempty=nempty,
                num_blocks=num_blocks[i],
                **stage_kwargs,
            ))
        self.layers = torch.nn.ModuleList(layers)
        
        self.downsamples = torch.nn.ModuleList([Downsample(
            channels[i], channels[i + 1], kernel_size=[2],
            nempty=nempty) for i in range(self.num_stages - 1)])

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        depth = depth - self.stem_down
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)
        return features

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
                 partition_by_batch: bool = False,
                 ttt_base_lr: float = 1.0,
                 ttt_update_train: bool = True,
                 ttt_update_test: bool = True,
                 ttt_layer_type: str = 'linear'):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.ttt_layer_type = str(ttt_layer_type).lower()
        self.dilation = dilation
        self.use_rpe = use_rpe
        # Keep the historical flat sequence as the default so existing
        # classification/segmentation experiments are bit-for-bit unchanged.
        # Detection enables this option to prevent TTT chunks from crossing
        # point-cloud boundaries in a multi-sample batch.
        self.partition_by_batch = partition_by_batch

        ttt_layer_type = self.ttt_layer_type
        ttt_layer_classes = {
            'linear': TTTLinear,
            'mlp': TTTMLP,
        }
        if ttt_layer_type not in ttt_layer_classes:
            raise ValueError(
                f'Unsupported ttt_layer_type {ttt_layer_type!r}; '
                f'choose from {tuple(ttt_layer_classes)}.')



        self.config = TTTConfig(
            #vocab_size=32000,
            hidden_size=dim,
            intermediate_size=dim * 4,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            ttt_layer_type=ttt_layer_type,
            ttt_base_lr=ttt_base_lr,
            ttt_update_train=ttt_update_train,
            ttt_update_test=ttt_update_test,
            mini_batch_size=patch_size,
            use_cache=False,
            share_qk=True,
            use_gate=True,
            pre_conv=True,
            tie_word_embeddings=False,
        )


        ttt_layer_class = ttt_layer_classes[ttt_layer_type]
        self.ttt_forward = ttt_layer_class(
            self.config, layer_idx=0)
        self.ttt_backward = ttt_layer_class(
            self.config, layer_idx=1)


        self.gate_forward = nn.Parameter(torch.tensor(0.1))
        self.gate_backward = nn.Parameter(torch.tensor(0.1))


        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device):
        """生成位置索引（前向用正常顺序，反向用反转顺序）"""
        return torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def _run_bidirectional(self, x: torch.Tensor):
        """Run both TTT directions on a dense ``[B, L, C]`` sequence."""
        B_seq, seq_len, _ = x.shape
        # --------------------------

        # --------------------------
        pos_forward = self._build_position_ids(B_seq, seq_len, x.device)
        out_forward = self.ttt_forward(
            hidden_states=x,
            attention_mask=None,
            position_ids=pos_forward,
            cache_params=None,
        )

        # --------------------------

        # --------------------------

        x_rev = torch.flip(x, dims=[1])
        pos_backward = self._build_position_ids(B_seq, seq_len, x.device)
        

        out_backward_rev = self.ttt_backward(
            hidden_states=x_rev,
            attention_mask=None,
            position_ids=pos_backward,
            cache_params=None,
        )
        

        out_backward = torch.flip(out_backward_rev, dims=[1])

        # --------------------------

        # --------------------------

        gate_f = torch.tanh(self.gate_forward)
        gate_b = torch.tanh(self.gate_backward)
        fused = gate_f * out_forward + gate_b * out_backward  # [B_seq, K, C]


        fused_with_residual = fused + x


        out = self.out_proj(fused_with_residual)
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
                f"ttt_layer_type={self.ttt_layer_type}, "
                f"gate_forward={self.gate_forward.item():.3f}, gate_backward={self.gate_backward.item():.3f}")


class OctreeAdaptiveNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.pixel_norm = nn.LayerNorm(dim)
        self.window_norm = nn.LayerNorm(dim)
        
    def forward(self, x, depth):

        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        B, C, L = x.shape
        assert C == self.dim
        if L == 0:
            return x
        

        if depth in (3, 4, 5):
            patch_size = 64
        elif depth in (6, 7, 8, 9):
            patch_size = 24
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        

        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            pad_idx = torch.arange(pad_len, device=x.device) % L
            borrowed_points = x.index_select(2, pad_idx)
            x_padded = torch.cat([x, borrowed_points], dim=2)
            L_padded = L + pad_len
        else:
            x_padded = x
            L_padded = L
        

        num_patches = L_padded // patch_size
        

        x_div = x_padded.reshape(B, C, num_patches, patch_size)
        x_div = x_div.permute(0, 3, 1, 2).contiguous()
        x_div = x_div.view(B * patch_size, C, num_patches)
        x_flat = x_div.transpose(1, 2)
        

        x_norm = self.pixel_norm(x_flat)
        

        x_out = x_norm.reshape(B, patch_size, num_patches, C)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        x_out = x_out.reshape(B, C, L_padded)
        

        pixel_output = x_out + x_padded
        


        num_patches_win = L_padded // patch_size
        

        pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        unpool = nn.Upsample(scale_factor=patch_size, mode='nearest')
        

        x_div_win = pool(pixel_output)
        x_flat_win = x_div_win.transpose(1, 2)
        x_norm_win = self.window_norm(x_flat_win)
        x_out_win = x_norm_win.transpose(1, 2)
        x_out_win = unpool(x_out_win)

        window_output = x_out_win + pixel_output

        if L_padded != L:
            window_output = window_output[:, :, :L]
        return window_output


class SerializationPerformanceEvaluator(nn.Module):
    """评估不同序列化方法的性能"""

    def evaluate_locality_preservation(self, data, octree, depth, method):
        """评估空间局部性保持能力"""
        try:
            key = octree.key(depth, octree.nempty)
            if key.numel() == 0 or key.numel() != data.shape[0]:
                return 0.5
                
            from ocnn.octree.shuffled_key import key2xyz
            x, y, z, b = key2xyz(key, depth)
            xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
            

            if method != 'z_order' and MULTI_SERIALIZATION_AVAILABLE:
                new_key = multi_xyz2key(x, y, z, b, depth, method)
                _, sort_idx = torch.sort(new_key)
            else:
                _, sort_idx = torch.sort(key)
            

            locality_score = self._compute_locality_score(xyz, sort_idx)
            return locality_score.item()
            
        except Exception as e:
            return 0.5
    
    def _compute_locality_score(self, xyz, sort_idx):
        """计算序列化后的局部性保持分数"""
        if len(xyz) < 2:
            return torch.tensor(0.5)
            

        sorted_xyz = xyz[sort_idx]
        consecutive_distances = torch.norm(
            sorted_xyz[1:] - sorted_xyz[:-1], dim=1
        )
        

        random_idx = torch.randperm(len(xyz), device=xyz.device)
        random_xyz = xyz[random_idx]
        random_distances = torch.norm(
            random_xyz[1:] - random_xyz[:-1], dim=1
        )
        

        locality_score = random_distances.mean() / (consecutive_distances.mean() + 1e-6)
        return torch.clamp(locality_score, 0.0, 1.0)


class AdaptiveSerializationSelector(nn.Module):
    """智能序列化方法选择器"""
    def __init__(self, feature_dim, num_methods):
        super().__init__()
        

        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        

        self.method_selector = nn.Linear(64, num_methods)
        

        self.performance_predictor = nn.Linear(64, num_methods)
        

        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features, training=True):
        h = self.feature_extractor(features)
        

        method_logits = self.method_selector(h)
        
        if training:

            probs = F.softmax(method_logits / self.temperature, dim=-1)
            method_idx = torch.multinomial(probs, 1).item()
        else:

            method_idx = method_logits.argmax().item()
        

        performance_pred = self.performance_predictor(h)
        
        return method_idx, method_logits, performance_pred


class MultiSerializationBiTTTLayer(BiTTTLayer):
    def __init__(self, dim, patch_size=64, num_heads=24, proj_drop=0.0,
                 partition_by_batch=False, ttt_base_lr=1.0,
                 ttt_update_train=True, ttt_update_test=True,
                 ttt_layer_type='linear'):
        super().__init__(dim, patch_size, num_heads, proj_drop,
                         partition_by_batch=partition_by_batch,
                         ttt_base_lr=ttt_base_lr,
                         ttt_update_train=ttt_update_train,
                         ttt_update_test=ttt_update_test,
                         ttt_layer_type=ttt_layer_type)
        

        config = MULTI_SERIALIZATION_CONFIG
        
        if MULTI_SERIALIZATION_AVAILABLE and config['enabled']:

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
        

        self.global_call_count = 0
        

        self.depth_call_counts = {}
        

        if MULTI_SERIALIZATION_AVAILABLE and self.selection_strategy == 'adaptive':

            feature_dim = 15
            
            self.adaptive_selector = AdaptiveSerializationSelector(
                feature_dim, len(self.serialization_methods)
            )
            
            self.performance_evaluator = SerializationPerformanceEvaluator()
            

            self.performance_history = []
            self.update_frequency = 20
            self.call_count = 0
            
            if config['debug']:
                print(f"🧠 Adaptive selector initialized with {len(self.serialization_methods)} methods")
    
    def extract_comprehensive_features(self, data, octree, depth):
        """提取基于八叉树结构的综合特征"""
        features = []
        
        try:

            features.extend([
                depth,
                octree.nnum[depth] if depth < len(octree.nnum) else 0,
                octree.nnum_nempty[depth] if (octree.nempty and depth < len(octree.nnum_nempty)) else 0,
            ])
            

            key = octree.key(depth, octree.nempty)
            if key.numel() > 0:
                from ocnn.octree.shuffled_key import key2xyz
                x, y, z, b = key2xyz(key, depth)
                xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
                

                center = xyz.mean(0)
                distances = torch.norm(xyz - center, dim=1)
                
                features.extend([
                    xyz.std(0).mean().item(),
                    (xyz.max(0)[0] - xyz.min(0)[0]).mean().item(),
                    distances.mean().item(),
                    distances.std().item(),
                    xyz.shape[0] / (8 ** depth),
                ])
            else:
                features.extend([0.0] * 5)
                

            features.extend([
                data.mean().item(),
                data.std().item(),
                (data.max() - data.min()).item(),
                (data > data.mean()).float().mean().item(),
                float(data.shape[0]),
            ])
            

            features.extend([
                float(depth / 10.0),
                float(data.shape[1] / 512.0),
            ])
            
        except Exception as e:

            features = [0.0] * 15
            
        return torch.tensor(features, device=data.device, dtype=torch.float32)

    def select_serialization_method(self, data, octree, depth):
        """智能选择序列化方法 - 保留所有策略"""
        if not MULTI_SERIALIZATION_AVAILABLE:
            return 'z_order'
        

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
        

        elif self.selection_strategy == 'adaptive' and hasattr(self, 'adaptive_selector'):

            features = self.extract_comprehensive_features(data, octree, depth)
            

            method_idx, method_logits, performance_pred = self.adaptive_selector(
                features, training=self.training
            )
            
            selected_method = self.serialization_methods[method_idx]
            

            if self.training:
                self.record_performance(selected_method, data, octree, depth, performance_pred, method_logits)
            
            return selected_method
        

        return 'z_order'
    
    def record_performance(self, method, data, octree, depth, performance_pred, method_logits):
        """记录性能用于自监督学习"""
        self.call_count += 1
        
        if self.call_count % self.update_frequency == 0:

            actual_performance = self.performance_evaluator.evaluate_locality_preservation(
                data, octree, depth, method
            )
            

            self.performance_history.append({
                'method': method,
                'depth': depth,
                'predicted': performance_pred.detach().cpu(),
                'actual': actual_performance,
                'logits': method_logits.detach().cpu(),
            })
            

            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
    
    def compute_adaptive_loss(self):
        """计算自适应学习损失"""
        if len(self.performance_history) < 10:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        losses = []
        for record in self.performance_history[-50:]:
            try:
                method_idx = self.serialization_methods.index(record['method'])
                predicted = record['predicted'][method_idx]
                actual = torch.tensor(record['actual'], device=predicted.device)
                

                perf_loss = F.mse_loss(predicted, actual)
                

                target_probs = torch.zeros_like(record['logits'])
                target_probs[method_idx] = record['actual']
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

        selected_method = self.select_serialization_method(data, octree, depth)
        
        config = MULTI_SERIALIZATION_CONFIG
        if config.get('debug', False):
            print(f"[Multi-Serialization] Depth {depth}: Using {selected_method} (Strategy: {self.selection_strategy})")
        
        if selected_method == 'z_order':

            result = super().forward(data, octree, depth)
        
        elif selected_method in ['trans_z', 'hilbert', 'trans_hilbert'] and MULTI_SERIALIZATION_AVAILABLE:

            try:
                key = octree.key(depth, octree.nempty)
                if key.numel() > 0 and key.numel() == data.shape[0]:
                    from ocnn.octree.shuffled_key import key2xyz
                    x, y, z, b = key2xyz(key, depth)
                    

                    new_key = multi_xyz2key(x, y, z, b, depth, selected_method)
                    

                    _, sort_indices = torch.sort(new_key)
                    _, original_indices = torch.sort(key)
                    
                    reorder_map = torch.empty_like(sort_indices)
                    reorder_map[original_indices] = sort_indices
                    
                    reordered_data = data[reorder_map]
                    result = super().forward(reordered_data, octree, depth)
                    

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

            result = super().forward(data, octree, depth)
        

        if (self.training and self.selection_strategy == 'adaptive' and 
            hasattr(self, 'adaptive_selector') and len(self.performance_history) > 10):
            

            adaptive_loss = self.compute_adaptive_loss()
            if adaptive_loss.requires_grad and adaptive_loss.item() > 0:

                try:
                    adaptive_loss.backward(retain_graph=True)
                except Exception as e:
                    if config.get('debug', False):
                        print(f"Adaptive loss backward failed: {e}")
        
        return result


class OctreeTTT(nn.Module):
    """支持多序列化策略的八叉树 TTT 模块。"""
    
    def __init__(self, dim: int, proj_drop: float = 0.0, 
                 ttt_patch_size: int = 64, ttt_num_heads: int = 24,
                 nempty: bool = True,
                 partition_by_batch: bool = False,
                 ttt_base_lr: float = 1.0,
                 ttt_update_train: bool = True,
                 ttt_update_test: bool = True,
                 ttt_layer_type: str = 'linear',
                 pointttt_hierarchical_enabled: bool = False,
                 pointttt_global_chunk_size: int = 128,
                 pointttt_summary_tokens: int = 1,
                 pointttt_global_bidirectional: bool = True,
                 pointttt_global_gate_init: float = 0.0):
        super().__init__()

        self.octree_ttt = MultiSerializationBiTTTLayer(
            dim=dim,
            patch_size=ttt_patch_size,
            num_heads=ttt_num_heads,
            proj_drop=proj_drop,
            partition_by_batch=partition_by_batch,
            ttt_base_lr=ttt_base_lr,
            ttt_update_train=ttt_update_train,
            ttt_update_test=ttt_update_test,
            ttt_layer_type=ttt_layer_type,
        )
        self.hierarchical_pointttt = None
        if pointttt_hierarchical_enabled:
            self.hierarchical_pointttt = HierarchicalPointTTTLayer(
                dim=dim,
                local_chunk_size=ttt_patch_size,
                num_heads=ttt_num_heads,
                global_chunk_size=pointttt_global_chunk_size,
                summary_tokens=pointttt_summary_tokens,
                global_bidirectional=pointttt_global_bidirectional,
                global_gate_init=pointttt_global_gate_init,
                nempty=nempty,
                ttt_base_lr=ttt_base_lr,
                ttt_update_train=ttt_update_train,
                ttt_update_test=ttt_update_test,
            )
        
        self.adaptive_norm = OctreeAdaptiveNorm(dim=dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Preserve strict loading for checkpoints created before this rename.
        legacy_name = 'bi_scale_mamba.'
        _remap_state_dict_prefix(
            state_dict, prefix + legacy_name, prefix + 'adaptive_norm.')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def forward(self, data: torch.Tensor, octree, depth: int):

        data_ttt = self.octree_ttt(data, octree, depth)
        if self.hierarchical_pointttt is not None:
            data_ttt = self.hierarchical_pointttt(data_ttt, octree, depth)
        

        normalized_data = data_ttt.unsqueeze(0).permute(0, 2, 1)
        normalized_data = self.adaptive_norm(normalized_data, depth)
        data = normalized_data.permute(0, 2, 1).squeeze(0)
        

        data = self.proj(data)
        data = self.proj_drop(data)
        
        return data
