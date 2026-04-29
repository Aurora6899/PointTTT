
import random
from typing import List, Optional

import dwconv
import ocnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ocnn.octree import Octree
from torch.utils.checkpoint import checkpoint

from .ttt import TTTConfig, TTTLinear

# PointTTT serialization policy used by the Adaptive Serialization Router (ASR).
POINTTTT_SERIALIZATION_CONFIG = {
    'enabled': True,
    'strategy': 'adaptive',
    'methods': ['z_order', 'trans_z', 'hilbert', 'trans_hilbert'],
    'debug': False,
}

# Backward-compatible name for existing experiment scripts.
MULTI_SERIALIZATION_CONFIG = POINTTTT_SERIALIZATION_CONFIG


def switch_strategy(strategy):
    """Switch the ASR strategy for ablations."""
    supported = {'sequential', 'sequential_by_depth', 'random',
                 'random_seeded', 'adaptive', 'z_order'}
    assert strategy in supported, f"Invalid strategy: {strategy}"
    MULTI_SERIALIZATION_CONFIG['strategy'] = strategy
    print(f"PointTTT ASR strategy switched to: {strategy}")


def enable_debug():
    MULTI_SERIALIZATION_CONFIG['debug'] = True
    print("PointTTT debug mode enabled")


def disable_debug():
    MULTI_SERIALIZATION_CONFIG['debug'] = False
    print("PointTTT debug mode disabled")

MULTI_SERIALIZATION_AVAILABLE = MULTI_SERIALIZATION_CONFIG['enabled']

try:
    from .multi_serialization import multi_xyz2key, multi_key2xyz
    if MULTI_SERIALIZATION_CONFIG['debug']:
        print("PointTTT multi-serialization module loaded")
        print(f"Available methods: {MULTI_SERIALIZATION_CONFIG['methods']}")
        print(f"ASR strategy: {MULTI_SERIALIZATION_CONFIG['strategy']}")
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
        self.dilation = dilation
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

class RPE(torch.nn.Module):
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



class PointTTTBlock(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, 
                 **kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        
        self.pointttt_op = OctreePointTTT(
            dim=dim, 
            proj_drop=proj_drop
        )
        
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.pointttt_op(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data


class PointTTTStage(torch.nn.Module):
    def __init__(self, dim: int,
                 proj_drop: float = 0.0, drop_path: float = 0.0, nempty: bool = True,
                 activation: torch.nn.Module = torch.nn.GELU, interval: int = 6,
                 use_checkpoint: bool = True, num_blocks: int = 2,
                 pim_block=PointTTTBlock,
                 **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        
        # Disable checkpointing when serialization order changes dynamically.
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
            activation=activation
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


class PointTTT(torch.nn.Module):
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
        self.layers = torch.nn.ModuleList([PointTTTStage(
            dim=channels[i],
            drop_path=drop_ratio[sum(num_blocks[:i]):sum(num_blocks[:i + 1])],
            nempty=nempty, 
            num_blocks=num_blocks[i]
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


class GatingBiTTTLayer(nn.Module):
    """
    PointTTT Gating Bi-TTTLayer.

    The layer runs TTT-Linear on forward and reversed serialized point
    sequences, fuses both directions with learnable gates, and applies the
    Feature Enhancement Projector (FEP) before returning to the point stream.
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

        self.config = TTTConfig(
            hidden_size=dim,
            intermediate_size=dim * 4,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            ttt_layer_type="linear",
            ttt_base_lr=1,
            mini_batch_size=patch_size,
            use_cache=False,
            share_qk=True,
            use_gate=True,
            pre_conv=True,
            tie_word_embeddings=False,
        )

        self.ttt_forward = TTTLinear(self.config, layer_idx=0)
        self.ttt_backward = TTTLinear(self.config, layer_idx=1)

        self.gate_forward = nn.Parameter(torch.tensor(0.1))
        self.gate_backward = nn.Parameter(torch.tensor(0.1))

        self.out_proj = FeatureEnhancementProjector(dim, drop=proj_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device):
        """Build position ids for a serialized patch."""
        return torch.arange(0, seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    def forward(self, data: torch.Tensor, octree, depth: int):
        if data.numel() == 0:
            return data
        N, C = data.shape
        K = self.patch_size
        pad_len = (-N) % K
        if pad_len > 0:
            pad = data[-pad_len:].clone()
            data_padded = torch.cat([data, pad], dim=0)
        else:
            data_padded = data
        B_seq = data_padded.shape[0] // K
        x = data_padded.view(B_seq, K, C)

        pos_forward = self._build_position_ids(B_seq, K, x.device)
        out_forward = self.ttt_forward(
            hidden_states=x,
            attention_mask=None,
            position_ids=pos_forward,
            cache_params=None,
        )

        x_rev = torch.flip(x, dims=[1])
        pos_backward = self._build_position_ids(B_seq, K, x.device)
        
        out_backward_rev = self.ttt_backward(
            hidden_states=x_rev,
            attention_mask=None,
            position_ids=pos_backward,
            cache_params=None,
        )
        
        out_backward = torch.flip(out_backward_rev, dims=[1])

        gate_f = torch.tanh(self.gate_forward)
        gate_b = torch.tanh(self.gate_backward)
        fused = gate_f * out_forward + gate_b * out_backward

        out = self.out_proj(fused + x)
        out = self.proj_drop(out)

        out = out.reshape(B_seq * K, C)
        if pad_len > 0:
            out = out[:-pad_len]
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, patch_size={self.patch_size}, "
                f"num_heads={self.num_heads}, dilation={self.dilation}, "
                f"gate_forward={self.gate_forward.item():.3f}, gate_backward={self.gate_backward.item():.3f}")


class OctreeAdaptiveNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.pixel_norm = nn.LayerNorm(dim)
        self.window_norm = nn.LayerNorm(dim)
        
    def forward(self, x, depth):
        # Normalize in float32 for numerical stability.
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        B, C, L = x.shape
        assert C == self.dim
        
        # Choose the normalization window by octree depth.
        if depth in (3, 4, 5):
            patch_size = 64
        elif depth in (6, 7, 8, 9):
            patch_size = 24
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # Pad to a full normalization window.
        if L % patch_size != 0:
            pad_len = patch_size - (L % patch_size)
            start_idx = L - (L % patch_size) - pad_len
            start_idx = max(start_idx, 0)
            borrowed_points = x[:, :, start_idx:start_idx + pad_len]
            x_padded = torch.cat([x, borrowed_points], dim=2)
            L_padded = L + pad_len
        else:
            x_padded = x
            L_padded = L
        
        # Pixel-level normalization stage.
        num_patches = L_padded // patch_size
        
        # Split features into depth-dependent windows.
        x_div = x_padded.reshape(B, C, num_patches, patch_size)
        x_div = x_div.permute(0, 3, 1, 2).contiguous()
        x_div = x_div.view(B * patch_size, C, num_patches)
        x_flat = x_div.transpose(1, 2)
        
        # Apply pixel-level normalization.
        x_norm = self.pixel_norm(x_flat)
        
        # Restore the padded sequence length.
        x_out = x_norm.reshape(B, patch_size, num_patches, C)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        x_out = x_out.reshape(B, C, L_padded)
        
        # Residual connection on the padded sequence.
        pixel_output = x_out + x_padded
        
        # Window-level normalization stage.
        num_patches_win = L_padded // patch_size
        
        # Pool and unpool over local windows.
        pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        unpool = nn.Upsample(scale_factor=patch_size, mode='nearest')
        
        # Apply window-level normalization.
        x_div_win = pool(pixel_output)
        x_flat_win = x_div_win.transpose(1, 2)
        x_norm_win = self.window_norm(x_flat_win)
        x_out_win = x_norm_win.transpose(1, 2)
        x_out_win = unpool(x_out_win)
        # Residual connection after window normalization.
        window_output = x_out_win + pixel_output
        # Remove padding.
        if L_padded != L:
            window_output = window_output[:, :, :L]
        return window_output


class SerializationPerformanceEvaluator(nn.Module):
    """Estimate locality preservation for candidate serialization methods."""
    def __init__(self, dim, num_methods):
        super().__init__()
        self.performance_history = {}
        
    def evaluate_locality_preservation(self, data, octree, depth, method):
        """Evaluate how well one serialization method preserves locality."""
        try:
            key = octree.key(depth, octree.nempty)
            if key.numel() == 0 or key.numel() != data.shape[0]:
                return 0.5
                
            from ocnn.octree.shuffled_key import key2xyz
            x, y, z, b = key2xyz(key, depth)
            xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
            
            # Build the 1D order produced by the candidate method.
            if method != 'z_order' and MULTI_SERIALIZATION_AVAILABLE:
                new_key = multi_xyz2key(x, y, z, b, depth, method)
                _, sort_idx = torch.sort(new_key)
            else:
                _, sort_idx = torch.sort(key)
            
            # Compute a locality-preservation score.
            locality_score = self._compute_locality_score(xyz, sort_idx)
            return locality_score.item()
            
        except Exception as e:
            return 0.5
    
    def _compute_locality_score(self, xyz, sort_idx):
        """Compute a locality score after serialization."""
        if len(xyz) < 2:
            return torch.tensor(0.5)
            
        # Mean 3D distance between adjacent points in the 1D sequence.
        sorted_xyz = xyz[sort_idx]
        consecutive_distances = torch.norm(
            sorted_xyz[1:] - sorted_xyz[:-1], dim=1
        )
        
        # Compare against a random order.
        random_idx = torch.randperm(len(xyz), device=xyz.device)
        random_xyz = xyz[random_idx]
        random_distances = torch.norm(
            random_xyz[1:] - random_xyz[:-1], dim=1
        )
        
        # Higher scores indicate better locality preservation.
        locality_score = random_distances.mean() / (consecutive_distances.mean() + 1e-6)
        return torch.clamp(locality_score, 0.0, 1.0)

class AdaptiveSerializationRouter(nn.Module):
    """Adaptive Serialization Router (ASR)."""
    def __init__(self, feature_dim, num_methods):
        super().__init__()
        
        # Feature extractor for structural and geometric statistics.
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Serialization-method selector.
        self.method_selector = nn.Linear(64, num_methods)
        
        # Performance predictor used by the self-supervised ASR objective.
        self.performance_predictor = nn.Linear(64, num_methods)
        
        # Temperature used for exploratory sampling.
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features, training=True):
        h = self.feature_extractor(features)
        
        # Method selection.
        method_logits = self.method_selector(h)
        
        if training:
            # Sample during training to preserve exploration.
            probs = F.softmax(method_logits / self.temperature, dim=-1)
            method_idx = torch.multinomial(probs, 1).item()
        else:
            # Use deterministic selection during evaluation.
            method_idx = method_logits.argmax().item()
        
        # Performance prediction used for ASR loss computation.
        performance_pred = self.performance_predictor(h)
        
        return method_idx, method_logits, performance_pred

class AdaptiveSerializationGatingBiTTTLayer(GatingBiTTTLayer):
    def __init__(self, dim, patch_size=64, num_heads=24, proj_drop=0.0):
        super().__init__(dim, patch_size, num_heads, proj_drop)
        
        # Global PointTTT ASR configuration.
        config = MULTI_SERIALIZATION_CONFIG
        
        if MULTI_SERIALIZATION_AVAILABLE and config['enabled']:
            # Ensure that the original Z-order path is always available.
            methods = config['methods'].copy()
            if 'z_order' not in methods:
                methods = ['z_order'] + methods
            
            self.serialization_methods = methods
            self.selection_strategy = config['strategy']
            
            if config['debug']:
                print(f"PointTTT ASR initialized with strategy: {config['strategy']}")
                print(f"Available serialization methods: {methods}")
        else:
            self.serialization_methods = ['z_order']
            self.selection_strategy = 'sequential'
            if config['debug']:
                print("Multi-serialization disabled, using z_order only")
        
        # Counters for deterministic ablation strategies.
        self.global_call_count = 0
        
        # Depth-specific counters.
        self.depth_call_counts = {}
        
        # ASR components.
        if MULTI_SERIALIZATION_AVAILABLE and self.selection_strategy == 'adaptive':
            # Feature dimensions: octree structure (3), geometry (5),
            # data statistics (5), and context (2).
            feature_dim = 15
            
            self.adaptive_selector = AdaptiveSerializationRouter(
                feature_dim, len(self.serialization_methods)
            )
            
            self.performance_evaluator = SerializationPerformanceEvaluator(
                dim, len(self.serialization_methods)
            )
            
            # Performance history for the auxiliary ASR objective.
            self.performance_history = []
            self.update_frequency = 20
            self.call_count = 0
            
            if config['debug']:
                print(f"AdaptiveSerializationRouter initialized with {len(self.serialization_methods)} methods")
    
    def extract_comprehensive_features(self, data, octree, depth):
        """Extract ASR features from octree structure and point features."""
        features = []
        
        try:
            # 1. Basic octree statistics.
            features.extend([
                depth,
                octree.nnum[depth] if depth < len(octree.nnum) else 0,
                octree.nnum_nempty[depth] if (octree.nempty and depth < len(octree.nnum_nempty)) else 0,
            ])
            
            # 2. Spatial geometry statistics.
            key = octree.key(depth, octree.nempty)
            if key.numel() > 0:
                from ocnn.octree.shuffled_key import key2xyz
                x, y, z, b = key2xyz(key, depth)
                xyz = torch.stack([x.float(), y.float(), z.float()], dim=1)
                
                # Spatial distribution statistics.
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
                
            # 3. Feature statistics.
            features.extend([
                data.mean().item(),
                data.std().item(),
                (data.max() - data.min()).item(),
                (data > data.mean()).float().mean().item(),
                float(data.shape[0]),
            ])
            
            # 4. Context statistics.
            features.extend([
                float(depth / 10.0),
                float(data.shape[1] / 512.0),
            ])
            
        except Exception as e:
            # Fall back to neutral features if extraction fails.
            features = [0.0] * 15
            
        return torch.tensor(features, device=data.device, dtype=torch.float32)

    def select_serialization_method(self, data, octree, depth):
        """Select a serialization method with ASR or an ablation policy."""
        if not MULTI_SERIALIZATION_AVAILABLE:
            return 'z_order'
        
        # Ablation policies.
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
        
        # Adaptive policy.
        elif self.selection_strategy == 'adaptive' and hasattr(self, 'adaptive_selector'):
            # Extract octree-aware structural features.
            features = self.extract_comprehensive_features(data, octree, depth)
            
            # Select using the adaptive router.
            method_idx, method_logits, performance_pred = self.adaptive_selector(
                features, training=self.training
            )
            
            selected_method = self.serialization_methods[method_idx]
            
            # Record locality feedback for the auxiliary ASR objective.
            if self.training:
                self.record_performance(selected_method, data, octree, depth, performance_pred, method_logits)
            
            return selected_method
        
        # Default to the original Z-order.
        return 'z_order'
    
    def record_performance(self, method, data, octree, depth, performance_pred, method_logits):
        """Record locality feedback for ASR self-supervision."""
        self.call_count += 1
        
        if self.call_count % self.update_frequency == 0:
            # Estimate observed locality preservation.
            actual_performance = self.performance_evaluator.evaluate_locality_preservation(
                data, octree, depth, method
            )
            
            # Store recent feedback.
            self.performance_history.append({
                'method': method,
                'depth': depth,
                'predicted': performance_pred.detach().cpu(),
                'actual': actual_performance,
                'logits': method_logits.detach().cpu(),
            })
            
            # Keep a bounded history.
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
    
    def compute_adaptive_loss(self):
        """Compute the auxiliary ASR loss."""
        if len(self.performance_history) < 10:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        losses = []
        for record in self.performance_history[-50:]:
            try:
                method_idx = self.serialization_methods.index(record['method'])
                predicted = record['predicted'][method_idx]
                actual = torch.tensor(record['actual'], device=predicted.device)
                
                # MSE loss for performance prediction.
                perf_loss = F.mse_loss(predicted, actual)
                
                # Adjust method probabilities using locality feedback.
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
        # Select serialization with ASR or an ablation policy.
        selected_method = self.select_serialization_method(data, octree, depth)
        
        config = MULTI_SERIALIZATION_CONFIG
        if config.get('debug', False):
            print(f"[Multi-Serialization] Depth {depth}: Using {selected_method} (Strategy: {self.selection_strategy})")
        
        if selected_method == 'z_order':
            # Z-order uses the original octree order.
            result = super().forward(data, octree, depth)
        
        elif selected_method in ['trans_z', 'hilbert', 'trans_hilbert'] and MULTI_SERIALIZATION_AVAILABLE:
            # Alternative serialization methods require a reorder.
            try:
                key = octree.key(depth, octree.nempty)
                if key.numel() > 0 and key.numel() == data.shape[0]:
                    from ocnn.octree.shuffled_key import key2xyz
                    x, y, z, b = key2xyz(key, depth)
                    
                    # Re-encode with the selected serialization method.
                    new_key = multi_xyz2key(x, y, z, b, depth, selected_method)
                    
                    # Reorder the point feature sequence.
                    _, sort_indices = torch.sort(new_key)
                    _, original_indices = torch.sort(key)
                    
                    reorder_map = torch.empty_like(sort_indices)
                    reorder_map[original_indices] = sort_indices
                    
                    reordered_data = data[reorder_map]
                    result = super().forward(reordered_data, octree, depth)
                    
                    # Restore the original octree order.
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
            # Default path.
            result = super().forward(data, octree, depth)
        
        # Update the ASR auxiliary objective in adaptive mode.
        if (self.training and self.selection_strategy == 'adaptive' and 
            hasattr(self, 'adaptive_selector') and len(self.performance_history) > 10):
            
            # Backpropagate only the ASR auxiliary loss.
            adaptive_loss = self.compute_adaptive_loss()
            if adaptive_loss.requires_grad and adaptive_loss.item() > 0:
                # Retain the graph so the main objective can still backpropagate.
                try:
                    adaptive_loss.backward(retain_graph=True)
                except Exception as e:
                    if config.get('debug', False):
                        print(f"Adaptive loss backward failed: {e}")
        
        return result


class OctreePointTTT(nn.Module):
    """Octree-guided PointTTT operator with ASR and Gating Bi-TTTLayer."""
    
    def __init__(self, dim: int, proj_drop: float = 0.0, 
                 ttt_patch_size: int = 64, ttt_num_heads: int = 24):
        super().__init__()
        self.dim = dim
        
        self.octree_ttt = AdaptiveSerializationGatingBiTTTLayer(
            dim=dim,
            patch_size=ttt_patch_size,
            num_heads=ttt_num_heads,
            proj_drop=proj_drop
        )
        
        self.octree_norm = OctreeAdaptiveNorm(dim=dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree, depth: int):
        data_ttt = self.octree_ttt(data, octree, depth)
        data_seq = data_ttt.unsqueeze(0).permute(0, 2, 1)
        data_seq = self.octree_norm(data_seq, depth)
        data = data_seq.permute(0, 2, 1).squeeze(0)

        data = self.proj(data)
        data = self.proj_drop(data)
        return data
    

class FeatureEnhancementProjector(nn.Module):
    """
    Feature Enhancement Projector (FEP).

    FEP replaces a plain linear output projection with local depthwise
    convolution and gated channel interactions while preserving [B, T, C].
    """
    def __init__(self, ninp, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.fc2 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.g   = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.dwconv1 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=ninp, bias=True, padding_mode='zeros')
        self.dwconv2 = nn.Conv1d(in_channels=ninp, out_channels=ninp, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=ninp, bias=True, padding_mode='zeros')

        self.drop = nn.Dropout(drop)
        self.act = nn.SiLU()

    def forward(self, x):
        """Project features while preserving [B, T, C]."""
        residual = x
        x = x.permute(0, 2, 1)

        x = self.dwconv1(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))

        x = x.permute(0, 2, 1)
        return residual + x
