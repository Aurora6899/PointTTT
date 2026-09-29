from typing import List

import dwconv
import ocnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ocnn.octree import Octree
from torch.utils.checkpoint import checkpoint

from .PointTTThierarchical import HierarchicalPointTTTLayer
from .multi_serialization import AXIS12_METHODS, multi_xyz2key
from .ttt import TTTConfig, TTTLinear, TTTMLP


class OctreeT(Octree):
    def __init__(self, octree: Octree, nempty: bool = True):
        super().__init__(octree.depth, octree.full_depth)
        self.__dict__.update(octree.__dict__)
        self.nempty = nempty


class OctreeDWConvBn(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: List[int] = [3],
        stride: int = 1,
        nempty: bool = False,
    ):
        super().__init__()
        self.conv = dwconv.OctreeDWConv(
            in_channels, kernel_size, nempty, use_bias=False
        )
        self.bn = torch.nn.BatchNorm1d(in_channels)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out


class PointTTTBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        nempty: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)

        self.pointttt = OctreePointTTTLayer(
            dim=dim,
            proj_drop=proj_drop,
            nempty=nempty,
            partition_by_batch=kwargs.get("partition_by_batch", False),
            ttt_base_lr=kwargs.get("ttt_base_lr", 1.0),
            ttt_update_train=kwargs.get("ttt_update_train", True),
            ttt_update_test=kwargs.get("ttt_update_test", True),
            ttt_patch_size=kwargs.get("ttt_patch_size", 64),
            ttt_num_heads=kwargs.get("ttt_num_heads", 24),
            ttt_layer_type=kwargs.get("ttt_layer_type", "linear"),
            ttt_share_directions=kwargs.get("ttt_share_directions", False),
            ttt_direction_mode=kwargs.get("ttt_direction_mode", "bidirectional"),
            ttt_fusion_mode=kwargs.get("ttt_fusion_mode", "gated"),
            pointttt_hierarchical_enabled=kwargs.get(
                "pointttt_hierarchical_active", False
            ),
            pointttt_global_chunk_size=kwargs.get("pointttt_global_chunk_size", 128),
            pointttt_summary_tokens=kwargs.get("pointttt_summary_tokens", 1),
            pointttt_global_bidirectional=kwargs.get(
                "pointttt_global_bidirectional", True
            ),
            pointttt_global_gate_init=kwargs.get("pointttt_global_gate_init", 0.0),
            serialization_enabled=kwargs.get("serialization_enabled", False),
            serialization_target_temperature=kwargs.get(
                "serialization_target_temperature", 0.25
            ),
            serialization_explore_temperature=kwargs.get(
                "serialization_explore_temperature", 1.0
            ),
            serialization_performance_weight=kwargs.get(
                "serialization_performance_weight", 1.0
            ),
        )

        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = OctreeDWConvBn(dim, nempty=nempty)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        data = self.cpe(data, octree, depth) + data
        attn = self.pointttt(self.norm1(data), octree, depth)
        data = data + self.drop_path(attn, octree, depth)
        return data


class PointTTTStage(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        nempty: bool = True,
        use_checkpoint: bool = True,
        num_blocks: int = 2,
        pim_block=PointTTTBlock,
        **kwargs,
    ):
        super().__init__()
        self.num_blocks = num_blocks

        # Avoid resampling routes and recomputing the ASR auxiliary objective
        # through activation checkpointing.
        serialization_enabled = bool(kwargs.get("serialization_enabled", False))
        self.use_checkpoint = False if serialization_enabled else use_checkpoint

        stage_idx = int(kwargs.get("pointttt_stage_idx", -1))
        hierarchical_enabled = bool(kwargs.get("pointttt_hierarchical_enabled", False))
        hierarchical_stages = tuple(
            int(stage) for stage in kwargs.get("pointttt_hierarchical_stages", [])
        )
        hierarchical_interval = int(
            kwargs.get("pointttt_hierarchical_block_interval", 0)
        )
        blocks = []
        for i in range(num_blocks):
            if hierarchical_interval > 0:
                selected_block = (
                    i + 1
                ) % hierarchical_interval == 0 or i == num_blocks - 1
            else:
                selected_block = i == num_blocks - 1
            block_kwargs = dict(kwargs)
            block_kwargs["pointttt_hierarchical_active"] = (
                hierarchical_enabled
                and stage_idx in hierarchical_stages
                and selected_block
            )
            blocks.append(
                pim_block(
                    dim=dim,
                    proj_drop=proj_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    nempty=nempty,
                    **block_kwargs,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, data: torch.Tensor, octree: OctreeT, depth: int):
        for i in range(self.num_blocks):
            if self.use_checkpoint and self.training:
                data = checkpoint(
                    self.blocks[i], data, octree, depth, use_reentrant=False
                )
            else:
                data = self.blocks[i](data, octree, depth)
        return data


class PatchEmbed(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 96,
        num_down: int = 2,
        nempty: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_stages = num_down
        self.delta_depth = -num_down
        channels = [int(dim * 2**i) for i in range(-self.num_stages, 1)]

        self.convs = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    in_channels if i == 0 else channels[i],
                    channels[i],
                    kernel_size=[3],
                    stride=1,
                    nempty=nempty,
                )
                for i in range(self.num_stages)
            ]
        )
        self.downsamples = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeConvBnRelu(
                    channels[i],
                    channels[i + 1],
                    kernel_size=[2],
                    stride=2,
                    nempty=nempty,
                )
                for i in range(self.num_stages)
            ]
        )
        self.proj = ocnn.modules.OctreeConvBnRelu(
            channels[-1], dim, kernel_size=[3], stride=1, nempty=nempty
        )

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.convs[i](data, octree, depth_i)
            data = self.downsamples[i](data, octree, depth_i)
        data = self.proj(data, octree, depth_i - 1)
        return data


class Downsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int] = [2],
        nempty: bool = True,
    ):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(out_channels)
        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=2,
            nempty=nempty,
            use_bias=True,
        )

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.conv(data, octree, depth)
        data = self.norm(data)
        return data


class PointTTTBackbone(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int] = [96, 192, 384, 384],
        num_blocks: List[int] = [2, 2, 18, 2],
        drop_path: float = 0.5,
        nempty: bool = True,
        stem_down: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.nempty = nempty
        self.num_stages = len(num_blocks)
        self.stem_down = stem_down
        drop_ratio = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.patch_embed = PatchEmbed(in_channels, channels[0], stem_down, nempty)
        layers = []
        for i in range(self.num_stages):
            stage_kwargs = dict(kwargs)
            stage_kwargs["pointttt_stage_idx"] = i
            layers.append(
                PointTTTStage(
                    dim=channels[i],
                    drop_path=drop_ratio[
                        sum(num_blocks[:i]) : sum(num_blocks[: i + 1])
                    ],
                    nempty=nempty,
                    num_blocks=num_blocks[i],
                    **stage_kwargs,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        self.downsamples = torch.nn.ModuleList(
            [
                Downsample(channels[i], channels[i + 1], kernel_size=[2], nempty=nempty)
                for i in range(self.num_stages - 1)
            ]
        )

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        data = self.patch_embed(data, octree, depth)
        depth = depth - self.stem_down
        octree = OctreeT(octree, nempty=self.nempty)
        features = {}
        for i in range(self.num_stages):
            depth_i = depth - i
            data = self.layers[i](data, octree, depth_i)
            features[depth_i] = data
            if i < self.num_stages - 1:
                data = self.downsamples[i](data, octree, depth_i)
        return features


class BiTTTLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        patch_size: int,
        num_heads: int,
        proj_drop: float = 0.0,
        partition_by_batch: bool = False,
        ttt_base_lr: float = 1.0,
        ttt_update_train: bool = True,
        ttt_update_test: bool = True,
        ttt_layer_type: str = "linear",
        ttt_share_directions: bool = False,
        ttt_direction_mode: str = "bidirectional",
        ttt_fusion_mode: str = "gated",
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.ttt_layer_type = str(ttt_layer_type).lower()
        self.ttt_share_directions = bool(ttt_share_directions)
        self.ttt_direction_mode = str(ttt_direction_mode).lower()
        self.ttt_fusion_mode = str(ttt_fusion_mode).lower()
        if self.ttt_direction_mode not in ("forward", "bidirectional"):
            raise ValueError("ttt_direction_mode must be forward or bidirectional.")
        allowed_fusions = ("none", "mean", "sum", "gated")
        if self.ttt_fusion_mode not in allowed_fusions:
            raise ValueError(
                "ttt_fusion_mode must be one of %s." % (", ".join(allowed_fusions))
            )
        if (
            self.ttt_direction_mode == "bidirectional"
            and self.ttt_fusion_mode == "none"
        ):
            raise ValueError("Bidirectional TTT requires mean, sum, or gated fusion.")
        # Keep the historical flat sequence as the default so existing
        # classification/segmentation experiments are bit-for-bit unchanged.
        # Detection enables this option to prevent TTT chunks from crossing
        # point-cloud boundaries in a multi-sample batch.
        self.partition_by_batch = partition_by_batch

        ttt_layer_type = self.ttt_layer_type
        ttt_layer_classes = {
            "linear": TTTLinear,
            "mlp": TTTMLP,
        }
        if ttt_layer_type not in ttt_layer_classes:
            raise ValueError(
                f"Unsupported ttt_layer_type {ttt_layer_type!r}; "
                f"choose from {tuple(ttt_layer_classes)}."
            )

        self.config = TTTConfig(
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
        self.ttt_forward = ttt_layer_class(self.config, layer_idx=0)
        if self.ttt_direction_mode == "forward":
            self.ttt_backward = None
        elif self.ttt_share_directions:
            self.ttt_backward = None
        else:
            self.ttt_backward = ttt_layer_class(self.config, layer_idx=1)

        if (
            self.ttt_direction_mode == "bidirectional"
            and self.ttt_fusion_mode == "gated"
        ):
            self.gate_forward = nn.Parameter(torch.tensor(0.1))
            self.gate_backward = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("gate_forward", None)
            self.register_parameter("gate_backward", None)

        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def _build_position_ids(self, batch_size: int, seq_len: int, device: torch.device):
        return (
            torch.arange(0, seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )

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

        if self.ttt_direction_mode == "forward":
            fused = out_forward
        else:
            # --------------------------

            # --------------------------

            x_rev = torch.flip(x, dims=[1])
            pos_backward = self._build_position_ids(B_seq, seq_len, x.device)

            backward_ttt = (
                self.ttt_forward if self.ttt_share_directions else self.ttt_backward
            )
            out_backward_rev = backward_ttt(
                hidden_states=x_rev,
                attention_mask=None,
                position_ids=pos_backward,
                cache_params=None,
            )

            out_backward = torch.flip(out_backward_rev, dims=[1])

            # --------------------------

            # --------------------------
            if self.ttt_fusion_mode == "gated":
                gate_f = torch.tanh(self.gate_forward)
                gate_b = torch.tanh(self.gate_backward)
                fused = gate_f * out_forward + gate_b * out_backward
            elif self.ttt_fusion_mode == "mean":
                fused = 0.5 * (out_forward + out_backward)
            elif self.ttt_fusion_mode == "sum":
                fused = out_forward + out_backward
            else:  # Guarded in __init__, retained for defensive clarity.
                raise RuntimeError(
                    "Invalid bidirectional fusion mode: %s" % self.ttt_fusion_mode
                )

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
        # Classification can keep empty octree nodes. Fall back to the full
        # node layout when it is the one aligned with the feature tensor.
        if batch_id.numel() != data.shape[0]:
            batch_id = octree.batch_id(depth, nempty=False).long()
        if batch_id.numel() != data.shape[0]:
            raise RuntimeError(
                f"Octree/data size mismatch at depth {depth}: "
                f"{batch_id.numel()} batch ids for {data.shape[0]} features"
            )

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
                full_indices = indices[: num_full * K].view(num_full, K)
                index_groups.setdefault(K, []).append(full_indices)
                chunk_groups.setdefault(K, []).append(
                    data.index_select(0, full_indices.reshape(-1)).view(num_full, K, -1)
                )
            tail_len = count - num_full * K
            if tail_len:
                tail_indices = indices[num_full * K :].view(1, tail_len)
                index_groups.setdefault(tail_len, []).append(tail_indices)
                chunk_groups.setdefault(tail_len, []).append(
                    data.index_select(0, tail_indices.reshape(-1)).view(1, tail_len, -1)
                )

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
        return (
            f"PointTTT dim={self.dim}, patch_size={self.patch_size}, "
            f"num_heads={self.num_heads}, "
            f"ttt_layer_type={self.ttt_layer_type}, "
            f"ttt_direction_mode={self.ttt_direction_mode}, "
            f"ttt_fusion_mode={self.ttt_fusion_mode}, "
            f"ttt_share_directions={self.ttt_share_directions}, "
            f"gate_forward={None if self.gate_forward is None else self.gate_forward.item():}, "
            f"gate_backward={None if self.gate_backward is None else self.gate_backward.item():}"
        )


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

        pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        unpool = nn.Upsample(scale_factor=patch_size, mode="nearest")

        x_div_win = pool(pixel_output)
        x_flat_win = x_div_win.transpose(1, 2)
        x_norm_win = self.window_norm(x_flat_win)
        x_out_win = x_norm_win.transpose(1, 2)
        x_out_win = unpool(x_out_win)

        window_output = x_out_win + pixel_output

        if L_padded != L:
            window_output = window_output[:, :, :L]
        return window_output


# Per-sample locality evaluator used by the sole paper-aligned ASR path.
class SerializationPerformanceEvaluator(nn.Module):
    @torch.no_grad()
    def evaluate_all_locality_costs_per_sample(self, data, octree, depth, methods):
        """Returns normalized-adjacency costs with shape ``[B, M]``."""
        nempty = bool(getattr(octree, "nempty", False))
        key = octree.key(depth, nempty)
        if key.numel() != data.shape[0]:
            nempty = not nempty
            key = octree.key(depth, nempty)
        if key.numel() == 0 or key.numel() != data.shape[0]:
            return data.new_zeros((0, len(methods)))

        from ocnn.octree.shuffled_key import key2xyz

        x, y, z, batch_id = key2xyz(key, depth)
        batch_id = batch_id.long()
        batch_size = int(getattr(octree, "batch_size", 0))
        if batch_size <= 0:
            batch_size = int(batch_id.max().item()) + 1
        method_count = len(methods)
        scale = float(max((1 << int(depth)) - 1, 1))
        xyz = torch.stack([x.float(), y.float(), z.float()], dim=1) / scale

        candidate_keys = []
        for method in methods:
            if method == "z_order":
                candidate_keys.append(key)
            else:
                candidate_keys.append(multi_xyz2key(x, y, z, batch_id, depth, method))
        orders = torch.argsort(torch.stack(candidate_keys, dim=0), dim=1)
        expanded_batch = batch_id.unsqueeze(0).expand(method_count, -1)
        ordered_batch = torch.gather(expanded_batch, 1, orders)
        ordered_xyz = xyz[orders]
        same_sample = ordered_batch[:, 1:] == ordered_batch[:, :-1]
        distances = torch.linalg.vector_norm(
            ordered_xyz[:, 1:] - ordered_xyz[:, :-1], dim=2
        )

        method_offsets = (
            torch.arange(method_count, device=batch_id.device, dtype=batch_id.dtype)[
                :, None
            ]
            * batch_size
        )
        reduction_index = method_offsets + ordered_batch[:, :-1]
        valid = same_sample.to(distances.dtype)
        distance_sum = xyz.new_zeros(method_count * batch_size)
        pair_count = xyz.new_zeros(method_count * batch_size)
        distance_sum.scatter_add_(
            0, reduction_index.reshape(-1), (distances * valid).reshape(-1)
        )
        pair_count.scatter_add_(0, reduction_index.reshape(-1), valid.reshape(-1))
        costs = distance_sum / pair_count.clamp_min(1.0)
        return costs.view(method_count, batch_size).transpose(0, 1)


class AdaptiveSerializationSelector(nn.Module):
    """Paper ASR network: 15 -> 128 -> 64 with two M-way heads."""

    def __init__(self, feature_dim, num_methods):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.method_selector = nn.Linear(64, num_methods)
        self.performance_predictor = nn.Linear(64, num_methods)

    def forward(self, features, training=True, sample_temperature=1.0):
        hidden = self.feature_extractor(features)
        method_logits = self.method_selector(hidden)
        if training:
            temperature = max(float(sample_temperature), 0.05)
            probabilities = F.softmax(method_logits / temperature, dim=-1)
            method_indices = torch.multinomial(probabilities, 1).squeeze(-1)
        else:
            method_indices = method_logits.argmax(dim=-1)
        performance_prediction = self.performance_predictor(hidden)
        return method_indices, method_logits, performance_prediction


class AdaptiveSerializationBiTTTLayer(BiTTTLayer):
    """Per-sample Axis12 ASR; disabled mode preserves original Z-order."""

    def __init__(
        self,
        dim,
        patch_size=64,
        num_heads=24,
        proj_drop=0.0,
        partition_by_batch=False,
        ttt_base_lr=1.0,
        ttt_update_train=True,
        ttt_update_test=True,
        ttt_layer_type="linear",
        ttt_share_directions=False,
        ttt_direction_mode="bidirectional",
        ttt_fusion_mode="gated",
        serialization_enabled=False,
        serialization_target_temperature=0.25,
        serialization_explore_temperature=1.0,
        serialization_performance_weight=1.0,
    ):
        super().__init__(
            dim,
            patch_size,
            num_heads,
            proj_drop,
            partition_by_batch=partition_by_batch,
            ttt_base_lr=ttt_base_lr,
            ttt_update_train=ttt_update_train,
            ttt_update_test=ttt_update_test,
            ttt_layer_type=ttt_layer_type,
            ttt_share_directions=ttt_share_directions,
            ttt_direction_mode=ttt_direction_mode,
            ttt_fusion_mode=ttt_fusion_mode,
        )
        self.serialization_enabled = bool(serialization_enabled)
        self.target_temperature = max(float(serialization_target_temperature), 1.0e-3)
        self.explore_temperature = max(float(serialization_explore_temperature), 0.05)
        self.performance_weight = float(serialization_performance_weight)
        self.last_serialization_aux_loss = None
        self.last_selected_method_indices = None

        if self.serialization_enabled:
            self.serialization_methods = AXIS12_METHODS
            self.adaptive_selector = AdaptiveSerializationSelector(
                15, len(self.serialization_methods)
            )
            self.performance_evaluator = SerializationPerformanceEvaluator()
        else:
            # The disabled path calls BiTTTLayer directly in the octree's
            # original Morton/Z-order and owns no ASR parameters.
            self.serialization_methods = ("z_order",)

    @torch.no_grad()
    def extract_features(self, data, octree, depth):
        """Builds one bounded 15-D routing descriptor per point cloud."""
        nempty = bool(getattr(octree, "nempty", False))
        key = octree.key(depth, nempty)
        if key.numel() != data.shape[0]:
            nempty = not nempty
            key = octree.key(depth, nempty)
        if key.numel() == 0 or key.numel() != data.shape[0]:
            return data.new_zeros((0, 15))

        from ocnn.octree.shuffled_key import key2xyz

        x, y, z, batch_id = key2xyz(key, depth)
        batch_id = batch_id.long()
        batch_size = int(getattr(octree, "batch_size", 0))
        if batch_size <= 0:
            batch_size = int(batch_id.max().item()) + 1
        coordinate_scale = float(max((1 << int(depth)) - 1, 1))
        xyz = torch.stack([x.float(), y.float(), z.float()], dim=1) / coordinate_scale
        detached_data = data.detach().float()

        node_count = torch.bincount(batch_id, minlength=batch_size).to(xyz.dtype)
        if nempty:
            occupied = torch.ones_like(batch_id, dtype=torch.bool)
        else:
            occupied = octree.nempty_mask(depth).to(batch_id.device)
            if occupied.numel() != batch_id.numel():
                occupied = torch.ones_like(batch_id, dtype=torch.bool)
        nonempty_count = torch.bincount(batch_id[occupied], minlength=batch_size).to(
            xyz.dtype
        )

        xyz_index = batch_id[:, None].expand(-1, 3)
        xyz_sum = xyz.new_zeros((batch_size, 3))
        xyz_square_sum = xyz.new_zeros((batch_size, 3))
        xyz_sum.scatter_add_(0, xyz_index, xyz)
        xyz_square_sum.scatter_add_(0, xyz_index, xyz.square())
        xyz_denominator = node_count.clamp_min(1.0)[:, None]
        xyz_mean = xyz_sum / xyz_denominator
        xyz_variance = (xyz_square_sum / xyz_denominator - xyz_mean.square()).clamp_min(
            0
        )
        xyz_std = torch.sqrt(xyz_variance)
        xyz_min = xyz.new_full((batch_size, 3), float("inf"))
        xyz_max = xyz.new_full((batch_size, 3), float("-inf"))
        xyz_min.scatter_reduce_(0, xyz_index, xyz, reduce="amin", include_self=True)
        xyz_max.scatter_reduce_(0, xyz_index, xyz, reduce="amax", include_self=True)
        xyz_span = xyz_max - xyz_min
        xyz_span = torch.where(
            torch.isfinite(xyz_span), xyz_span, torch.zeros_like(xyz_span)
        )

        channel_count = float(detached_data.shape[1])
        feature_denominator = (node_count * channel_count).clamp_min(1.0)
        row_sum = detached_data.sum(dim=1)
        row_square_sum = detached_data.square().sum(dim=1)
        feature_sum = xyz.new_zeros(batch_size)
        feature_square_sum = xyz.new_zeros(batch_size)
        feature_sum.scatter_add_(0, batch_id, row_sum)
        feature_square_sum.scatter_add_(0, batch_id, row_square_sum)
        feature_mean = feature_sum / feature_denominator
        feature_variance = (
            feature_square_sum / feature_denominator - feature_mean.square()
        ).clamp_min(0)
        feature_std = torch.sqrt(feature_variance)

        row_min = detached_data.min(dim=1).values
        row_max = detached_data.max(dim=1).values
        feature_min = xyz.new_full((batch_size,), float("inf"))
        feature_max = xyz.new_full((batch_size,), float("-inf"))
        feature_min.scatter_reduce_(
            0, batch_id, row_min, reduce="amin", include_self=True
        )
        feature_max.scatter_reduce_(
            0, batch_id, row_max, reduce="amax", include_self=True
        )
        feature_range = feature_max - feature_min
        feature_range = torch.where(
            torch.isfinite(feature_range),
            feature_range,
            torch.zeros_like(feature_range),
        )
        above_per_node = (
            (detached_data > feature_mean.index_select(0, batch_id)[:, None])
            .to(xyz.dtype)
            .sum(dim=1)
        )
        above_mean = xyz.new_zeros(batch_size)
        above_mean.scatter_add_(0, batch_id, above_per_node)
        above_mean = above_mean / feature_denominator

        depth_column = node_count.new_full((batch_size,), float(depth) / 16.0)
        log_count = torch.log1p(node_count) / 16.0
        occupancy = nonempty_count / node_count.clamp_min(1.0)
        feature_width = node_count.new_full((batch_size,), float(data.shape[1]) / 512.0)
        return torch.cat(
            [
                torch.stack([depth_column, log_count, occupancy], dim=1),
                xyz_std,
                xyz_span,
                torch.stack(
                    [feature_mean, feature_std, feature_range, above_mean], dim=1
                ),
                torch.stack([log_count, feature_width], dim=1),
            ],
            dim=1,
        )

    def select_adaptive(self, data, octree, depth):
        features = self.extract_features(data, octree, depth)
        if features.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long, device=data.device)
        method_indices, method_logits, performance_prediction = self.adaptive_selector(
            features,
            training=self.training,
            sample_temperature=self.explore_temperature,
        )

        if self.training:
            costs = self.performance_evaluator.evaluate_all_locality_costs_per_sample(
                data, octree, depth, self.serialization_methods
            )
            if costs.shape[:1] != method_logits.shape[:1]:
                raise RuntimeError(
                    "ASR feature/cost batch mismatch: %s vs %s."
                    % (tuple(method_logits.shape), tuple(costs.shape))
                )
            cost_min = costs.min(dim=1, keepdim=True).values
            cost_range = costs.max(dim=1, keepdim=True).values - cost_min
            normalized_cost = (costs - cost_min) / cost_range.clamp_min(1.0e-6)
            target_probabilities = F.softmax(
                -normalized_cost / self.target_temperature, dim=1
            )
            routing_loss = (
                -(target_probabilities * F.log_softmax(method_logits, dim=1))
                .sum(dim=1)
                .mean()
            )
            target_quality = 1.0 - normalized_cost
            performance_loss = F.mse_loss(
                torch.sigmoid(performance_prediction), target_quality
            )
            self.last_serialization_aux_loss = (
                routing_loss + self.performance_weight * performance_loss
            )
        else:
            self.last_serialization_aux_loss = None

        self.last_selected_method_indices = method_indices.detach()
        return method_indices

    def _forward_with_sample_methods(self, data, octree, depth, method_indices):
        nempty = bool(getattr(octree, "nempty", False))
        key = octree.key(depth, nempty)
        if key.numel() != data.shape[0]:
            key = octree.key(depth, not nempty)
        if key.numel() == 0 or key.numel() != data.shape[0]:
            return super().forward(data, octree, depth)

        from ocnn.octree.shuffled_key import key2xyz

        x, y, z, batch_id = key2xyz(key, depth)
        batch_id = batch_id.long()
        if method_indices.numel() <= int(batch_id.max().item()):
            raise RuntimeError("ASR did not produce a route for every sample.")
        point_methods = method_indices.index_select(0, batch_id)
        mixed_key = torch.empty_like(key)
        for method_index, method in enumerate(self.serialization_methods):
            mask = point_methods == method_index
            if not mask.any():
                continue
            if method == "z_order":
                mixed_key[mask] = key[mask]
            else:
                mixed_key[mask] = multi_xyz2key(
                    x[mask], y[mask], z[mask], batch_id[mask], depth, method
                )

        sort_indices = torch.argsort(mixed_key)
        original_indices = torch.argsort(key)
        reorder_map = torch.empty_like(sort_indices)
        reorder_map[original_indices] = sort_indices
        reordered_data = data[reorder_map]
        result = super().forward(reordered_data, octree, depth)
        inverse_map = torch.empty_like(reorder_map)
        inverse_map[reorder_map] = torch.arange(
            len(reorder_map), device=reorder_map.device
        )
        return result[inverse_map]

    def forward(self, data: torch.Tensor, octree, depth: int):
        self.last_serialization_aux_loss = None
        self.last_selected_method_indices = None
        if not self.serialization_enabled:
            return super().forward(data, octree, depth)
        method_indices = self.select_adaptive(data, octree, depth)
        return self._forward_with_sample_methods(data, octree, depth, method_indices)

    def get_serialization_aux_loss(self):
        return self.last_serialization_aux_loss


class OctreePointTTTLayer(nn.Module):
    """Octree PointTTT with optional paper-aligned adaptive serialization."""

    def __init__(
        self,
        dim: int,
        proj_drop: float = 0.0,
        ttt_patch_size: int = 64,
        ttt_num_heads: int = 24,
        nempty: bool = True,
        partition_by_batch: bool = False,
        ttt_base_lr: float = 1.0,
        ttt_update_train: bool = True,
        ttt_update_test: bool = True,
        ttt_layer_type: str = "linear",
        ttt_share_directions: bool = False,
        ttt_direction_mode: str = "bidirectional",
        ttt_fusion_mode: str = "gated",
        pointttt_hierarchical_enabled: bool = False,
        pointttt_global_chunk_size: int = 128,
        pointttt_summary_tokens: int = 1,
        pointttt_global_bidirectional: bool = True,
        pointttt_global_gate_init: float = 0.0,
        serialization_enabled: bool = False,
        serialization_target_temperature: float = 0.25,
        serialization_explore_temperature: float = 1.0,
        serialization_performance_weight: float = 1.0,
    ):
        super().__init__()
        self.dim = dim

        self.octree_ttt = AdaptiveSerializationBiTTTLayer(
            dim=dim,
            patch_size=ttt_patch_size,
            num_heads=ttt_num_heads,
            proj_drop=proj_drop,
            partition_by_batch=partition_by_batch,
            ttt_base_lr=ttt_base_lr,
            ttt_update_train=ttt_update_train,
            ttt_update_test=ttt_update_test,
            ttt_layer_type=ttt_layer_type,
            ttt_share_directions=ttt_share_directions,
            ttt_direction_mode=ttt_direction_mode,
            ttt_fusion_mode=ttt_fusion_mode,
            serialization_enabled=serialization_enabled,
            serialization_target_temperature=serialization_target_temperature,
            serialization_explore_temperature=serialization_explore_temperature,
            serialization_performance_weight=serialization_performance_weight,
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

        self.pointttt_norm = OctreeAdaptiveNorm(dim=dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, data: torch.Tensor, octree, depth: int):
        data_ttt = self.octree_ttt(data, octree, depth)
        if self.hierarchical_pointttt is not None:
            data_ttt = self.hierarchical_pointttt(data_ttt, octree, depth)

        pointttt_features = data_ttt.unsqueeze(0).permute(0, 2, 1)
        pointttt_features = self.pointttt_norm(pointttt_features, depth)
        data = pointttt_features.permute(0, 2, 1).squeeze(0)

        data = self.proj(data)
        data = self.proj_drop(data)

        return data
