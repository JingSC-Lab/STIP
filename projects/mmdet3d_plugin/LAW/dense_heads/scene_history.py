"""Clip-local scene-token history enhancement for LAW."""

from typing import List, Tuple

import torch
import torch.nn as nn


class _TokenSelectionMLP(nn.Module):
    """Predict spatial attention maps used by TokenLearner."""

    def __init__(self, channels, hidden_channels, num_tokens, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_tokens),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        return self.layers(features)


class TokenLearner(nn.Module):
    """Compress one camera feature map into a few learned scene tokens."""

    def __init__(
        self,
        channels=256,
        num_tokens=2,
        bottleneck_channels=64,
        dropout=0.0,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError('num_tokens must be positive')
        self.num_tokens = int(num_tokens)
        self.normalization = nn.GroupNorm(1, channels, eps=1e-6)
        self.selection = _TokenSelectionMLP(
            channels,
            bottleneck_channels,
            self.num_tokens,
            dropout=dropout,
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError(
                'features must have shape [B, C, H, W], got '
                f'{tuple(features.shape)}')
        batch_size, channels, height, width = features.shape
        normalized = self.normalization(features)
        normalized = normalized.flatten(2).transpose(1, 2)
        attention = self.selection(normalized).transpose(1, 2)
        attention = attention.softmax(dim=-1)
        flattened = features.flatten(2).transpose(1, 2)
        tokens = torch.einsum('bnh,bhc->bnc', attention, flattened)
        if tokens.shape != (batch_size, self.num_tokens, channels):
            raise RuntimeError('TokenLearner produced an unexpected shape')
        return tokens


class CommandPositionChannelGate(nn.Module):
    """Condition image channels on existing route and position information."""

    def __init__(self, channels=256, command_dims=3, reduction=8):
        super().__init__()
        if channels % reduction != 0:
            raise ValueError('channels must be divisible by reduction')
        hidden_channels = channels // reduction
        self.feature_projection = nn.Linear(channels, hidden_channels)
        self.command_projection = nn.Linear(command_dims, hidden_channels)
        self.position_projection = nn.Linear(channels, hidden_channels)
        self.output_projection = nn.Linear(hidden_channels, channels)

    def forward(self, features, positions, commands):
        if features.shape != positions.shape or features.ndim != 4:
            raise ValueError(
                'features and positions must share shape [B, C, H, W]')
        if commands.ndim != 2 or commands.shape[0] != features.shape[0]:
            raise ValueError('commands must have shape [B, command_dims]')
        pooled_features = features.mean(dim=(2, 3))
        pooled_positions = positions.mean(dim=(2, 3))
        hidden = (
            self.feature_projection(pooled_features)
            + self.command_projection(commands)
            + self.position_projection(pooled_positions)
        )
        channel_weights = torch.sigmoid(
            self.output_projection(torch.relu(hidden)))
        return features * channel_weights[:, :, None, None]


class SceneHistoryEnhancer(nn.Module):
    """Select, unfold, and temporally enhance clip-local scene tokens.

    Each queued tensor has shape [B, N, T, C]. Historical future step
    s + frame_offset is aligned with current step s; scene and frame metadata
    prevent state from crossing samples or clips.
    """

    def __init__(
        self,
        embed_dims=256,
        num_views=6,
        tokens_per_view=2,
        num_steps=6,
        interact_steps=None,
        num_heads=8,
        ffn_dim=1024,
        max_history=1,
        max_residual=0.2,
        dropout=0.0,
    ):
        super().__init__()
        if not isinstance(max_history, int) or max_history < 1:
            raise ValueError('max_history must be a positive integer')
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError('num_steps must be a positive integer')
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')
        if max_residual < 0:
            raise ValueError('max_residual must be non-negative')

        self.embed_dims = embed_dims
        self.num_views = num_views
        self.num_steps = num_steps
        if interact_steps is None:
            interact_steps = num_steps
        if not isinstance(interact_steps, int) or interact_steps < 1:
            raise ValueError('interact_steps must be a positive integer')
        self.interact_steps = min(interact_steps, num_steps)
        self.max_history = max_history
        self.max_residual = float(max_residual)

        self.channel_gate = CommandPositionChannelGate(channels=embed_dims)
        self.token_learner = TokenLearner(
            channels=embed_dims,
            num_tokens=tokens_per_view,
            dropout=dropout,
        )
        self.step_embedding = nn.Embedding(num_steps, embed_dims)
        # Sharing this MLP across cameras preserves camera symmetry and follows
        # the original scene-unfolding idea with fewer parameters.
        self.temporal_unfold = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

        cross_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.history_cross = nn.TransformerDecoder(
            cross_layer, num_layers=1)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_self = nn.TransformerEncoder(
            temporal_layer, num_layers=1)

        # Offsets index elapsed dataset frames, not queue positions.
        self.history_offset_embedding = nn.Embedding(64, embed_dims)
        self.cross_residual_parameter = nn.Parameter(torch.zeros(1))
        self.self_residual_parameter = nn.Parameter(torch.zeros(1))
        self._history: List[
            Tuple[torch.Tensor, Tuple[int, ...], Tuple[str, ...]]
        ] = []

    @staticmethod
    def _metadata(img_metas, batch_size):
        if not isinstance(img_metas, (list, tuple)):
            raise TypeError('img_metas must be a list or tuple')
        if len(img_metas) != batch_size:
            raise ValueError(
                'img_metas batch size does not match feature batch size: '
                f'{len(img_metas)} != {batch_size}')
        frame_indices = []
        scene_tokens = []
        for meta in img_metas:
            if 'frame_idx' not in meta:
                raise KeyError('frame_idx is required for scene history')
            if 'scene_token' not in meta:
                raise KeyError('scene_token is required for scene history')
            frame_indices.append(int(meta['frame_idx']))
            scene_tokens.append(meta['scene_token'])
        return tuple(frame_indices), tuple(scene_tokens)

    @staticmethod
    def _commands(img_metas, device, dtype):
        commands = []
        for meta in img_metas:
            if 'ego_fut_cmd' not in meta:
                raise KeyError('ego_fut_cmd is required for scene tokens')
            command = torch.as_tensor(
                meta['ego_fut_cmd'], device=device, dtype=dtype)
            if command.numel() != 3:
                raise ValueError(
                    'ego_fut_cmd must contain exactly three values, got '
                    f'shape {tuple(command.shape)}')
            commands.append(command.reshape(3))
        return torch.stack(commands, dim=0)

    def reset(self):
        """Clear runtime state at the beginning of every temporal clip."""
        self._history.clear()

    def _select_tokens(self, image_features, image_positions, img_metas):
        if image_features.shape != image_positions.shape:
            raise ValueError('image features and positions must share shape')
        if image_features.ndim != 5:
            raise ValueError(
                'image_features must have shape [B, V, C, H, W]')
        batch_size, num_views, channels, height, width = image_features.shape
        if num_views != self.num_views or channels != self.embed_dims:
            raise ValueError(
                'Unexpected scene feature dimensions: '
                f'views={num_views}, channels={channels}')

        commands = self._commands(
            img_metas, image_features.device, image_features.dtype)
        flat_features = image_features.reshape(
            batch_size * num_views, channels, height, width)
        flat_positions = image_positions.reshape(
            batch_size * num_views, channels, height, width)
        flat_commands = commands[:, None, :].expand(
            -1, num_views, -1).reshape(batch_size * num_views, 3)
        conditioned = self.channel_gate(
            flat_features, flat_positions, flat_commands)
        tokens = self.token_learner(conditioned)
        return tokens.reshape(batch_size, -1, channels)

    def _unfold_tokens(self, tokens):
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dims:
            raise ValueError('tokens must have shape [B, N, embed_dims]')
        batch_size, num_tokens, channels = tokens.shape
        steps = self.step_embedding.weight
        base = tokens[:, :, None, :].expand(
            batch_size, num_tokens, self.num_steps, channels)
        step_features = steps[None, None, :, :].expand(
            batch_size, num_tokens, -1, -1)
        delta = self.temporal_unfold(
            torch.cat([base, step_features], dim=-1))
        return base + delta

    def _encode(self, image_features, image_positions, img_metas):
        tokens = self._select_tokens(
            image_features, image_positions, img_metas)
        return self._unfold_tokens(tokens)

    def push(self, image_features, image_positions, img_metas):
        """Encode and cache a detached historical scene frame."""
        with torch.no_grad():
            tokens = self._encode(
                image_features, image_positions, img_metas)
        frame_indices, scene_tokens = self._metadata(
            img_metas, tokens.shape[0])
        if self._history and tokens.shape != self._history[-1][0].shape:
            raise ValueError('Scene token shape changed within a clip')
        self._history.append(
            (tokens.detach(), frame_indices, scene_tokens))
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def _zero_parameter_dependency(self, reference):
        zero = reference.new_zeros(())
        for parameter in self.history_cross.parameters():
            zero = zero + parameter.reshape(-1)[0] * 0.0
        for parameter in self.history_offset_embedding.parameters():
            zero = zero + parameter.reshape(-1)[0] * 0.0
        zero = zero + self.cross_residual_parameter.reshape(-1)[0] * 0.0
        return zero

    @staticmethod
    def _run_decoder(decoder, target, memory):
        if target.is_cuda:
            # Avoid the invalid fused SDP path seen with PyTorch 2.0 on H20;
            # the math backend is deterministic for strict-seed runs.
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ):
                return decoder(tgt=target, memory=memory)
        return decoder(tgt=target, memory=memory)

    @staticmethod
    def _run_encoder(encoder, tokens):
        if tokens.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ):
                return encoder(tokens)
        return encoder(tokens)

    def effective_cross_scale(self):
        return self.max_residual * torch.tanh(
            self.cross_residual_parameter)

    def effective_self_scale(self):
        return self.max_residual * torch.tanh(
            self.self_residual_parameter)

    def forward(self, image_features, image_positions, img_metas):
        current = self._encode(
            image_features, image_positions, img_metas)
        batch_size, num_tokens, num_steps, _ = current.shape
        current_frames, current_scenes = self._metadata(
            img_metas, batch_size)
        cross_scale = self.effective_cross_scale().to(dtype=current.dtype)

        enhanced_batches = []
        used_history = False
        for batch_idx in range(batch_size):
            enhanced_steps = []
            for step_idx in range(num_steps):
                memories = []
                seen_frames = set()
                if step_idx >= self.interact_steps:
                    enhanced_steps.append(
                        current[
                            batch_idx:batch_idx + 1,
                            :, step_idx:step_idx + 1, :
                        ])
                    continue
                for tokens, frame_indices, scene_tokens in self._history:
                    if tokens.shape != current.shape:
                        raise ValueError('Current and historical scene shapes differ')
                    if scene_tokens[batch_idx] != current_scenes[batch_idx]:
                        continue
                    history_frame = frame_indices[batch_idx]
                    if history_frame in seen_frames:
                        continue
                    frame_offset = (
                        current_frames[batch_idx]
                        - frame_indices[batch_idx]
                    )
                    history_step = step_idx + frame_offset
                    if frame_offset <= 0 or history_step >= num_steps:
                        continue
                    seen_frames.add(history_frame)
                    offset_index = min(
                        frame_offset,
                        self.history_offset_embedding.num_embeddings,
                    ) - 1
                    memories.append(
                        tokens[batch_idx, :, history_step, :]
                        + self.history_offset_embedding.weight[offset_index])

                step_tokens = current[
                    batch_idx:batch_idx + 1, :, step_idx, :]
                if memories:
                    memory = torch.cat(memories, dim=0).unsqueeze(0)
                    interaction = self._run_decoder(
                        self.history_cross, step_tokens, memory)
                    step_tokens = step_tokens + cross_scale * (
                        interaction - step_tokens)
                    used_history = True
                enhanced_steps.append(step_tokens.unsqueeze(2))
            enhanced_batches.append(torch.cat(enhanced_steps, dim=2))

        enhanced = torch.cat(enhanced_batches, dim=0)
        if not used_history:
            enhanced = enhanced + self._zero_parameter_dependency(enhanced)

        temporal_input = enhanced.reshape(
            batch_size * num_tokens, num_steps, self.embed_dims)
        temporal_output = self._run_encoder(
            self.temporal_self, temporal_input)
        self_scale = self.effective_self_scale().to(dtype=enhanced.dtype)
        enhanced = temporal_input + self_scale * (
            temporal_output - temporal_input)
        return enhanced.reshape(
            batch_size, num_tokens, num_steps, self.embed_dims)


class StepwiseSceneToPlan(nn.Module):
    """Fuse time-aligned scene tokens into the corresponding plan query."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        ffn_dim=1024,
        max_residual=0.2,
        dropout=0.0,
    ):
        super().__init__()
        if max_residual < 0:
            raise ValueError('max_residual must be non-negative')
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.cross = nn.TransformerDecoder(layer, num_layers=1)
        self.max_residual = float(max_residual)
        self.residual_parameter = nn.Parameter(torch.zeros(1))

    def effective_residual_scale(self):
        return self.max_residual * torch.tanh(self.residual_parameter)

    def forward(self, plan_tokens, scene_tokens):
        if plan_tokens.ndim != 3 or scene_tokens.ndim != 4:
            raise ValueError(
                'plan and scene tokens must have shapes [B, T, C] and '
                '[B, N, T, C]')
        batch_size, num_steps, channels = plan_tokens.shape
        if (
            scene_tokens.shape[0] != batch_size
            or scene_tokens.shape[2] != num_steps
            or scene_tokens.shape[3] != channels
        ):
            raise ValueError('Plan and scene token dimensions do not align')
        num_scene_tokens = scene_tokens.shape[1]
        target = plan_tokens.reshape(batch_size * num_steps, 1, channels)
        memory = scene_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * num_steps, num_scene_tokens, channels)
        interaction = SceneHistoryEnhancer._run_decoder(
            self.cross, target, memory)
        scale = self.effective_residual_scale().to(dtype=plan_tokens.dtype)
        fused = target + scale * (interaction - target)
        return fused.reshape(batch_size, num_steps, channels)
