"""Conservative trajectory-query history enhancement for LAW."""

import torch
import torch.nn as nn


class TrajectoryHistoryEnhancer(nn.Module):
    """Fuse time-aligned historical waypoint queries into current queries.

    The queue stores only detached query tensors and their frame metadata. A
    historical query at future step ``s + offset`` is paired with current step
    ``s``. This uses the real frame offset rather than assuming that queue
    length equals elapsed time.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        ffn_dim=1024,
        max_history=1,
        max_residual=0.2,
    ):
        super().__init__()
        if not isinstance(max_history, int) or max_history < 1:
            raise ValueError('max_history must be a positive integer')
        if max_residual < 0:
            raise ValueError('max_residual must be non-negative')
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.cross = nn.TransformerDecoder(layer, num_layers=1)
        self.max_history = max_history
        self.max_residual = float(max_residual)

        # Exact identity at initialization. The bounded scalar first learns
        # whether the historical correction is useful; cross-attention then
        # starts receiving non-zero gradients as the scalar leaves zero.
        self.residual_parameter = nn.Parameter(torch.zeros(1))
        self._history = []

    @staticmethod
    def _metadata(img_metas, batch_size):
        if not isinstance(img_metas, (list, tuple)):
            raise TypeError('img_metas must be a list or tuple')
        if len(img_metas) != batch_size:
            raise ValueError(
                'img_metas batch size does not match query batch size: '
                f'{len(img_metas)} != {batch_size}')

        frame_indices = []
        scene_tokens = []
        for meta in img_metas:
            if 'frame_idx' not in meta:
                raise KeyError(
                    'frame_idx is required for trajectory history alignment')
            if 'scene_token' not in meta:
                raise KeyError(
                    'scene_token is required for trajectory history alignment')
            frame_indices.append(int(meta['frame_idx']))
            scene_tokens.append(meta['scene_token'])
        return tuple(frame_indices), tuple(scene_tokens)

    def reset(self):
        """Clear all runtime history at a clip boundary."""
        self._history.clear()

    def push(self, queries, img_metas):
        """Append one detached historical query tensor and its metadata."""
        if queries.ndim != 3:
            raise ValueError(
                'queries must have shape [B, T, C], got '
                f'{tuple(queries.shape)}')
        frame_indices, scene_tokens = self._metadata(
            img_metas, queries.shape[0])

        if self._history and queries.shape != self._history[-1][0].shape:
            raise ValueError(
                'Historical query shape changed within a clip: '
                f'{tuple(self._history[-1][0].shape)} -> '
                f'{tuple(queries.shape)}')

        self._history.append(
            (queries.detach(), frame_indices, scene_tokens))
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def effective_residual_scale(self):
        """Return the bounded scalar multiplying the historical correction."""
        return self.max_residual * torch.tanh(self.residual_parameter)

    def _zero_parameter_dependency(self, reference):
        """Keep DDP reduction hooks active when a clip has no valid history."""
        zero = reference.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.reshape(-1)[0] * 0.0
        return zero

    def forward(self, current_queries, img_metas):
        if current_queries.ndim != 3:
            raise ValueError(
                'current_queries must have shape [B, T, C], got '
                f'{tuple(current_queries.shape)}')
        batch_size, num_steps, _ = current_queries.shape
        current_frames, current_scenes = self._metadata(
            img_metas, batch_size)

        if not self._history:
            return current_queries + self._zero_parameter_dependency(
                current_queries)

        scale = self.effective_residual_scale().to(
            dtype=current_queries.dtype)
        enhanced_batches = []
        used_history = False

        for batch_idx in range(batch_size):
            enhanced_steps = []
            for step_idx in range(num_steps):
                aligned_history = []
                seen_frames = set()
                for queries, frame_indices, scene_tokens in self._history:
                    if queries.shape != current_queries.shape:
                        raise ValueError(
                            'Current and historical query shapes differ: '
                            f'{tuple(current_queries.shape)} versus '
                            f'{tuple(queries.shape)}')
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
                    aligned_history.append(
                        queries[batch_idx, history_step])

                current_step = current_queries[
                    batch_idx:batch_idx + 1,
                    step_idx:step_idx + 1,
                ]
                if aligned_history:
                    memory = torch.stack(
                        aligned_history, dim=0).unsqueeze(0)
                    if current_step.is_cuda:
                        # PyTorch 2.0 + H20 may select an invalid fused kernel
                        # for zero-dropout, length-one attention. The math path
                        # is deterministic and preserves the same operation.
                        with torch.backends.cuda.sdp_kernel(
                            enable_flash=False, enable_math=True,
                            enable_mem_efficient=False):
                            interaction = self.cross(
                                tgt=current_step, memory=memory)
                    else:
                        interaction = self.cross(
                            tgt=current_step, memory=memory)
                    current_step = current_step + scale * (
                        interaction - current_step)
                    used_history = True
                enhanced_steps.append(current_step)
            enhanced_batches.append(torch.cat(enhanced_steps, dim=1))

        enhanced = torch.cat(enhanced_batches, dim=0)
        if not used_history:
            enhanced = enhanced + self._zero_parameter_dependency(enhanced)
        return enhanced
