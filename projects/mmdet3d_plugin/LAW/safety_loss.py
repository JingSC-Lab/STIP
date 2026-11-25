"""Training-only box-clearance penalty for the selected LAW trajectory.

Uses the nuScenes/VAD 34-D agent attributes and the continuous box geometry
underlying metric_stp3.py. This is a differentiable surrogate, not the raster
collision metric or a reconstruction of the deleted Safety implementation.
"""

import math

import torch
from torch import nn


class TrajectorySafetyLoss(nn.Module):
    def __init__(self, enabled=False, weight=0.01, delay_epochs=3,
                 warmup_epochs=3, margin=0.2):
        super().__init__()
        if not isinstance(enabled, bool):
            raise ValueError('Safety enabled must be a boolean')
        for name, value in (('weight', weight), ('margin', margin)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Safety {name} must be finite and non-negative')
        for name, value in (('delay_epochs', delay_epochs),
                            ('warmup_epochs', warmup_epochs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'Safety {name} must be a non-negative integer')
        self.enabled = enabled
        self.weight = float(weight)
        self.delay_epochs = delay_epochs
        self.warmup_epochs = warmup_epochs
        self.margin = float(margin)

    def weight_at_epoch(self, epoch):
        """Runner epoch is zero based; delay=3 leaves displayed E1--E3 off."""
        if not self.enabled or epoch < self.delay_epochs:
            return 0.0
        fraction = min(1.0, (epoch - self.delay_epochs + 1)
                       / max(1, self.warmup_epochs))
        return self.weight * fraction

    @staticmethod
    def box_clearance(ego_xy, agent_xy, agent_yaw, half_width, half_length):
        """Signed separating-axis gap for fixed-heading ego / oriented agents.

        Inputs broadcast to [agents, steps]. Negative means box overlap.
        Ego dimensions and +0.5 m forward center offset follow PlanningMetric.
        Agent yaw is the angle of its length axis in the current lidar frame.
        """
        dx = ego_xy[None, :, 0] - agent_xy[..., 0]
        dy = ego_xy[None, :, 1] + 0.5 - agent_xy[..., 1]
        c, s = agent_yaw.cos(), agent_yaw.sin()
        ac, ass = c.abs(), s.abs()
        ego_hw, ego_hl = 1.85 / 2, 4.084 / 2
        gaps = torch.stack((
            dx.abs() - (ego_hw + half_length * ac + half_width * ass),
            dy.abs() - (ego_hl + half_length * ass + half_width * ac),
            (dx * c + dy * s).abs()
            - (half_length + ego_hw * ac + ego_hl * ass),
            (-dx * s + dy * c).abs()
            - (half_width + ego_hw * ass + ego_hl * ac),
        ), dim=-1)
        return gaps.max(dim=-1).values

    def forward(self, pred_deltas, gt_bboxes_3d, gt_attr_labels, ego_fut_masks):
        """Return unweighted risk, averaged over valid ego future steps.

        pred_deltas: [B, 6, 2], already selected by the navigation command.
        GT lists contain only the CURRENT frame, in its lidar coordinates.
        Per step, use the maximum valid-agent penalty: empty/far agents cannot
        dilute a dangerous neighbour. No GT enters the prediction network.
        """
        if pred_deltas.ndim != 3 or pred_deltas.shape[1:] != (6, 2):
            raise ValueError('Safety expects command-selected deltas [B, 6, 2]')
        batch = pred_deltas.shape[0]
        if gt_bboxes_3d is None or gt_attr_labels is None or ego_fut_masks is None:
            raise ValueError('Active Safety requires current agent GT and ego masks')
        if len(gt_bboxes_3d) != batch or len(gt_attr_labels) != batch:
            raise ValueError('Safety GT batch size differs from prediction batch')
        if ego_fut_masks.numel() != batch * 6:
            raise ValueError('Safety ego mask must contain B * 6 entries')
        # Keep the cumsum differentiable, in float32 even under mixed precision.
        ego_xy = pred_deltas.float().cumsum(dim=1)
        ego_valid = ego_fut_masks.detach().to(ego_xy.device).reshape(batch, 6) > 0
        total = ego_xy.sum() * 0.0
        for b in range(batch):
            boxes = gt_bboxes_3d[b]
            boxes = boxes.tensor if hasattr(boxes, 'tensor') else boxes
            boxes = boxes.detach().to(device=ego_xy.device, dtype=torch.float32)
            attrs = gt_attr_labels[b].detach().to(
                device=ego_xy.device, dtype=torch.float32)
            if boxes.ndim != 2 or boxes.shape[1] < 7 or attrs.shape != (len(boxes), 34):
                raise ValueError('Safety requires aligned [N, >=7] boxes / [N, 34] attributes')
            # Match the vehicle + pedestrian occupancy categories in the metric.
            category = attrs[:, 27]
            keep = (((category >= 2) & (category <= 8))
                    | ((category >= 14) & (category <= 23)))
            keep &= torch.isfinite(boxes[:, :7]).all(dim=-1)
            keep &= (boxes[:, 3:5] > 0).all(dim=-1)
            boxes, attrs = boxes[keep], attrs[keep]
            if len(boxes) == 0:
                continue
            deltas = attrs[:, :12].reshape(-1, 6, 2)
            yaw_deltas = attrs[:, 28:34]
            valid = ((attrs[:, 12:18] > 0)
                     & torch.isfinite(deltas).all(dim=-1)
                     & torch.isfinite(yaw_deltas))
            # Missing an increment makes all subsequent absolute positions unknown.
            valid = valid.long().cumprod(dim=1).bool()
            deltas = torch.where(valid[..., None], deltas, torch.zeros_like(deltas))
            yaw_deltas = torch.where(valid, yaw_deltas, torch.zeros_like(yaw_deltas))
            agent_xy = boxes[:, None, :2] + deltas.cumsum(dim=1)
            agent_yaw = -(boxes[:, 6:7] + math.pi / 2) + yaw_deltas.cumsum(dim=1)
            gap = self.box_clearance(ego_xy[b], agent_xy, agent_yaw,
                                     boxes[:, 3:4] / 2, boxes[:, 4:5] / 2)
            risk = (self.margin - gap).clamp_min(0)
            risk = risk.masked_fill(~(valid & ego_valid[b][None]), 0)
            total = total + risk.max(dim=0).values.sum()
        return total / ego_valid.sum().clamp_min(1)
