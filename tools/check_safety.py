"""Semantic checks for LAW Safety; run with the lawh20 Python interpreter."""

import argparse
import importlib.util
import math
from pathlib import Path
import unittest

import torch
from mmcv import Config
from law_train_options import add_law_arguments, resolve_law_options

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'law_safety', ROOT / 'projects/mmdet3d_plugin/LAW/safety_loss.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TrajectorySafetyLoss = MODULE.TrajectorySafetyLoss


def scene(x=1.5):
    # A width-2, length-4 agent pointing along +y, centered at ego's +0.5 offset.
    boxes = torch.tensor([[x, 0.5, 0.0, 2.0, 4.0, 1.5, -math.pi]])
    attrs = torch.zeros(1, 34)
    attrs[:, 12:18] = 1
    attrs[:, 27] = 17  # nuScenes vehicle category used by the occupancy metric.
    return boxes, attrs


class SafetySemantics(unittest.TestCase):
    def setUp(self):
        self.loss = TrajectorySafetyLoss(enabled=True)
        self.pred = torch.zeros(1, 6, 2, requires_grad=True)
        self.mask = torch.ones(1, 1, 1, 6)
        self.boxes, self.attrs = scene()

    def risk(self, boxes=None, attrs=None, mask=None, pred=None):
        return self.loss(self.pred if pred is None else pred,
                         [self.boxes if boxes is None else boxes],
                         [self.attrs if attrs is None else attrs],
                         self.mask if mask is None else mask)

    def test_delay_and_ramp_have_no_off_by_one(self):
        actual = [self.loss.weight_at_epoch(e) for e in range(7)]
        expected = [0, 0, 0, 0.01 / 3, 0.02 / 3, 0.01, 0.01]
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b)
        instant = TrajectorySafetyLoss(enabled=True, delay_epochs=0, warmup_epochs=0)
        self.assertEqual(instant.weight_at_epoch(0), 0.01)
        disabled = TrajectorySafetyLoss(enabled=False)
        self.assertEqual(disabled.weight_at_epoch(100), 0)

    def test_geometry_and_cumsum_gradient_direction(self):
        self.boxes.requires_grad_()
        self.attrs.requires_grad_()
        risk = self.risk()
        self.assertAlmostEqual(risk.item(), 0.625, places=5)
        risk.backward()
        torch.testing.assert_close(self.pred.grad[0, :, 0], torch.arange(6, 0, -1) / 6)
        torch.testing.assert_close(self.pred.grad[0, :, 1], torch.zeros(6))
        self.assertIsNone(self.boxes.grad)
        self.assertIsNone(self.attrs.grad)
        better = self.pred.detach() - 0.1 * self.pred.grad
        self.assertLess(self.risk(pred=better).item(), risk.item())

    def test_agent_future_deltas_are_accumulated_once(self):
        self.attrs[:, :12:2] = 0.2
        self.assertAlmostEqual(self.risk().item(), (0.425 + 0.225 + 0.025) / 6, places=5)

    def test_agent_rotation_and_future_yaw(self):
        rotated = self.boxes.clone()
        rotated[:, 6] = -math.pi / 2
        self.assertAlmostEqual(self.risk(boxes=rotated).item(), 1.625, places=5)
        self.attrs[:, 28] = -math.pi / 2
        self.assertAlmostEqual(self.risk().item(), 1.625, places=5)

    def test_translation_invariance(self):
        shift = torch.tensor([7.0, -4.0])
        translated_pred = self.pred.detach().clone()
        translated_pred[:, 0] += shift
        translated_boxes = self.boxes.clone()
        translated_boxes[:, :2] += shift
        torch.testing.assert_close(self.risk(), self.risk(
            pred=translated_pred, boxes=translated_boxes))

    def test_far_empty_and_invalid_agents_are_safe(self):
        far_boxes, far_attrs = scene(100)
        cases = [(far_boxes, far_attrs), (torch.empty(0, 7), torch.empty(0, 34))]
        invalid = self.attrs.clone()
        invalid[:, 12:18] = 0
        invalid[:, :12] = float('nan')
        invalid[:, 28:] = float('nan')
        cases.append((self.boxes, invalid))
        for boxes, attrs in cases:
            with self.subTest(count=len(boxes)):
                pred = self.pred.detach().clone().requires_grad_()
                risk = self.risk(boxes, attrs, pred=pred)
                self.assertEqual(risk.item(), 0)
                risk.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())
                self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_missing_increment_invalidates_later_positions(self):
        self.attrs[:, :12:2] = 0.2
        self.attrs[:, 13] = 0
        self.assertAlmostEqual(self.risk().item(), 0.425 / 6, places=5)

    def test_agent_count_does_not_dilute_penalty(self):
        boxes, attrs = scene(100)
        for extra_boxes, extra_attrs in [(boxes, attrs), (self.boxes, self.attrs)]:
            torch.testing.assert_close(self.risk(), self.risk(
                torch.cat([self.boxes, extra_boxes]),
                torch.cat([self.attrs, extra_attrs])))

    def test_masks_and_batch_normalization(self):
        pred = self.pred.detach().repeat(2, 1, 1).requires_grad_()
        mask = torch.zeros(2, 1, 1, 6)
        mask[0, ..., 0] = 1
        risk = self.loss(pred, [self.boxes, self.boxes],
                         [self.attrs, self.attrs], mask)
        self.assertAlmostEqual(risk.item(), 0.625, places=5)
        risk.backward()
        self.assertEqual(pred.grad[1].abs().sum().item(), 0)
        self.assertEqual(self.risk(mask=torch.zeros_like(self.mask)).item(), 0)

    def test_metric_categories_and_invalid_box_filter(self):
        self.attrs[:, 27] = 9
        self.assertEqual(self.risk().item(), 0)
        self.attrs[:, 27] = 2
        self.assertGreater(self.risk().item(), 0)
        self.boxes[:, 3] = 0
        self.assertEqual(self.risk().item(), 0)

    def test_no_parameters_buffers_or_rng_changes(self):
        before = torch.get_rng_state()
        module = TrajectorySafetyLoss(enabled=True)
        self.assertEqual(len(module.state_dict()), 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        risk = self.risk(pred=self.pred.detach().half().requires_grad_())
        self.assertEqual(risk.dtype, torch.float32)
        risk.backward()

    def test_invalid_configuration_is_rejected(self):
        for kwargs in [dict(weight=-1), dict(weight=float('nan')),
                       dict(margin=float('inf')), dict(delay_epochs=-1),
                       dict(warmup_epochs=1.5), dict(enabled='False')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrajectorySafetyLoss(**kwargs)

    def test_cli_presets_and_precedence(self):
        parser = argparse.ArgumentParser()
        add_law_arguments(parser)
        for mode, frames, stride in [('H1Q1', 1, 3), ('H3Q3', 3, 1)]:
            for ema in ['on', 'off']:
                for safety in ['on', 'off']:
                    cfg = Config.fromfile(str(ROOT / 'projects/configs/stip/default.py'))
                    cfg.model.safety_cfg.weight = 0.8
                    args = parser.parse_args([
                        '--history-mode', mode, '--ema', ema, '--ema-decay', '0.999',
                        '--safety', safety, '--safety-weight', '0.01',
                        '--safety-delay-epochs', '3', '--safety-warmup-epochs', '3'])
                    resolve_law_options(cfg, args)
                    self.assertEqual(cfg.model.safety_cfg['weight'], 0.01)
                    self.assertEqual(cfg.model.safety_cfg['enabled'], safety == 'on')
                    self.assertEqual(cfg.ema_enabled, ema == 'on')
                    self.assertEqual(cfg.model.pts_bbox_head.history_queue_len, frames)
                    self.assertEqual(cfg.model.pts_bbox_head.scene_history_queue_len, frames)
                    for split in ('train', 'val', 'test'):
                        self.assertEqual(cfg.data[split].history_num_frames, frames)
                        self.assertEqual(cfg.data[split].history_stride, stride)
                    if ema == 'on':
                        hook = next(h for h in cfg.custom_hooks if h.get('type') == 'EMAHook')
                        self.assertAlmostEqual(hook['momentum'], 0.001)
                    if safety == 'on':
                        self.assertEqual(sum(h.get('type') == 'CustomSetEpochInfoHook'
                                             for h in cfg.custom_hooks), 1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
