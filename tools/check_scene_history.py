"""CPU semantic checks; run with the LAW Python environment from any cwd."""
import argparse
import importlib.util
from pathlib import Path

import torch
from torch import nn
from mmcv import Config

from law_train_options import add_law_arguments, resolve_law_options

REPO = Path(__file__).resolve().parents[1]


def load_module(name):
    path = REPO / 'projects/mmdet3d_plugin/LAW/dense_heads' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene = load_module('scene_history')
trajectory = load_module('trajectory_history')


def meta(frame, scene_id='a'):
    return [dict(frame_idx=frame, scene_token=scene_id,
                 ego_fut_cmd=torch.tensor([0., 0., 1.]))]


class Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.memories = []

    def forward(self, tgt, memory):
        self.memories.append(memory.detach().clone())
        return tgt


def check_token_selection():
    learner = scene.TokenLearner(channels=8, num_tokens=2)
    features = torch.arange(48.).reshape(1, 8, 2, 3)
    logits = torch.tensor([[0., 7.], [1., 2.], [5., 1.],
                           [2., 0.], [0., 3.], [1., 4.]])

    class Selection(nn.Module):
        def forward(self, x):
            return logits.unsqueeze(0).expand(x.shape[0], -1, -1)

    learner.selection = Selection()
    expected = torch.stack([
        (features.flatten(2)[0] * logits[:, token].softmax(0)).sum(-1)
        for token in range(2)], dim=0).unsqueeze(0)
    torch.testing.assert_close(learner(features), expected)
    wrong = logits.reshape(2, 6).softmax(-1) @ features.flatten(2)[0].T
    assert not torch.allclose(expected[0], wrong)


def check_history_alignment():
    # Absolute times: current frame 10, old frames 7 / 9. Ignore duplicates,
    # another scene, the current frame and future frames.
    hep = trajectory.TrajectoryHistoryEnhancer(
        embed_dims=8, num_heads=2, ffn_dim=16, max_history=8)
    history = torch.arange(6.).reshape(1, 6, 1).expand(1, 6, 8)
    for frame, sid in ((7, 'a'), (7, 'a'), (9, 'a'),
                       (8, 'b'), (10, 'a'), (11, 'a')):
        hep.push(history + frame * 10, meta(frame, sid))
    recorder = Recorder()
    hep.cross = recorder
    hep(torch.zeros_like(history), meta(10))
    expected = [[73., 91.], [74., 92.], [75., 93.], [94.], [95.]]
    assert [m[0, :, 0].tolist() for m in recorder.memories] == expected

    enhancer = scene.SceneHistoryEnhancer(
        embed_dims=8, num_views=1, num_heads=2, ffn_dim=16, max_history=8)
    current = torch.zeros(1, 2, 6, 8)
    enhancer._encode = lambda *args: current
    enhancer._history = [(q[:, None].expand(-1, 2, -1, -1), f, s)
                         for q, f, s in hep._history]
    with torch.no_grad():
        enhancer.history_offset_embedding.weight.zero_()
    recorder = Recorder()
    enhancer.history_cross = recorder
    enhancer(None, None, meta(10))
    assert [m[0, :, 0].tolist() for m in recorder.memories] == [
        [v for v in row for _ in range(2)] for row in expected]
    enhancer.reset()
    assert not enhancer._history
    hep.reset()
    assert not hep._history


def check_stepwise_fusion():
    fusion = scene.StepwiseSceneToPlan(embed_dims=8, num_heads=2, ffn_dim=16)
    recorder = Recorder()
    fusion.cross = recorder
    memory = torch.zeros(2, 3, 6, 8)
    for batch in range(2):
        for token in range(3):
            for step in range(6):
                memory[batch, token, step] = 1000 * batch + 100 * token + step
    query = torch.randn(2, 6, 8)
    torch.testing.assert_close(fusion(query, memory), query, rtol=0, atol=0)
    for batch in range(2):
        for step in range(6):
            torch.testing.assert_close(recorder.memories[0][batch * 6 + step],
                                       memory[batch, :, step])


def check_gradients_and_detach():
    for has_history in (False, True):
        enhancer = scene.SceneHistoryEnhancer(
            embed_dims=8, num_views=2, num_heads=2, ffn_dim=16, max_history=3)
        fusion = scene.StepwiseSceneToPlan(embed_dims=8, num_heads=2, ffn_dim=16)
        hist_img = torch.randn(1, 2, 8, 2, 3, requires_grad=True)
        img = torch.randn_like(hist_img, requires_grad=True)
        pos = torch.randn_like(img, requires_grad=True)
        if has_history:
            for frame in range(6, 10):
                enhancer.push(hist_img, pos, meta(frame))
            assert len(enhancer._history) == 3
            assert all(not item[0].requires_grad for item in enhancer._history)
        query = torch.randn(1, 6, 8, requires_grad=True)
        with torch.no_grad():
            # Avoid an all-dead ReLU in this tiny 8-channel test gate.
            enhancer.channel_gate.feature_projection.bias.fill_(10.0)
            enhancer.cross_residual_parameter.fill_(0.4)
            enhancer.self_residual_parameter.fill_(0.4)
            fusion.residual_parameter.fill_(0.4)
        output = fusion(query, enhancer(img, pos, meta(10)))
        (output * torch.randn_like(output)).sum().backward()
        for module in (enhancer, fusion):
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in module.parameters())
        assert hist_img.grad is None
        assert img.grad.abs().sum() > 0 and pos.grad.abs().sum() > 0
        assert query.grad.abs().sum() > 0
        assert enhancer.token_learner.selection.layers[0].weight.grad.abs().sum() > 0
        assert enhancer.temporal_unfold[0].weight.grad.abs().sum() > 0
        if has_history:
            assert enhancer.history_cross.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
        else:
            assert enhancer.cross_residual_parameter.grad.item() == 0
    # Empty HEP history must still participate in DDP reduction.
    hep = trajectory.TrajectoryHistoryEnhancer(embed_dims=8, num_heads=2, ffn_dim=16)
    query = torch.randn(1, 6, 8, requires_grad=True)
    hep(query, meta(0)).sum().backward()
    assert all(p.grad is not None for p in hep.parameters())


def check_options():
    parser = argparse.ArgumentParser()
    add_law_arguments(parser)
    for mode, count, stride in (('H1Q1', 1, 3), ('H3Q3', 3, 1)):
        for ema in ('off', 'on'):
            cfg = Config.fromfile(str(REPO / 'projects/configs/stip/default.py'))
            args = parser.parse_args(['--history-mode', mode, '--ema', ema])
            resolve_law_options(cfg, args)
            for split in ('train', 'val', 'test'):
                assert cfg.data[split].history_num_frames == count
                assert cfg.data[split].history_stride == stride
            assert cfg.model.pts_bbox_head.history_queue_len == count
            assert cfg.model.pts_bbox_head.scene_history_queue_len == count
            assert cfg.ema_enabled == (ema == 'on')
            assert sum(h['type'] == 'EMAHook' for h in cfg.custom_hooks) == (ema == 'on')
    args = parser.parse_args(['--history-mode', 'H3Q3', '--history-frames', '2',
                             '--history-stride', '2', '--history-queue-len', '4',
                             '--scene-history-queue-len', '5'])
    resolve_law_options(cfg, args)
    assert cfg.data.train.history_num_frames == 2 and cfg.data.test.history_stride == 2
    assert cfg.model.pts_bbox_head.history_queue_len == 4
    assert cfg.model.pts_bbox_head.scene_history_queue_len == 5


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for check in (check_token_selection, check_history_alignment,
                  check_stepwise_fusion, check_gradients_and_detach, check_options):
        check()
        print('PASS', check.__name__)
