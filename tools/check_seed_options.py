"""Regression checks for single-seed configuration and checkpoints; run from LAW with its Python environment."""

import argparse
import copy
import logging
import random
import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import EpochBasedRunner, Hook

import train
from law_train_options import add_law_arguments, resolve_law_seed, resolve_law_options
from projects.mmdet3d_plugin.VAD.apis.checkpoint import (
    PrivateCheckpointHook, ResumableEMAHook, RestoreRNGHook,
    resume_with_optional_ema, save_training_checkpoint,
)
from projects.mmdet3d_plugin.VAD.apis import train as train_api
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

ROOT = Path(__file__).resolve().parents[1]


def packet():
    return (random.random(), float(np.random.rand()), float(torch.rand(())))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)

    def init_weights(self):
        # Exercise RNG use in both construction and init_weights().
        torch.nn.init.normal_(self.layer.weight)
        random.random()
        np.random.rand()


class RandomDataset(torch.utils.data.Dataset):
    CLASSES = ('test',)
    flag = np.zeros(8, dtype=np.uint8)

    def __len__(self):
        return len(self.flag)

    def __getitem__(self, index):
        return torch.tensor((index,) + packet(), dtype=torch.float64)


class SeedResolution(unittest.TestCase):
    def resolve(self, cli=(), initial=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int)
        add_law_arguments(parser)
        cfg = Config(copy.deepcopy(initial or {}))
        resolve_law_seed(cfg, parser.parse_args(cli))
        return cfg

    def test_config_seed_and_cli_precedence(self):
        self.assertEqual(self.resolve().seed, 0)
        self.assertEqual(self.resolve(initial=dict(seed=2)).seed, 2)
        cfg = self.resolve(['--seed', '1'], dict(seed=2))
        self.assertEqual(cfg.seed, 1)
        self.assertFalse(any(k in cfg for k in ('init_seed', 'train_seed', 'seed_mode')))

    def test_obsolete_config_is_rejected(self):
        for key in ('init_seed', 'train_seed', 'seed_mode'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.resolve(initial={key: 1})

    def test_invalid_seeds_fail_before_training(self):
        for value in (-1, 2**31, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(initial=dict(seed=value))

    def test_file_settings_reach_model_datasets_and_runner(self):
        parser = argparse.ArgumentParser()
        add_law_arguments(parser)
        cfg = Config.fromfile(str(ROOT / 'projects/configs/stip/default.py'))
        cfg.merge_from_dict(dict(seed=2, total_epochs=14, history_mode='H3Q3',
                                 ema_enabled=True, ema_decay=0.999,
                                 **{'model.safety_cfg.enabled': True,
                                    'model.safety_cfg.weight': 0.05}))
        resolve_law_options(cfg, parser.parse_args([]))
        self.assertEqual(cfg.seed, 2)
        self.assertEqual(cfg.runner.max_epochs, 14)
        self.assertEqual(cfg.lr_config.periods, [16])
        self.assertTrue(cfg.model.safety_cfg['enabled'])
        self.assertEqual(cfg.model.safety_cfg['weight'], 0.05)
        self.assertEqual(cfg.model.pts_bbox_head.history_queue_len, 3)
        self.assertEqual(cfg.model.pts_bbox_head.scene_history_queue_len, 3)
        for split in ('train', 'val', 'test'):
            self.assertEqual(cfg.data[split].history_num_frames, 3)
            self.assertEqual(cfg.data[split].history_stride, 1)
        hook = next(h for h in cfg.custom_hooks if h['type'] == 'EMAHook')
        self.assertAlmostEqual(hook['momentum'], 0.001)
        cfg.total_epochs = 17
        with self.assertRaises(ValueError):
            resolve_law_options(cfg, parser.parse_args([]))


class SeedFlow(unittest.TestCase):
    def run_main(self, cli):
        result = {}
        def dataset(_):
            result['dataset_rng'] = packet()
            return RandomDataset()
        def capture(model, datasets, cfg, **kwargs):
            result['weights'] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            result['training_rng'] = packet()
            result['cfg'] = cfg
        with tempfile.TemporaryDirectory(prefix='seed-check-', dir=ROOT / 'work_dirs') as work:
            argv = ['train.py', str(ROOT / 'projects/configs/stip/default.py'),
                    '--work-dir', work, '--deterministic'] + list(cli)
            with patch('sys.argv', argv), \
                    patch.object(train, 'build_model', side_effect=lambda *a, **k: TinyModel()), \
                    patch.object(train, 'build_dataset', side_effect=dataset), \
                    patch.object(train, 'collect_env', return_value={'test': 'seed-flow'}), \
                    patch.object(train, 'get_root_logger'), \
                    patch.object(train_api, 'custom_train_model', side_effect=capture):
                train.main()
        return result

    def test_same_seed_repeats_weights_and_rngs(self):
        a = self.run_main(['--seed', '2'])
        b = self.run_main(['--seed', '2'])
        c = self.run_main(['--seed', '1'])
        for key in a['weights']:
            self.assertTrue(torch.equal(a['weights'][key], b['weights'][key]))
        self.assertFalse(torch.equal(a['weights']['layer.weight'], c['weights']['layer.weight']))
        self.assertEqual(a['dataset_rng'], b['dataset_rng'])
        self.assertEqual(a['training_rng'], b['training_rng'])
        self.assertNotEqual(a['dataset_rng'], c['dataset_rng'])
        self.assertEqual(a['cfg'].seed, 2)

    def test_legacy_matches_original_rng_consumption(self):
        train.set_random_seed(2, deterministic=True)
        expected = TinyModel()
        expected.init_weights()
        dataset_rng, training_rng = packet(), packet()
        actual = self.run_main(['--seed', '2'])
        self.assertEqual(actual['dataset_rng'], dataset_rng)
        self.assertEqual(actual['training_rng'], training_rng)
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(actual['weights'][key], value))

    def test_sampler_and_worker_rngs_follow_seed(self):
        def collect(seed):
            train.set_random_seed(seed)
            model = TinyModel()
            model.init_weights()
            loader = build_dataloader(
                RandomDataset(), samples_per_gpu=1, workers_per_gpu=2,
                num_gpus=1, dist=True, seed=seed,
                shuffler_sampler=dict(type='DistributedGroupSampler'),
                runner_type=dict(type='EpochBasedRunner'))
            return torch.cat(list(loader))
        a, b, c = collect(2), collect(2), collect(1)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertFalse(torch.equal(a[:, 0], c[:, 0]))


class CheckpointMetadata(unittest.TestCase):
    def make_runner(self, work, epochs=12):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        runner = EpochBasedRunner(
            model, batch_processor=lambda *a, **k: None, optimizer=optimizer,
            logger=logging.getLogger('checkpoint-check'), work_dir=work,
            max_epochs=epochs, meta=dict(seed=2))
        runner.save_checkpoint = MethodType(save_training_checkpoint, runner)
        return runner

    def test_training_state_saved_without_private_metadata(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-check-', dir=ROOT / 'work_dirs') as work:
            runner = self.make_runner(work)
            private = '/private/machine/user-secret'
            runner.meta.update(config=private, env_info=private, work_dir=private,
                               exp_name='user-secret.py', hostname='private-host',
                               hook_msgs=dict(best_score=0.2, best_ckpt=private),
                               fp16=dict(loss_scaler=dict(scale=128., growth_factor=2.,
                                   backoff_factor=0.5, growth_interval=2000, _growth_tracker=3)))
            runner.model.layer(torch.ones(2, 3)).square().mean().backward()
            runner.optimizer.step()
            runner._epoch = 11
            runner._iter = 100
            save_training_checkpoint(runner, work, meta=dict(epoch=999, config=private))
            path = Path(work) / 'epoch_12.pth'
            saved = torch.load(str(path), map_location='cpu', weights_only=True)
            self.assertEqual(saved['meta']['seed'], 2)
            self.assertEqual(saved['meta']['epoch'], 12)
            self.assertEqual(saved['meta']['iter'], 100)
            self.assertEqual(saved['meta']['step'], 100)
            self.assertTrue(saved['optimizer']['state'])
            self.assertEqual(saved['meta']['fp16']['loss_scaler']['scale'], 128.)
            self.assertEqual(saved['meta']['hook_msgs'], dict(best_score=0.2))
            for key in ('config', 'env_info', 'work_dir', 'exp_name', 'hostname'):
                self.assertNotIn(key, saved['meta'])
            def check_strings(value):
                if isinstance(value, str):
                    self.assertNotIn('user-secret', value)
                    self.assertNotIn('private-host', value)
                    self.assertFalse(value.startswith('/'))
                elif isinstance(value, dict):
                    for k, v in value.items():
                        check_strings(k); check_strings(v)
                elif isinstance(value, (tuple, list)):
                    for v in value: check_strings(v)
            check_strings(saved)
            self.assertEqual((Path(work) / 'latest.pth').resolve(), path)
            restored = self.make_runner(work, epochs=16)
            resume_with_optional_ema(restored, str(path))
            self.assertEqual((restored.epoch, restored.iter), (12, 100))
            for key, value in runner.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[key]))
            for key, value in runner.optimizer.state_dict()['state'].items():
                for name, tensor in value.items():
                    torch.testing.assert_close(
                        tensor, restored.optimizer.state_dict()['state'][key][name],
                        rtol=0, atol=0)

    def test_weights_only_and_wrong_seed_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-check-', dir=ROOT / 'work_dirs') as work:
            runner = self.make_runner(work)
            path = str(Path(work) / 'epoch_1.pth')
            save_training_checkpoint(runner, work, save_optimizer=False)
            with self.assertRaisesRegex(ValueError, 'weights only'):
                resume_with_optional_ema(runner, path)
            save_training_checkpoint(runner, work)
            runner.meta['seed'] = 1
            with self.assertRaisesRegex(ValueError, 'seed'):
                resume_with_optional_ema(runner, path)

    def test_epoch_resume_matches_uninterrupted_training_with_and_without_ema(self):
        class EvaluationRNG(Hook):
            def after_train_epoch(self, runner):
                # Real validation creates DataLoader iterators after optimization.
                packet()

        class RecordLR(Hook):
            def before_train_iter(self, runner):
                runner.observed_lr.append(runner.current_lr()[0])

        def process(model, data, train_mode):
            prediction = torch.nn.functional.dropout(model.layer(data[:, :3].float()),
                                                     p=0.25, training=True)
            loss = (prediction - random.random() - float(np.random.rand())).square().mean()
            return dict(loss=loss, log_vars=dict(loss=float(loss)), num_samples=len(data))

        def run(work, epochs, ema, resume=None):
            train.set_random_seed(2, deterministic=True)
            runner = self.make_runner(work, epochs)
            runner.batch_processor = process
            runner.observed_lr = []
            runner.register_training_hooks(
                dict(policy='CosineRestart', by_epoch=True, periods=[16],
                     restart_weights=[1], min_lr_ratio=0.001, warmup='linear',
                     warmup_iters=12, warmup_ratio=1 / 3), dict(grad_clip=dict(max_norm=35)),
                None, dict(interval=1, hooks=[dict(type='TextLoggerHook')]))
            if ema:
                runner.register_hook(ResumableEMAHook(momentum=0.001), priority='ABOVE_NORMAL')
            runner.register_hook(RecordLR(), priority='NORMAL')
            runner.register_hook(EvaluationRNG(), priority='LOW')
            runner.register_hook(PrivateCheckpointHook(interval=1), priority='LOWEST')
            runner.register_hook(RestoreRNGHook(), priority='LOWEST')
            if resume:
                resume_with_optional_ema(runner, resume)
            loader = torch.utils.data.DataLoader(RandomDataset(), batch_size=2, shuffle=True)
            with patch('mmcv.runner.epoch_based_runner.time.sleep'):
                runner.run([loader], [('train', 1)])
            return runner

        for ema in (False, True):
            with self.subTest(ema=ema), tempfile.TemporaryDirectory(
                    prefix='resume-check-', dir=ROOT / 'work_dirs') as work:
                full = run(str(Path(work) / 'full'), 2, ema)
                first = run(str(Path(work) / 'first'), 1, ema)
                resumed = run(str(Path(work) / 'resumed'), 2, ema,
                              str(Path(work) / 'first/epoch_1.pth'))
                self.assertEqual(full.observed_lr, first.observed_lr + resumed.observed_lr)
                self.assertEqual((resumed.epoch, resumed.iter), (2, 8))
                for key, tensor in full.model.state_dict().items():
                    torch.testing.assert_close(tensor, resumed.model.state_dict()[key], rtol=0, atol=0)
                for key, state in full.optimizer.state_dict()['state'].items():
                    for name, value in state.items():
                        torch.testing.assert_close(value, resumed.optimizer.state_dict()['state'][key][name],
                                                   rtol=0, atol=0)


if __name__ == '__main__':
    # Match the production launcher on NAS: avoid resource_sharer sockets.
    torch.multiprocessing.set_sharing_strategy('file_system')
    unittest.main(verbosity=2)
