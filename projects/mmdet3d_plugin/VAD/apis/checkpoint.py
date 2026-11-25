"""Path-free, resumable epoch checkpoints for the LAW training runner."""

import copy
import random
import re
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import is_module_wrapper
from mmcv.runner import Hook, get_dist_info, save_checkpoint
from mmcv.runner.hooks import CheckpointHook, EMAHook


def _public_values(value):
    """Only structured numbers and simple labels enter checkpoint settings."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_]+', value):
        return value
    if isinstance(value, (tuple, list)):
        return [_public_values(v) for v in value]
    if isinstance(value, dict):
        return {_public_values(k): _public_values(v) for k, v in value.items()}
    raise ValueError('Checkpoint settings must contain only path-free labels and numbers')


def resume_settings(cfg):
    """Record the training choices needed to check an epoch-boundary resume."""
    head = cfg.model.get('pts_bbox_head', {})
    fields = ('use_trajectory_history', 'history_queue_len', 'history_max_residual',
              'use_scene_history', 'scene_tokens_per_view', 'scene_history_queue_len',
              'scene_interact_steps', 'scene_max_residual', 'num_proposals',
              'num_traj_modal', 'hidden_channel')
    settings = dict(
        model=cfg.model.type,
        head={k: head[k] for k in fields if k in head},
        step_embedding_type=head.get('scene_step_embedding_type', 'learned'),
        lr_config=dict(cfg.lr_config),
        optimizer_type=cfg.optimizer.type,
        grad_clip=cfg.optimizer_config.get('grad_clip'),
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        history_num_frames=cfg.data.train.get('history_num_frames'),
        history_stride=cfg.data.train.get('history_stride'),
        safety=dict(cfg.model.get('safety_cfg', {})),
        ema_enabled=cfg.get('ema_enabled', False),
        ema_decay=cfg.get('ema_decay', 0.9999),
    )
    # Target max_epochs is intentionally excluded so a fixed schedule can continue.
    return _public_values(settings)


def _rng_state():
    numpy = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=(numpy[0], torch.from_numpy(numpy[1].astype(np.int64)),
               int(numpy[2]), int(numpy[3]), float(numpy[4])),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    )


def save_training_checkpoint(runner, out_dir, filename_tmpl='epoch_{}.pth',
                             save_optimizer=True, meta=None, create_symlink=True,
                             file_client_args=None):
    """Save training state using an allowlist instead of copying runner metadata."""
    source = dict(meta or {})
    source.update(runner.meta or {})
    checkpoint_meta = dict(checkpoint_format=1, epoch=runner.epoch + 1,
                           iter=runner.iter, step=runner.iter,
                           world_size=runner.world_size)
    if source.get('seed') is not None:
        checkpoint_meta['seed'] = int(source['seed'])
    if hasattr(runner, '_resume_settings'):
        checkpoint_meta['training_settings'] = _public_values(runner._resume_settings)
    # AMP's dynamic loss scaler is numeric state, not environment information.
    scaler = source.get('fp16', {}).get('loss_scaler', {})
    if scaler:
        checkpoint_meta['fp16'] = dict(loss_scaler=_public_values(scaler))
    best = source.get('hook_msgs', {}).get('best_score')
    if isinstance(best, (int, float)):
        checkpoint_meta['hook_msgs'] = dict(best_score=float(best))
    if save_optimizer:
        # Ordinary epoch saves collect every rank after validation. Direct saves
        # can restore RNG only for the saving rank and are marked accordingly.
        states = getattr(runner, '_checkpoint_rng_states', None)
        checkpoint_meta['rng_states'] = (states if states is not None
                                         else {get_dist_info()[0]: _rng_state()})
    filename = filename_tmpl.format(runner.epoch + 1)
    save_checkpoint(runner.model, osp.join(out_dir, filename),
                    optimizer=runner.optimizer if save_optimizer else None,
                    meta=checkpoint_meta, file_client_args=file_client_args)
    if create_symlink:
        mmcv.symlink(filename, osp.join(out_dir, 'latest.pth'))


class PrivateCheckpointHook(CheckpointHook):
    """All ranks snapshot RNG before the inherited rank-zero-only file write."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.by_epoch:
            raise ValueError('Resumable LAW checkpoints currently require by_epoch=True')

    def _save_checkpoint(self, runner):
        states = [_rng_state()]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, states[0])
            states = gathered
        runner._checkpoint_rng_states = dict(enumerate(states))
        try:
            super()._save_checkpoint(runner)
        finally:
            del runner._checkpoint_rng_states


class ResumableEMAHook(EMAHook):
    """Register EMA buffers before load, preserving them when before_run repeats."""

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        self.param_ema_buffer = {}
        self.model_parameters = dict(model.named_parameters(recurse=True))
        buffers = dict(model.named_buffers(recurse=True))
        for name, value in self.model_parameters.items():
            buffer_name = 'ema_' + name.replace('.', '_')
            self.param_ema_buffer[name] = buffer_name
            if buffer_name not in buffers:
                model.register_buffer(buffer_name, value.detach().clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))


class RestoreRNGHook(Hook):
    def before_run(self, runner):
        state = getattr(runner, '_resume_rng_state', None)
        if state is None:
            return
        random.setstate(state['python'])
        numpy = state['numpy']
        np.random.set_state((numpy[0], numpy[1].cpu().numpy().astype(np.uint32),
                             numpy[2], numpy[3], numpy[4]))
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] is not None:
            if not torch.cuda.is_available():
                raise RuntimeError('This checkpoint requires CUDA to restore its RNG state')
            torch.cuda.set_rng_state(state['cuda'].cpu())
        del runner._resume_rng_state


def resume_with_optional_ema(runner, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu')
    meta = saved.get('meta', {})
    if not all(k in meta for k in ('epoch', 'iter', 'seed')) or 'optimizer' not in saved:
        raise ValueError('This checkpoint contains weights only, without complete training '
                         'state. It cannot be resumed; start a new run or use load_from '
                         'for an explicitly new fine-tuning run.')
    if meta['seed'] != (runner.meta or {}).get('seed'):
        raise ValueError('Resume must use the checkpoint seed; pass --seed accordingly')
    if meta.get('world_size', runner.world_size) != runner.world_size:
        raise ValueError('Resume must keep the same GPU/process count')
    if meta.get('training_settings') != getattr(runner, '_resume_settings', None):
        raise ValueError('Resume must keep model, data, optimizer, Safety, EMA and LR '
                         'settings; only the target training duration may change')
    ema_hooks = [h for h in runner.hooks if isinstance(h, EMAHook)]
    saved_ema = any(k.startswith(('ema_', 'module.ema_')) for k in saved['state_dict'])
    if len(ema_hooks) > 1 or saved_ema != bool(ema_hooks):
        raise ValueError('Resume must keep the checkpoint EMA setting')
    if ema_hooks:
        hook = ema_hooks[0]
        if not isinstance(hook, ResumableEMAHook):
            raise TypeError('Resume requires ResumableEMAHook')
        if hook.checkpoint not in (None, checkpoint):
            raise ValueError('Conflicting EMA resume checkpoints')
        hook.before_run(runner)
    state = meta.get('rng_states', {}).get(get_dist_info()[0])
    if state is None:
        raise ValueError("Checkpoint is missing this rank's RNG state")
    runner._resume_rng_state = copy.deepcopy(state)
    del saved
    # Restore before LR/AMP hooks initialize; EMA buffers already exist.
    runner.resume(checkpoint, map_location='cpu')
    # TextLoggerHook serializes runner.meta as JSON; RNG tensors live only in
    # the checkpoint and the pending restoration state, never in log metadata.
    runner.meta.pop('rng_states', None)
