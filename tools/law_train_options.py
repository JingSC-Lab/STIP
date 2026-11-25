"""Resolve LAW ablation options before building datasets, hooks and models."""

import math


def add_law_arguments(parser):
    parser.add_argument('--history-mode', type=str.upper, choices=['H1Q1', 'H3Q3'],
                        help='H1Q1: t-3 / queue 1; H3Q3: t-3,t-2,t-1 / queue 3')
    parser.add_argument('--history-frames', type=int,
                        help='number of historical images read per clip')
    parser.add_argument('--history-stride', type=int,
                        help='spacing of historical images in dataset frames')
    parser.add_argument('--history-queue-len', type=int,
                        help='maximum number of stored trajectory frames')
    parser.add_argument('--scene-history-queue-len', type=int,
                        help='maximum number of stored scene frames')
    parser.add_argument('--ema', choices=['on', 'off'],
                        help='use EMA for epoch validation and checkpoint weights')
    parser.add_argument('--ema-decay', type=float,
                        help='EMA decay, default 0.9999 in LAW config')
    parser.add_argument('--safety', choices=['on', 'off'],
                        help='enable training-only current-trajectory Safety')
    parser.add_argument('--safety-weight', type=float,
                        help='maximum Safety coefficient (default 0.01)')
    parser.add_argument('--safety-delay-epochs', type=int,
                        help='number of initial epochs with Safety off (default 3)')
    parser.add_argument('--safety-warmup-epochs', type=int,
                        help='ramp epochs after the delay; 0 enables full weight immediately')
    parser.add_argument('--safety-margin', type=float,
                        help='additional box clearance in metres (default 0.2)')


def resolve_law_seed(cfg, args):
    """Use one seed, without resetting RNGs after model initialization."""
    obsolete = [key for key in ('init_seed', 'train_seed', 'seed_mode') if key in cfg]
    if obsolete:
        raise ValueError('Remove obsolete split-seed fields and configure seed: '
                         + ', '.join(obsolete))
    seed = getattr(args, 'seed', None)
    seed = cfg.get('seed', 0) if seed is None else seed
    seed = 0 if seed is None else seed
    # Leave room for the DataLoader's rank/worker offset.
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError('seed must be an integer in [0, 2**31)')
    cfg.seed = seed


def resolve_law_epochs(cfg, args):
    """Resolve the same training duration in the launcher and training entry."""
    cli_epochs = getattr(args, 'epochs', None)
    runner = cfg.get('runner', {})
    epochs = (cli_epochs if cli_epochs is not None
              else cfg.get('total_epochs', runner.get('max_epochs')))
    if epochs is None:
        return
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('total_epochs / --epochs must be a positive integer')
    cfg.total_epochs = epochs
    runner = dict(runner) if runner else dict(type='EpochBasedRunner')
    runner['max_epochs'] = epochs
    cfg.runner = runner
    if cli_epochs is not None and cfg.get('checkpoint_config') is not None:
        cfg.checkpoint_config['max_keep_ckpts'] = epochs
    lr = cfg.get('lr_config', {})
    if lr.get('policy') == 'CosineRestart' and lr.get('by_epoch', True):
        periods = lr.get('periods', [])
        if not periods or any(isinstance(p, bool) or not isinstance(p, int)
                              or p < 1 for p in periods):
            raise ValueError('CosineRestart periods must be positive integers')
        if epochs > sum(periods):
            raise ValueError('Training epochs exceed the configured cosine periods; '
                             'increase cosine_period_epochs in the config')


def resolve_law_options(cfg, args):
    """CLI presets override cfg-options; explicit CLI sizes override presets.

    Write final values to every split and to the head before cfg.dump(), so the
    saved config can be used by test.py without repeating training CLI flags.
    """
    resolve_law_seed(cfg, args)
    resolve_law_epochs(cfg, args)
    mode = args.history_mode
    if mode is None:
        mode = cfg.get('history_mode')
        if mode == 'custom':
            mode = None  # Keep explicit sizes from a saved custom configuration.
    if mode not in (None, 'H1Q1', 'H3Q3'):
        raise ValueError('history_mode must be H1Q1, H3Q3, or custom')
    head = cfg.model.get('pts_bbox_head', {})
    history_args = (args.history_frames, args.history_stride,
                    args.history_queue_len, args.scene_history_queue_len)
    if mode is not None or any(value is not None for value in history_args):
        if head.get('type') != 'WaypointHead':
            raise ValueError('History CLI options require LAW WaypointHead')
        if mode is not None:
            frames, stride = (1, 3) if mode == 'H1Q1' else (3, 1)
            for split in ('train', 'val', 'test'):
                cfg.data[split].history_num_frames = frames
                cfg.data[split].history_stride = stride
            head.history_queue_len = frames
            head.scene_history_queue_len = frames
            cfg.history_mode = mode
        for value, key in ((args.history_frames, 'history_num_frames'),
                           (args.history_stride, 'history_stride')):
            if value is not None:
                if value < 1:
                    raise ValueError(f'{key} must be positive')
                for split in ('train', 'val', 'test'):
                    cfg.data[split][key] = value
        for value, key in ((args.history_queue_len, 'history_queue_len'),
                           (args.scene_history_queue_len, 'scene_history_queue_len')):
            if value is not None:
                if value < 1:
                    raise ValueError(f'{key} must be positive')
                head[key] = value
        if any(value is not None for value in history_args):
            cfg.history_mode = 'custom'

    if args.ema is not None or args.ema_decay is not None or 'ema_enabled' in cfg:
        enabled = cfg.get('ema_enabled', False) if args.ema is None else args.ema == 'on'
        decay = cfg.get('ema_decay', 0.9999) if args.ema_decay is None else args.ema_decay
        if not 0.0 < decay < 1.0:
            raise ValueError('ema_decay must lie strictly between 0 and 1')
        hooks = [hook for hook in cfg.get('custom_hooks', [])
                 if hook.get('type') != 'EMAHook']
        if enabled:
            # MMCV momentum is the new-weight fraction, not the EMA decay.
            # Swap before checkpoint / evaluation hooks and restore next epoch.
            hooks.append(dict(type='EMAHook', momentum=1.0 - decay,
                              interval=1, priority='ABOVE_NORMAL'))
        cfg.custom_hooks = hooks
        cfg.ema_enabled = enabled
        cfg.ema_decay = decay

    safety_fields = ('weight', 'delay_epochs', 'warmup_epochs', 'margin')
    safety_args = {field: getattr(args, 'safety_' + field, None)
                   for field in safety_fields}
    if (getattr(args, 'safety', None) is not None
            or any(value is not None for value in safety_args.values())
            or 'safety_cfg' in cfg.model):
        if cfg.model.type != 'LAW':
            raise ValueError('Safety options require the LAW model')
        safety = dict(enabled=False, weight=0.01, delay_epochs=3,
                      warmup_epochs=3, margin=0.2)
        safety.update(cfg.model.get('safety_cfg', {}))
        if getattr(args, 'safety', None) is not None:
            safety['enabled'] = args.safety == 'on'
        safety.update({key: value for key, value in safety_args.items()
                       if value is not None})
        if not isinstance(safety['enabled'], bool):
            raise ValueError('Safety enabled must be a boolean')
        for key in ('weight', 'margin'):
            if not math.isfinite(safety[key]) or safety[key] < 0:
                raise ValueError(f'Safety {key} must be finite and non-negative')
        for key in ('delay_epochs', 'warmup_epochs'):
            value = safety[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'Safety {key} must be a non-negative integer')
        cfg.model.safety_cfg = safety
        if safety['enabled']:
            # The existing hook also sets the head epoch; ensure direct CLI use
            # with an older LAW config cannot silently skip the Safety schedule.
            hooks = cfg.get('custom_hooks', [])
            if not any(hook.get('type') == 'CustomSetEpochInfoHook' for hook in hooks):
                hooks.append(dict(type='CustomSetEpochInfoHook'))
            cfg.custom_hooks = hooks
