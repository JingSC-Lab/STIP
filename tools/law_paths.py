"""Repository-relative configuration and private local-path logging."""

import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def enter_repository():
    """All public entry points interpret relative paths from the repository."""
    sys.path.insert(0, str(ROOT))
    os.chdir(str(ROOT))


def relative_config_paths(value):
    """Copy configuration values for export, without changing runtime values."""
    if isinstance(value, dict):
        return {key: relative_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relative_config_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(relative_config_paths(item) for item in value)
    if isinstance(value, str) and os.path.isabs(value):
        return os.path.relpath(value, str(ROOT))
    return value


def scrub_local_paths(text):
    """Keep diagnostics useful while hiding repository and environment roots."""
    prefixes = {str(ROOT): '<REPO>', str(ROOT.parent): '<WORKSPACE>',
                str(Path(sys.prefix).resolve()): '<ENV>',
                str(Path.home()): '<HOME>'}
    for name in ('CONDA_PREFIX', 'CUDA_HOME'):
        path = os.environ.get(name)
        if path and os.path.isabs(path):
            prefixes[str(Path(path).resolve())] = '<ENV>'
    conda = os.environ.get('CONDA_EXE')
    if conda and os.path.isabs(conda):
        prefixes[str(Path(conda).resolve().parents[1])] = '<CONDA_BASE>'
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix != os.path.sep:
            text = text.replace(prefix, prefixes[prefix])
    return text


class LocalPathFilter(logging.Filter):
    def filter(self, record):
        record.msg = scrub_local_paths(record.getMessage())
        record.args = ()
        return True


def install_log_privacy_filter(logger):
    # Handler filters cover child loggers too; logger filters do not.
    for handler in logger.handlers:
        if not any(isinstance(f, LocalPathFilter) for f in handler.filters):
            handler.addFilter(LocalPathFilter())
