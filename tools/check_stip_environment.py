"""Check the locked packages, local editable sources, and real CUDA operators."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path(sys.prefix).resolve()
sys.path.insert(0, str(ROOT))
expected = {}
for line in (ROOT / 'requirements.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, version = line.strip().split('==', 1)
        expected[canonicalize_name(name)] = str(Version(version))
expected.update({'mmcv-full': '1.4.0', 'mmdet3d': '0.17.1'})
actual = {canonicalize_name(d.metadata['Name']): str(Version(d.version))
          for d in metadata.distributions()}
assert actual == expected, {k: (expected.get(k), actual.get(k))
                            for k in expected.keys() | actual.keys()
                            if expected.get(k) != actual.get(k)}
locked = {}
for line in (ROOT / 'docs/stip-conda-linux-64.lock.txt').read_text().splitlines():
    if line.startswith('https://'):
        filename, checksum = line.rsplit('/', 1)[1].split('#')
        locked[filename] = checksum
installed = {}
for path in (PREFIX / 'conda-meta').glob('*.json'):
    record = json.loads(path.read_text())
    installed[record['fn']] = record['md5']
assert installed == locked, 'Conda artifacts differ from the explicit lock'

import torch
import mmcv
import mmdet3d
import mmdet, mmseg, mmengine, numba, sqlite3, IPython
import projects.mmdet3d_plugin
from mmcv.ops import get_compiling_cuda_version
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmdet3d.ops import furthest_point_sample

assert torch.__version__ == '2.0.1+cu118' and torch.version.cuda == '11.8'
assert mmcv.__version__ == '1.4.0' and mmdet3d.__version__ == '0.17.1'
for module in (mmcv, mmdet3d):
    Path(module.__file__).resolve().relative_to(PREFIX / 'src')
v = torch.randn(1, 4, 1, 4, device='cuda', requires_grad=True)
s = torch.tensor([[2, 2]], device='cuda', dtype=torch.long)
i = torch.tensor([0], device='cuda', dtype=torch.long)
p = torch.full((1, 1, 1, 1, 1, 2), 0.5, device='cuda')
a = torch.ones(1, 1, 1, 1, 1, device='cuda')
out = MultiScaleDeformableAttnFunction.apply(v, s, i, p, a, 64)
out.sum().backward()
assert torch.isfinite(out).all() and torch.isfinite(v.grad).all()
assert furthest_point_sample(torch.rand(1, 8, 3, device='cuda'), 2).shape == (1, 2)
torch.cuda.synchronize()
print('Python:', sys.executable)
print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('MMCV:', mmcv.__file__, 'compiled CUDA:', get_compiling_cuda_version())
print('MMDetection3D:', mmdet3d.__file__)
print(f'PASS: {len(actual)} Python packages, {len(installed)} Conda artifacts, LAW imports, CUDA forward/backward and FPS')
