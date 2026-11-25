# STIP

STIP is an end-to-end autonomous driving framework with historical trajectory and scene enhancement.

## Installation

Tested on **Ubuntu 22.04 (x86_64), NVIDIA H20**, with Python 3.8.20, PyTorch 2.0.1, CUDA 11.8, and GCC 11.4. Prerequisites: Conda, Git, GNU patch, `libgl1`, and `libglib2.0-0`. Run the steps in order on a machine with a visible GPU.

### 1. Create and activate the environment

```bash
git clone https://github.com/Forgotten0116/STIP.git
cd STIP
conda create -n stip -y --file docs/stip-conda-linux-64.lock.txt
conda activate stip

conda env config vars set CUDA_HOME="$CONDA_PREFIX" PYTHONNOUSERSITE=1
conda activate stip
```

### 2. Install PyTorch and dependencies

```bash
python -m pip install --no-deps torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-deps -r requirements.txt
```

### 3. Install MMCV-full

The patches below provide compatibility with the tested PyTorch and dependency versions.

```bash
export MAX_JOBS=4
mkdir -p "$CONDA_PREFIX/src"
python -m pip download --no-deps --no-binary=:all: --no-build-isolation mmcv-full==1.4.0 -d "$CONDA_PREFIX/src"
tar -xzf "$CONDA_PREFIX/src/mmcv-full-1.4.0.tar.gz" -C "$CONDA_PREFIX/src"
patch -N -p1 -d "$CONDA_PREFIX/src/mmcv-full-1.4.0" < docs/mmcv-1.4.0-torch2.patch
python -m pip install --no-deps --no-build-isolation -e "$CONDA_PREFIX/src/mmcv-full-1.4.0"
```

### 4. Install MMDetection and MMSegmentation

`mmdet==2.14.0`, `mmsegmentation==0.14.1`, and `timm==0.6.13` are installed in Step 2 via `requirements.txt`.

### 5. Install MMDetection3D

```bash
git clone --branch v0.17.1 --depth 1 https://github.com/open-mmlab/mmdetection3d.git "$CONDA_PREFIX/src/mmdetection3d-0.17.1-h20"
patch -N -p1 -d "$CONDA_PREFIX/src/mmdetection3d-0.17.1-h20" < docs/mmdet3d-0.17.1-torch2.patch
python -m pip install --no-deps --no-build-isolation -e "$CONDA_PREFIX/src/mmdetection3d-0.17.1-h20"
```

### 6. Install nuScenes DevKit

`nuscenes-devkit==1.1.9` and `yapf==0.40.1` are installed in Step 2 via `requirements.txt`.

### 7. Verify the installation

```bash
python -m pip check
```

<details>
<summary>Optional: check the reference environment and CUDA operators</summary>

This checks the exact package set and builds, project imports, and CUDA operators. Additional packages or different versions will fail the strict environment check.

```bash
python tools/check_stip_environment.py
```

</details>

<details>
<summary>Troubleshooting: C++ runtime library errors</summary>

If a command reports a missing `GLIBCXX_*` version or loads an incompatible system C++ runtime, run the following in the activated environment and retry:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
```

</details>

## Data preparation

Download and arrange the following files:

- [nuScenes v1.0 trainval](https://www.nuscenes.org/download): metadata, camera images, and LiDAR.
- [nuScenes map expansion](https://github.com/nutonomy/nuscenes-devkit#map-expansion).
- VAD temporal annotations: [train.pkl](https://drive.google.com/file/d/1OVd6Rw2wYjT_ylihCixzF6_olrAQsctx/view?usp=sharing) and [val.pkl](https://drive.google.com/file/d/16DZeA-iepMCaeyi57XSXL3vYyhrOQI9S/view?usp=sharing).

```text
STIP/
├── projects/
├── tools/
└── data/
    └── nuscenes/
        ├── maps/
        │   └── expansion/
        ├── samples/
        ├── sweeps/
        ├── v1.0-trainval/
        ├── vad_nuscenes_infos_temporal_train.pkl
        └── vad_nuscenes_infos_temporal_val.pkl
```

## Training

All released checkpoints were trained on 8 GPUs with one sample per GPU (global batch size: 8). With the per-GPU batch size unchanged, changing the number of training GPUs changes the global batch size and may affect the results.

Run from the repository root with the `stip` environment activated:

```bash
SEED=0 bash tools/nusc_my_train.sh stip/default 8
```

The [Swin-Tiny pretrained weights](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth) are downloaded automatically.

## Evaluation and testing

Place a released checkpoint at `path/to/pth`. Run from the repository root with the `stip` environment activated:

```bash
./tools/dist_test.sh projects/configs/stip/default.py path/to/pth 1
```

The three released seed checkpoints share this configuration. Change the checkpoint filename for SEED1 or SEED2; the last argument is the number of GPUs.

## Results and Checkpoints

| Method | 1s L2 (m) | 2s L2 (m) | 3s L2 (m) | avg L2 (m) | 1s col (%) | 2s col (%) | 3s col (%) | avg col (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STIP | 0.32905±0.01935 | 0.63630±0.02768 | 1.05847±0.03406 | **0.67461±0.02699** | 0.0651±0.0482 | 0.1009±0.0508 | 0.4146±0.0754 | **0.1935±0.0543** |

checkpoint [download](https://huggingface.co/Forgotten0116/STIP/tree/main/checkpoint).

## Acknowledgements
This project is based on [LAW](https://github.com/BraveGroup/LAW), with data preparation and evaluation components from [VAD](https://github.com/hustvl/VAD), [OpenMMLab](https://github.com/open-mmlab), and [nuScenes](https://github.com/nutonomy/nuscenes-devkit).
