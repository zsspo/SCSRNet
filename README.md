# SCSRNet: A multi-scale spatial-conditioned spectral routing network for hyperspectral and multispectral image fusion

PyTorch implementation of:

> **SCSRNet: A multi-scale spatial-conditioned spectral routing network for hyperspectral and multispectral image fusion**
> *International Journal of Applied Earth Observation and Geoinformation*, 2026.
> [https://doi.org/10.1016/j.jag.2026.105547](https://doi.org/10.1016/j.jag.2026.105547)

## Overview

**SCSRNet** is a multi-scale spatial-conditioned spectral routing network for hyperspectral and multispectral image fusion (HMIF). It fuses a low-resolution hyperspectral image (LR-HSI) with a high-resolution multispectral image (HR-MSI) to reconstruct a high-resolution HSI.

The network is built around three key designs:

- **Dynamic Fusion Gate (DFG)** — adaptively balances HSI and MSI contributions based on local spatial characteristics.
- **Large-Kernel Spatial–Spectral Residual Block (LK-SSRB)** — a Multi-Scale Spatial Perception Module (MSPM) captures spatial context with large receptive fields, while a Spatial-Conditioned Spectral Routing Module (SCSRM) dynamically establishes spectral dependencies guided by spatial priors.
- **Residual Scaling Connection (RSC)** — improves optimization stability and feature propagation.

Experiments on Pavia Center, Chikusei, MDAS, and a self-collected UAV-based Daxing dataset show that SCSRNet consistently outperforms seven state-of-the-art methods while keeping computational complexity low.

<p align="center">
  <img src="assets/arch.jpg" width="90%">
</p>

## Environment

Tested with **CUDA 11.8 + Python 3.9 + PyTorch 2.0**.

```bash
# Option 1: conda (recommended)
conda env create -f environment.yml
conda activate lk

# Option 2: pip + PyTorch with cu118
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install einops timm kornia opencv-python scipy numpy tifffile \
    matplotlib pandas wandb tensorboard torchsummary torchstat
```

> **Note on GDAL:** GeoTIFF I/O relies on `gdal` (`from osgeo import gdal`).

## Data Preparation

Datasets should be placed under the `datasets/` directory. Both `.mat` and `.tif` formats are supported.

### Supported datasets

| Dataset | Format | Loader class in `dataloaders/HSI_datasets.py` | Download |
|---------|--------|-----------------------------------------------|----------|
| Pavia Center | `.mat` | `pavia_dataset` | https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |
| Chikusei    | `.mat` | `chikusei_dataset` | https://naotoyokoya.com/Download.html |
| MDAS        | `.mat` / `.tif` | `mdasn_dataset` | https://mediatum.ub.tum.de/1657312 |
| Daxing      | `.mat` / `.tif` | `dx_dataset` | [Baidu Netdisk](#daxing) |

### Pavia & Chikusei

To directly use the provided dataloader, process the Pavia and Chikusei datasets through the MATLAB toolbox **Hyperspec_Chikusei_MATLAB** (Naoto Yokoya, 2016):

- http://park.itc.u-tokyo.ac.jp/sal/hyperdata/Hyperspec_Chikusei_MATLAB.zip

### Daxing

The self-collected UAV-based Daxing dataset is provided in this repository under `datasets/Daxing/`.

### Custom data

For your own data organization, simply modify the data loading class in [`dataloaders/HSI_datasets.py`](dataloaders/HSI_datasets.py) and register it in the `__dataset__` dictionary. A `.mat` -> `.tif` conversion helper is provided in [`dataloaders/convert_mat_to_tiff.py`](dataloaders/convert_mat_to_tiff.py).

## Getting Started

### Train / Validate / Test

Training, validation, and test are all handled by a single entry point. The test runs automatically after training completes and saves the final prediction and metrics.

```bash
# Pavia
python train_lk_rgb.py --config='config_lk/config_rgb_lk_pavia.json'

# Daxing
python train_lk_rgb.py --config='config_lk/config_rgb_lk_dx.json'
```

Outputs (model weights, metrics, predictions, TensorBoard logs) are saved under `./{experim_name}/{model}/{train_dataset}/`.



## Citation

If you find this work useful, please cite:

```bibtex
@article{ZHOU2026105547,
title = {A multi-scale spatial-conditioned spectral routing network for hyperspectral and multispectral image fusion},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {153},
pages = {105547},
year = {2026},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2026.105547},
url = {https://www.sciencedirect.com/science/article/pii/S1569843226004632},
author = {Bo Zhou and Xianfeng Zhang and Ziyuan Feng and Miao Ren and Xiaobo Zhi},
keywords = {Images fusion, Multi-scale, Spatial-spectral features, Large-kernel convolution, Residual scaling}
}
```

