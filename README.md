<div align="center">

<h2>PointTTT: A Test-Time Training Framework for Point Cloud Understanding</h2>

</div>

![PointTTT Framework](Asr.jpg)

## Introduction

This repository contains the official implementation of **PointTTT**, a test-time training framework for point cloud understanding. PointTTT adapts point cloud models at inference time with lightweight self-supervised objectives, improving robustness across object classification, few-shot recognition, and indoor scene segmentation benchmarks.

The codebase provides training and evaluation scripts for ModelNet40 classification and ScanNet segmentation, together with configuration files and model implementations used in the paper.

## Environment

```bash
conda create -n pointttt python=3.10 -y
conda activate pointttt
pip install -r requirements.txt
```

Please install the PyTorch and CUDA versions that match your local GPU environment before installing the remaining dependencies.

## Dataset

### ModelNet40 Classification

Download ModelNet40 and place it under:

```text
data/ModelNet40/
```

Expected layout:

```text
data/ModelNet40/
  modelnet40_shape_names.txt
  modelnet40_train.txt
  modelnet40_test.txt
  airplane/
  chair/
  ...
```

### Few-Shot Classification

PointTTT follows the few-shot setting and data preparation protocol used by Point-BERT. Please download the few-shot data splits from the Point-BERT dataset instructions:

[Point-BERT Dataset README](https://github.com/Julie-tang00/Point-BERT/blob/master/DATASET.md)

### ScanNet Segmentation

Download and preprocess ScanNet following the official ScanNet data instructions, then place the processed files under:

```text
data/ScanNet/
```

Expected layout:

```text
data/ScanNet/
  train/
  val/
  test/
```

## Training and Evaluation

### ModelNet40 Classification

Train PointTTT on ModelNet40:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/cls_modelnet.py --config configs/cls_m40.yaml --exp_name pointttt_modelnet40
```

Evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/cls_modelnet.py --config configs/cls_m40.yaml --exp_name pointttt_modelnet40 --resume path/to/checkpoint.pth --test
```

### ScanNet Segmentation

Train PointTTT on ScanNet:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/seg_scannet.py --config configs/seg_scannet.yaml --exp_name pointttt_scannet
```

Evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/seg_scannet.py --config configs/seg_scannet.yaml --exp_name pointttt_scannet --resume path/to/checkpoint.pth --test
```

## Repository Structure

```text
PointTTT-code/
  configs/        Configuration files
  datasets/       Dataset loaders and preprocessing utilities
  models/         PointTTT model definitions
  scripts/        Dataset and experiment helper scripts
  tools/          Training and evaluation entry points
```

## Acknowledgement

This implementation is adapted from [Point-Mamba](https://github.com/IRMVLab/Point-Mamba). We thank the authors for releasing their codebase.
