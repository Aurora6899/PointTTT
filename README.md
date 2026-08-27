
<div  align="center">    
 <img src="./figure/snake.png" width = "200"  align=center />
</div>


<div align="center">
<h1>PointTTT</h1>
<h3>Simple Test-Time Training for 3D Point Cloud Analysis</h3>


# Overview

<div  align="center">    
 <img src="PointTTT.png" width = ""  align=center />
</div>

<div align="left">
 

## 1. Environment
The code has been tested on Ubuntu 20.04.

1. Python 3.10.13
    ```bash
    conda create -n your_env_name python=3.10.13
    ```

2. Install torch 2.1.1 + cu118

    ```bash
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

3. Clone this repository and install the requirements.

    ```bash
    pip install -r requirements.txt
    ```

## 2. ScanNet Segmentation

1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in scannet
    ```
    The filelist should be like this:
    ```bash

    ├── scannet
    │ ├── scans
    │ │ ├── [scene_id]
    │ │ │ ├── [scene_id].aggregation.json
    │ │ │ ├── [scene_id].txt
    │ │ │ ├── [scene_id]_vh_clean.aggregation.json
    │ │ │ ├── [scene_id]_vh_clean.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.0.010000.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.labels.ply
    │ │ │ ├── [scene_id]_vh_clean_2.ply
    │ ├── scans_test
    │ │ ├── [scene_id]
    │ │ │ ├── [scene_id].aggregation.json
    │ │ │ ├── [scene_id].txt
    │ │ │ ├── [scene_id]_vh_clean.aggregation.json
    │ │ │ ├── [scene_id]_vh_clean.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.0.010000.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.ply
    │ ├── scannetv2-labels.combined.tsv
    ```

2. **Train**: The default command uses two GPUs with OctFormer's per-GPU batch
   size of 4 (global batch size 8). Checkpoints and training logs are written to
   `logs/scannet/from_scratch_1cm`.

    ```bash
    python scripts/run_seg_scannet.py --run train --gpu 0,1 --port 10001
    ```

3. **Validate**: Reload the selected `best_model.pth`, run the official
   120-vote validation, generate per-point labels, and calculate mIoU.

    ```bash
    python scripts/run_seg_scannet.py --run validate --gpu 0,1 --port 10001
    ```
## 3. ModelNet40 Classification (Point Mamba(O))

1. **Data**: Run the following command to prepare the dataset.

    ```bash
    python tools/cls_modelnet.py
    ```

2. **Train**: Run the following command to train the network with 1 GPU. The classification accuracy on the testing set without voting is 97.4%. The code for Point Mamba(C) will be released in another branch later.
   Checkpoints will be released later.
    ```bash
    python classification.py --config configs/cls_m40.yaml SOLVER.gpu 0,
    ```

## 4. ScanObjectNN Classification

The ScanObjectNN loader reads the original HDF5 files directly and supports the
three fixed train/test protocols used by Point-BERT:

| Protocol | Config |
| --- | --- |
| OBJ_BG | `configs/cls_scanobjectnn_objbg.yaml` |
| OBJ_ONLY | `configs/cls_scanobjectnn_objonly.yaml` |
| PB_T50_RS | `configs/cls_scanobjectnn_pbt50rs.yaml` |

The configs use all 2048 points by default and do not require PointNet++ FPS:

```bash
python classification.py --config configs/cls_scanobjectnn_pbt50rs.yaml SOLVER.gpu 0,
```

To use Point-BERT-style 1024-point sampling, override the sampling parameters
for both splits. The implementation uses CPU FPS and caches the FPS indices in
each data-loader worker, so it does not require the PointNet++ CUDA extension:

```bash
python classification.py \
  --config configs/cls_scanobjectnn_pbt50rs.yaml \
  SOLVER.gpu 0, 
```

In `pointbert` mode, training performs FPS to 1200 candidate points followed by
random sampling to 1024 points, while testing directly performs FPS to 1024
points. The reported `test/accu` is overall classification accuracy, matching
the evaluation metric used by Point-BERT. ScanObjectNN contains XYZ only, so
these configs use `feature: P`, `channel: 3`, and a 15-class output head.

## 5. ShapeNet55 Supervised Pre-training

`configs/cls_shapenet55_pretrain.yaml` trains the same PointMamba backbone as
the ScanObjectNN configs with a temporary 55-class classification head. It
uses the official ShapeNet55 train/test lists, randomly samples 1024 points
from each 8192-point cloud, and applies the unit-sphere normalization used by
the Point-BERT ShapeNet55 loader.

Train with two GPUs:

```bash
python classification.py \
  --config configs/cls_shapenet55_pretrain.yaml \
  SOLVER.gpu 0,1,
```

The best complete 55-class model is saved to:

```text
logs/shapenet55/supervised_1024/best_model.pth
```

Use it to initialize a ScanObjectNN run by setting `SOLVER.pretrained`. Only
`model.backbone` is loaded; the incompatible 55-class head is discarded and
the ScanObjectNN 15-class head remains randomly initialized. Use a new log
directory so that an older fine-tuning checkpoint is not resumed by mistake:

```bash
python classification.py \
  --config configs/cls_scanobjectnn_pbt50rs.yaml \
  SOLVER.gpu 0,1, \
  SOLVER.pretrained logs/shapenet55/supervised_1024/best_model.pth \
  SOLVER.logdir logs/scanobjectnn/pbt50rs_ft_shapenet55_1024
```

The same `SOLVER.pretrained` option works with the OBJ_BG and OBJ_ONLY configs.
An existing `*.solver.tar` in the target log directory takes priority and
resumes the interrupted ScanObjectNN fine-tuning run, including its optimizer
and scheduler state.

## 6. ShapeNetPart Segmentation From Scratch

The ShapeNetPart loader reads
`data/shapenet_part_seg_hdf5_data/hdf5_data` directly. The benchmark config
trains on the official train+validation split (14007 shapes), tests on 2874
shapes, and keeps all 2048 HDF5 points without CUDA FPS.

Train with two GPUs and a global batch size of 16:

```bash
python segmentation.py \
  --config configs/seg_shapenetpart.yaml \
  SOLVER.gpu 0,1,
```

The four-stage `PointMambaSeg_base` uses channels `[96, 192, 384, 384]` and
blocks `[2, 2, 18, 2]`. Evaluation reports Point-BERT's ShapeNetPart metrics:
instance-average `test/mIoUI`, class-average `test/mIoUC`, and all 16 category
IoUs. The checkpoint selected by `mIoUI` is saved to
`logs/shapenetpart/from_scratch_2048/best_model.pth`.

The ShapeNetPart config keeps all periodic validation and ordinary
`SOLVER.run test` evaluations single-pass. After training completes it reloads
`best_model.pth` and performs the Pointcept v1.7 Utonia ten-vote TTA recipe:
identity, identity with `RandomFlip(p=0.5)`, and scales 0.8, 0.85, 0.9, 0.95,
1.05, 1.1, 1.15, and 1.2. Softmax probabilities are summed before computing
`test/mIoUI` and `test/mIoUC`; the separate result is appended to
`logs/shapenetpart/from_scratch_2048/final_tta_log.csv`.

To run only this final TTA for an existing best checkpoint:

```bash
python segmentation.py \
  --config configs/seg_shapenetpart.yaml \
  SOLVER.run test_tta \
  SOLVER.gpu 0,1, \
  SOLVER.ckpt logs/shapenetpart/from_scratch_2048/best_model.pth \
  SOLVER.logdir logs/shapenetpart/from_scratch_2048/tta_eval
```

Set `SOLVER.final_test_best False` to disable the automatic post-training TTA.

## 7. SUN RGB-D Detection (PointTTT + FCAF3D)

SUN RGB-D detection is an optional MMDetection3D entry and does not add any
imports to the existing classification or segmentation programs. It follows
the official OctFormer protocol: 10 classes, 100000 XYZRGB points, depth-12
octrees at 0.01 m, FCAF3D, an 18-epoch schedule, and evaluation at 3D IoU 0.25
and 0.50. The only replaced component is the four-stage PointTTT backbone with
channels `[96, 192, 384, 384]` and blocks `[2, 2, 18, 2]`.

Use the existing `mamba` environment. It already contains the compiled OCNN,
MMCV and MinkowskiEngine operators; the compatible mmsegmentation version is:

```bash
conda activate mamba
python -m pip install mmsegmentation==0.29.1
```

The launcher automatically uses the sibling OctFormer checkout at
`../octformer-master/mmdetection3d`. If it is elsewhere, export its root:

```bash
export MMDET3D_ROOT=/path/to/mmdetection3d
```

Before training, validate all 10335 point files and the official split:

```bash
python tools/check_sunrgbd.py --full
```

In the current data directory, `sunrgbd_infos_train.pkl` overlaps validation
by 2200 scenes. The config therefore reads the existing correct official split
from `sunrgbd_infos_train（复件）.pkl` (IDs 5051--10335) and leaves both files
untouched.

Train and validate on two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 \
torchrun --nproc_per_node=2 --master_port=29501 \
  detection.py configs/det_sunrgbd.py --launcher pytorch
```

The official micro-batch remains 8 scenes per GPU. With two GPUs,
`cumulative_iters=2` gives an effective batch of `8 x 2 x 2 = 32`, matching
OctFormer's four-GPU global batch while retaining its learning rate `1e-3`.
If a 24 GB GPU runs out of memory, use four scenes per GPU and accumulate four
iterations; the effective batch and learning rate remain unchanged:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 \
torchrun --nproc_per_node=2 --master_port=29501 \
  detection.py configs/det_sunrgbd.py --launcher pytorch \
  --cfg-options data.samples_per_gpu=4 \
  data.val.samples_per_gpu=4 data.test.samples_per_gpu=4 \
  optimizer_config.cumulative_iters=4
```

Checkpoints, copied configs, TensorBoard events and text logs are written to
`work_dirs/pointttt_sunrgbd/`. Validation runs every epoch and reports
per-class AP/recall plus the main `mAP_0.25` and `mAP_0.50` metrics.

Visualize the supplied checkpoints on six deterministic, class-covering
validation scenes (inference uses all 100000 points per scene):

```bash
conda activate mamba
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
python tools/visualize_sunrgbd_detection.py \
  --octformer work_dirs/pointttt_sunrgbd/epoch_13.pth \
  --3det-mamba work_dirs/pointttt_sunrgbd/epoch_12.pth \
  --pointttt work_dirs/pointttt_sunrgbd/epoch_18.pth
```

Each scene directory contains the four-panel `comparison.png`, individual
input/GT/prediction PNGs, numerical boxes and scores in NPZ, and editable PLY
point clouds/box edges. Results and checkpoint metadata are recorded under
`visual_results/sunrgbd_detection_comparison/`. Use `--indices 114 2307` for
explicit zero-based validation indices, `--show-labels` for class text, or
`--skip-existing` to resume without repeating completed inference.

For the same shaded-sphere appearance as the ShapeNetPart figures, render the
saved predictions with Mitsuba. This does not rerun detector inference:

```bash
conda activate mamba
python tools/render_sunrgbd_detection_mitsuba.py \
  --render-points 4096 --radius 0.005 --box-radius 0.003 \
  --width 1600 --height 1200 --spp 128 --dpi 1200
```

The four panels share the sampled points, normalization, camera, lighting and
crop. Points are rough-plastic spheres and box edges are solid cylinders on a
pure-white background. Mitsuba results are written to
`visual_results/sunrgbd_detection_comparison/mitsuba/`. The default scalar
variant is safe while both GPUs are training; use `--variant cuda_ad_rgb` only
when a GPU is free. Use `--scenes 2307_002308` for a single saved scene and
`--no-titles` for a title-free paper panel.

For dense scene-level figures matching the 3DET-Mamba qualitative layout, use
all 100000 RGB points with the headless Open3D splat renderer:

```bash
EGL_PLATFORM=surfaceless python tools/render_sunrgbd_detection_open3d.py \
  --scenes 2307_002308 1981_001982 0114_000115 0791_000792 \
  --max-points 0 --point-size 2.0 --line-width 3.0 \
  --width 1600 --height 1200 --dpi 1200 \
  --output-dir visual_results/sunrgbd_detection_comparison/open3d_dense_paper
```

This creates five rows (`Input`, three prediction sets and `GT`) with orange
prediction boxes and green GT boxes. All rows for a scene share the complete
point cloud, camera and foreground crop. The assembled figure is
`comparison_grid.png`.

## 8. PartNetE Part Segmentation

The loader uses the Pointcept-preprocessed `data/PartNetE/few_shot` and
`data/PartNetE/test` splits directly. It follows the Utonia protocol with 45
object categories, 148 category-specific part labels, 1 cm grid sampling,
CE+Lovasz loss, 800 epochs, and validation every 100 epochs. Train on two GPUs:

```bash
python segmentation.py \
  --config configs/seg_partnete.yaml \
  SOLVER.gpu 0,1,
```

The primary checkpoint metric is `test/mIoU_part`, computed over the 103 named
parts while excluding every category's `other` part. The best checkpoint is
saved to `logs/partnete/from_scratch_1cm/best_model.pth`. Training automatically
reloads it for the official ten-branch Utonia TTA. Run that final test alone:

```bash
python segmentation.py \
  --config configs/seg_partnete.yaml \
  SOLVER.run test_tta \
  SOLVER.gpu 0,1, \
  SOLVER.ckpt logs/partnete/from_scratch_1cm/best_model.pth \
  SOLVER.logdir logs/partnete/from_scratch_1cm/tta_eval
```

For an ordinary single-pass test, replace `SOLVER.run test_tta` with
`SOLVER.run test`. The latest epoch-boundary solver checkpoint is written to
`checkpoints/last.solver.tar` and is automatically used to resume an
interrupted run. The best validation checkpoint is saved separately as
`best_model.pth`.

## 9. Acknowledgement 
Our project is based on 
- Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba))
- Octformer([paper](https://arxiv.org/abs/2305.03045), [code](https://github.com/octree-nn/octformer))
- Vision Mamba([paper](https://arxiv.org/abs/2401.09417),[code](https://github.com/hustvl/Vim))
- Point Cloud Transformer([paper](https://arxiv.org/abs/2012.09688), [code](https://github.com/MenghaoGuo/PCT))

Thanks for their wonderful works!
