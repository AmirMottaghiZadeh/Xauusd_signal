# OHLC Detection Project

Detectron2-based object detection pipeline for OHLC chart images, focused on extracting:
- trading signals (`Buy` / `Sell`)
- price-time ranges (zone detection on chart snapshots)

This repository is production-oriented for inference and result export (image overlays + CSV outputs).

## Table of Contents
- [1. Project Scope](#1-project-scope)
- [2. Training Ownership and Provenance](#2-training-ownership-and-provenance)
- [3. Model Architecture](#3-model-architecture)
- [4. Dataset and COCO Labeling](#4-dataset-and-coco-labeling)
- [5. Detailed Modeling Configuration](#5-detailed-modeling-configuration)
- [6. Important Training Notebook](#6-important-training-notebook)
- [7. Repository Structure](#7-repository-structure)
- [8. Setup](#8-setup)
- [9. Model Weights](#9-model-weights)
- [10. Inference Workflows](#10-inference-workflows)
- [11. Output Specifications](#11-output-specifications)
- [12. Sample Results](#12-sample-results)
- [13. Troubleshooting](#13-troubleshooting)
- [14. Notes and Limitations](#14-notes-and-limitations)

## 1. Project Scope

The project processes pre-rendered OHLC candle images and detects meaningful regions using trained object detectors.
Two inference pipelines are included:

1. Signal detection (`scripts/detectron2/show_trained_model.py`)
2. Range detection (`scripts/detectron2/run_range_detection_model.py`)

The range pipeline also maps detected pixel coordinates to real-world market values (time/price) using per-image CSV metadata.

## 2. Training Ownership and Provenance

The downloadable model weights in this repository are not generic third-party checkpoints.
They are based on custom training performed by the repository author on manually labeled OHLC chart data.

Provenance summary:
- Labels were created manually for chart patterns/signals.
- Annotation format follows COCO object-detection standard.
- Annotations were stored in JSON and registered in Detectron2.
- Training was performed with Faster R-CNN in Detectron2 using custom hyperparameters.
- Final weights/config were exported from the training workflow and are consumed by this repository.

## 3. Model Architecture

Both pipelines are based on Faster R-CNN in Detectron2.

Core architecture:
- `MODEL.META_ARCHITECTURE`: `GeneralizedRCNN`
- `MODEL.BACKBONE.NAME`: `build_resnet_fpn_backbone`
- Backbone depth: `ResNet-50`
- Feature pyramid: `FPN`
- Proposal module: `RPN`
- ROI head: `StandardROIHeads`

Relevant config files:
- `configs/detectron2/signal_detection/config_final.yaml`
- `configs/detectron2/range_trained/faster_rcnn_R_50_FPN_3x.yaml`
- `configs/detectron2/range_trained/Base-RCNN-FPN.yaml`

## 4. Dataset and COCO Labeling

Training/finetuning assumptions are aligned with COCO object detection annotation style.

COCO conventions used:
- Annotation structure: `images`, `annotations`, `categories`
- Bounding box convention: COCO style `[x, y, width, height]`
- Dataset split names used in config: `xau_train`, `xau_val`

Signal classes:
- `OB_BUY` (mapped in inference as Buy class)
- `OB_SELL` (mapped in inference as Sell class)

Observed training snapshot from notebook output:
- Total labeled samples: `733`
- Train/validation split: `586 / 147` (80/20)
- Class metadata: `['OB_BUY', 'OB_SELL']`

## 5. Detailed Modeling Configuration

This section documents concrete modeling values from the saved Detectron2 configs and training artifact outputs.

### 5.1 Signal Detection Model (`configs/detectron2/signal_detection/config_final.yaml`)

Model and head setup:
- `MODEL.META_ARCHITECTURE = GeneralizedRCNN`
- `MODEL.BACKBONE.NAME = build_resnet_fpn_backbone`
- `MODEL.RESNETS.DEPTH = 50`
- `MODEL.ROI_HEADS.NUM_CLASSES = 2`
- `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512`

Anchor and proposal setup:
- `MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256]]`
- `MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]`
- `MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000`
- `MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000`

Input and augmentation scale setup:
- `INPUT.MIN_SIZE_TRAIN = [400, 480, 512, 544, 576, 608, 640, 672, 704]`
- `INPUT.MAX_SIZE_TRAIN = 1333`
- `INPUT.MIN_SIZE_TEST = 640`
- `INPUT.MAX_SIZE_TEST = 1333`

Solver and schedule:
- `SOLVER.IMS_PER_BATCH = 4`
- `SOLVER.BASE_LR = 0.00025`
- `SOLVER.MAX_ITER = 12000`
- `SOLVER.STEPS = [8000, 10000]`
- `SOLVER.WARMUP_ITERS = 1000`
- `TEST.EVAL_PERIOD = 500`

Validation snapshot (from notebook logs):
- Best logged checkpoint event: `AP = 30.9933` at iteration `6000`
- Final printed bbox metrics:
  - `AP = 28.8633`
  - `AP50 = 45.4790`
  - `AP75 = 31.6035`
  - `AP-OB_BUY = 26.3022`
  - `AP-OB_SELL = 31.4244`

### 5.2 Range Detection Model (`configs/detectron2/range_trained/*`)

Base config (`Base-RCNN-FPN.yaml`):
- `MODEL.META_ARCHITECTURE = GeneralizedRCNN`
- `MODEL.BACKBONE.NAME = build_resnet_fpn_backbone`
- `MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]`
- `MODEL.ROI_HEADS.NAME = StandardROIHeads`
- `MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]`
- `MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]`
- `MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]`
- `SOLVER.BASE_LR = 0.02`
- `SOLVER.MAX_ITER = 90000`
- `SOLVER.STEPS = (60000, 80000)`

Top-level override (`faster_rcnn_R_50_FPN_3x.yaml`):
- `_BASE_ = Base-RCNN-FPN.yaml`
- `MODEL.WEIGHTS = detectron2://ImageNetPretrained/MSRA/R-50.pkl`
- `MODEL.RESNETS.DEPTH = 50`
- `SOLVER.MAX_ITER = 270000`
- `SOLVER.STEPS = (210000, 250000)`

Inference-time behavior (`scripts/detectron2/run_range_detection_model.py`):
- `cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1` is set explicitly for deployed range inference.

## 6. Important Training Notebook

`new_setup_opencv (1).ipynb` is a high-value project artifact and should be preserved.

Why this notebook is important:
- It contains the end-to-end custom training pipeline used for the signal-detection model.
- It registers COCO-style annotations with `register_coco_instances`.
- It creates train/val splits (`xau_train`, `xau_val`).
- It configures Faster R-CNN hyperparameters and exports config snapshots.
- It includes a custom trainer (`EarlyStoppingTrainer`) with periodic COCO evaluation.
- It exports model artifacts:
  - `model_final.pth`
  - `config_final.yaml`
- It demonstrates post-training inference and visualization on chart images.

This notebook is direct technical evidence of custom model development and training work, not a superficial inference-only wrapper.

## 7. Repository Structure

| Path | Purpose |
|---|---|
| `configs/detectron2/` | Model configuration files (Faster R-CNN variants) |
| `configs/model_sources.json` | Download sources for `.pth` model weights |
| `data/external/image_input/<year>/` | Input OHLC chart images |
| `data/external/csv_input/<year>/` | Per-image metadata CSVs (used by range pipeline) |
| `data/metadata/predictions/` | Generated predictions (ignored in git) |
| `docs/assets/samples/` | Versioned sample images used in README |
| `models/detectron2/` | Local model destination (weights are not versioned) |
| `scripts/detectron2/` | Inference scripts |
| `scripts/download_models.py` | Model weight downloader |
| `new_setup_opencv (1).ipynb` | Colab training notebook for signal model |

## 8. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_models.py
```

## 9. Model Weights

Model files are intentionally excluded from git history.
They are downloaded using `configs/model_sources.json`.

Configured public sources:
- Range-trained model: https://drive.google.com/file/d/18-NDOWh_VPnmVonfx5rIIREmdwwLI8pS/view?usp=drive_link
- Signal detection model: https://drive.google.com/file/d/1ec8zqmd0eWHFjPiVbmTlF69QoeArulZR/view?usp=drive_link

Download command:

```bash
python3 scripts/download_models.py
```

Optional:
- `--force` to redownload existing files
- `--timeout <seconds>` to adjust timeout

## 10. Inference Workflows

### 10.1 Signal Detection

```bash
python3 scripts/detectron2/show_trained_model.py --no-display
```

Common overrides:
- `--image-dir data/external/image_input/2025`
- `--output-dir data/metadata/predictions/signal_detection`
- `--config configs/detectron2/signal_detection/config_final.yaml`
- `--weights models/detectron2/signal_detection/setup_detection.pth`
- `--score-thresh 0.5`
- `--device cpu` or `--device cuda`

### 10.2 Range Detection

```bash
python3 scripts/detectron2/run_range_detection_model.py --no-display
```

Common overrides:
- `--image-dir data/external/image_input/2025`
- `--csv-dir data/external/csv_input/2025`
- `--output-dir data/metadata/predictions/range_trained`
- `--config configs/detectron2/range_trained/faster_rcnn_R_50_FPN_3x.yaml`
- `--weights models/detectron2/range_trained/model_final.pth`
- `--score-thresh 0.2`
- `--device cpu` or `--device cuda`

## 11. Output Specifications

### Signal Detection Outputs

- Annotated images in the selected output directory
- Detection CSV files under `<output-dir>/csv/`
- CSV columns:
  - `x_min`, `y_min`, `x_max`, `y_max`, `score`, `class_id`

### Range Detection Outputs

- Annotated images in the selected output directory
- Per-image detection CSV files in the same directory
- CSV columns:
  - `x_min`, `x_max`, `y_min`, `y_max`
  - `time_start`, `time_end`
  - `price_high`, `price_low`

`time_*` and `price_*` values are reconstructed from pixel coordinates using metadata in `data/external/csv_input/<year>/`.

## 12. Sample Results

Sample images are stored in `docs/assets/samples` to keep README links stable after push.

### Signal Detection

![signal detection sample 1](docs/assets/samples/signal_detection/candle_2025-01-02_part_1.png)
![signal detection sample 2](docs/assets/samples/signal_detection/candle_2025-01-02_part_10.png)
![signal detection sample 3](docs/assets/samples/signal_detection/candle_2025-01-02_part_11.png)
![signal detection sample 4](docs/assets/samples/signal_detection/candle_2025-01-02_part_12.png)
![signal detection sample 5](docs/assets/samples/signal_detection/candle_2025-01-02_part_13.png)

### Range-Trained Detection

![range trained sample 1](docs/assets/samples/range_trained/candle_2025-01-02_part_1.png)
![range trained sample 2](docs/assets/samples/range_trained/candle_2025-01-02_part_2.png)
![range trained sample 3](docs/assets/samples/range_trained/candle_2025-01-02_part_3.png)
![range trained sample 4](docs/assets/samples/range_trained/candle_2025-01-02_part_4.png)
![range trained sample 5](docs/assets/samples/range_trained/candle_2025-01-02_part_5.png)

## 13. Troubleshooting

- `Weights file not found`:
  - Run `python3 scripts/download_models.py`.
- Google Drive URL issues:
  - Keep model links in `configs/model_sources.json`.
  - Shared Google Drive links are normalized by the downloader.
- GUI display errors (`cv2.imshow` / headless server):
  - Use `--no-display`.
- Empty predictions:
  - Lower `--score-thresh` and verify the selected model/config pair.
- Missing metadata for range detection:
  - Ensure each image has a matching CSV with the same stem in `--csv-dir`.

## 14. Notes and Limitations

- `data/metadata/predictions/` is ignored in git (generated outputs).
- `detectron2/` directory is ignored at project root to avoid nested-git/submodule conflicts.
- Large `.pth` files in `models/detectron2/` are intentionally not versioned.
