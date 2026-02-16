# OHLC Detection Project

Portfolio project for detecting zones/signals on OHLC chart images using OpenCV + Detectron2.

## Repository layout
- `data/external/image_input/<year>`: chart images (`.png`) used for inference.
- `data/external/csv_input/<year>`: CSV metadata per image.
- `data/metadata/predictions`: generated prediction CSV/image outputs.
- `configs/detectron2`: model config files.
- `models/detectron2`: local model destination path (weights are downloaded, not versioned in git).
- `scripts/detectron2`: inference runners.
- `scripts/download_models.py`: download helper for model weights.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
```

## Run inference
```bash
python scripts/detectron2/show_trained_model.py --no-display
python scripts/detectron2/run_range_detection_model.py --no-display
```

You can override paths with CLI args:
- `--image-dir data/external/image_input/2025`
- `--csv-dir data/external/csv_input/2025`
- `--weights models/detectron2/...`
- `--config configs/detectron2/...`

## Model download setup
1. Upload the model files to your preferred storage (GitHub Releases, Hugging Face, S3, etc.).
2. Put direct download URLs (and optional sha256) into `configs/model_sources.json`.
3. Run:
```bash
python scripts/download_models.py
```

## Publishing notes
- `data/external` images and CSV files are intentionally versioned in this repo.
- Local cloned source folder `detectron2/` is ignored in the root repo to avoid embedded-git/submodule issues.
- Heavy `.pth` files under `models/detectron2` are ignored to keep the git history lightweight.
