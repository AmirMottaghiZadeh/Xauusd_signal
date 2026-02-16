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

## Signal labels and confidence
- `signal = 0`: Buy
- `signal = 1`: Sell
- The percentage shown on each image is the model confidence (probability) for the predicted signal, interpreted as the estimated signal correctness.

## Sample output images
Sample images below are stored under `docs/assets/samples` so links stay valid after push.

### Signal detection (`docs/assets/samples/signal_detection`)
![signal detection sample 1](docs/assets/samples/signal_detection/candle_2025-01-02_part_1.png)
![signal detection sample 2](docs/assets/samples/signal_detection/candle_2025-01-02_part_11.png)
![signal detection sample 3](docs/assets/samples/signal_detection/candle_2025-01-02_part_12.png)
![signal detection sample 4](docs/assets/samples/signal_detection/candle_2025-01-02_part_13.png)

### Range-trained detection (`docs/assets/samples/range_trained`)
![range trained sample 1](docs/assets/samples/range_trained/candle_2025-01-02_part_1.png)
![range trained sample 2](docs/assets/samples/range_trained/candle_2025-01-02_part_2.png)
![range trained sample 3](docs/assets/samples/range_trained/candle_2025-01-02_part_3.png)
![range trained sample 4](docs/assets/samples/range_trained/candle_2025-01-02_part_5.png)

## Model download setup
1. Model URLs are preconfigured in `configs/model_sources.json` (Google Drive links).
2. You can use either a shared Drive link (`.../file/d/<id>/view`) or a direct URL; the downloader normalizes Drive links automatically.
3. Run:
```bash
python scripts/download_models.py
```

Configured sources:
- Range-trained model: https://drive.google.com/file/d/18-NDOWh_VPnmVonfx5rIIREmdwwLI8pS/view?usp=drive_link
- Signal detection model: https://drive.google.com/file/d/1ec8zqmd0eWHFjPiVbmTlF69QoeArulZR/view?usp=drive_link

## Publishing notes
- `data/external` images and CSV files are intentionally versioned in this repo.
- Local cloned source folder `detectron2/` is ignored in the root repo to avoid embedded-git/submodule issues.
- Heavy `.pth` files under `models/detectron2` are ignored to keep the git history lightweight.
- `data/metadata/predictions` is ignored (generated output), so README images are kept in `docs/assets/samples`.
