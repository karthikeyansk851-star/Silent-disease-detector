# Silent Disease Detector

Low-resource early detection of silent diseases (skin lesions, oral cancer,
diabetic retinopathy) via smartphone imaging. Offline-capable, privacy-preserving,
explainable.

## Phase 1 target: Skin lesion classification (HAM10000)

## Project layout

```
silent-disease-detector/
├── data/
│   ├── raw/            # untouched downloaded datasets
│   └── processed/      # resized/cleaned images + labels.csv
├── models/              # saved checkpoints (.pt) and exported (.tflite)
├── src/
│   ├── config.py        # paths, hyperparameters, class names
│   ├── dataset.py        # PyTorch Dataset + DataLoader
│   ├── augmentations.py  # low-quality-camera simulation transforms
│   ├── train.py          # training loop (EfficientNet-B0 transfer learning)
│   ├── gradcam.py        # Grad-CAM heatmap generation
│   ├── quantize.py        # export -> ONNX -> TFLite (int8)
│   └── infer.py          # single-image inference + Grad-CAM overlay
├── app/
│   └── streamlit_app.py  # upload image -> prediction + heatmap demo
├── scripts/
│   └── download_data.sh   # Kaggle download helper
├── notebooks/             # exploration / EDA
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

HAM10000 ("Skin Cancer MNIST") on Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

1. Download `kaggle.json` API token from your Kaggle account settings.
2. Place it at `~/.kaggle/kaggle.json`.
3. Run:
   ```bash
   bash scripts/download_data.sh
   ```

This drops images into `data/raw/HAM10000_images/` and metadata into
`data/raw/HAM10000_metadata.csv`.

## Train

```bash
python -m src.train
```

Checkpoints save to `models/best_model.pt`.

## Export to TFLite (offline, on-device)

```bash
python -m src.quantize
```

Produces `models/model_int8.tflite`.

## Run the demo app

```bash
streamlit run app/streamlit_app.py
```

## Roadmap

- [x] Phase 0: Scaffolding
- [ ] Phase 1: Skin lesion model + Grad-CAM + TFLite + Streamlit demo
- [ ] Phase 2: Oral cancer model (reuse pipeline)
- [ ] Phase 3: Diabetic retinopathy model (APTOS 2019)
- [ ] Phase 4: Unified multi-disease app
