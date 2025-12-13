# Solar Panel Detection System

## Overview
This repository contains an end-to-end pipeline for detecting solar panels in satellite imagery. It uses a fine-tuned YOLOv12 model with a specialized multi-stage fallback inference strategy to handle difficult cases (small panels, low contrast).

![Audit Example](outputs/audit_images/1.0.jpg)
*(Green: Selected Panel, Red: Rejected Candidates)*

## Deliverables Structure
```
solar_project_package/
├── inference.py          # Main inference pipeline code
├── train.py              # Training script (V2 with hard negative mining)
├── utils.py              # Geometry and visualization utilities
├── image_retriever.py    # Google Maps Image Downloader
├── requirements.txt      # Python dependencies
├── run_inference.sh      # AUTOMATION SCRIPT (Setup + Run)
├── best.pt               # Trained Model Weights
├── EI_train_data.xlsx    # Input Dataset
├── .env                  # API Keys (Google Maps)
├── MODEL_CARD.md         # Detailed Model Documentation
└── training_logs/        # Logs from model training
```

## Setup & Installation

**Prerequisite**: Linux/MacOS with Python 3.8+.

The project includes an **automation script** `run_inference.sh` that handles:
1.  Virtual Environment creation (`venv`)
2.  Dependency installation
3.  Execution

### 1. Configure API Key
Ensure the `.env` file exists in the root directory and contains your Google Maps API Key:
```bash
GOOGLE_MAPS_API_KEY=your_key_here
```

### 2. Run Inference
To run the model on the data in `EI_train_data.xlsx`:

```bash
# Run on the first 10 samples (Auto-install dependencies if needed)
./run_inference.sh --limit 10
```

To run on specific Sample IDs:
```bash
./run_inference.sh --samples "1-5,20,55-60"
```

## Output
Results are generated in the `outputs/` folder:
-   **`results.json`**: Structural JSON output containing detection status, confidence, estimated area, and metadata.
-   **`audit_images/`**: Visual audit frames showing the detected panels (Green) and rejected candidates (Red).

## QC Status Logic
-   **VERIFIABLE**: 
    -   Solar Panel Found with **High Confidence (> 70%)**.
    -   **OR** Solar Panel **Not Found** after exhaustive search (Verified Absent).
-   **NOT_VERIFIABLE**:
    -   Solar Panel Found but with **Low Confidence (<= 70%)**.

## Technical Methodology
### 1. Multi-Stage Fallback Strategy
To address challenges with small panels and low-contrast satellite imagery, we implemented a robust **6-stage fallback mechanism**. If a panel is not detected in the initial pass, the system progressively relaxes constraints and applies enhancements:

1.  **Initial Check (1200 sqft)**: Check for panels within a small 1200 sqft buffer (approx. residential roof size) from the standard inference.
2.  **Saturated Check (1200 sqft)**: If failed, saturate the image (HSV +50%) to boost contrast and re-run inference.
3.  **Cropped Check (1200 sqft)**: If failed, physically crop the image to the 1200 sqft region and re-run inference (improves relative object size).
4.  **Saturated Crop Check (1200 sqft)**: Combine cropping and saturation.
5.  **Expanded Check (2400 sqft)**: If still failed, repeat the initial check on a larger 2400 sqft buffer.
6.  **Expanded Saturated Check (2400 sqft)**: Final attempt using saturation on the larger buffer.

### 2. Advanced Training Strategy
-   **Hard Negative Sampling**: We explicitly trained the model on "negative" samples—satellite images of rooftops *without* solar panels—to drastically reduce false positives. This forces the model to learn the specific features of solar panels rather than just "rectangular things on roofs."
-   **Data Augmentation**: Heavy use of Mosaic, Mixup, and HSV augmentation during training to generalize across different lighting conditions and resolutions.

## Model Performance
The model utilizes **YOLOv12** architecture fine-tuned on our custom dataset.

![Validation Batch Predictions](training_logs/val_batch0_pred.jpg)
*(Validation Batch Predictions: Ground Truth vs Model Output)*

## Training
To retrain the model:
1.  Ensure data is prepared in `data_segmented`.
2.  Run valid training script:
    ```bash
    python train.py
    ```
3.  Logs will be saved to `outputs/training/yolov12_v2`.
