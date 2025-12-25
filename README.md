# Solar Panel Detection System

## Overview
This repository contains an end-to-end pipeline for detecting solar panels in satellite imagery. It uses a fine-tuned YOLOv12 model with a specialized multi-stage fallback inference strategy to handle difficult cases (small panels, low contrast).

![Audit Example](outputs/audit_images/1.0.jpg)


*(Green: Selected Panel, Red: Rejected Candidates)*

## Deliverables Structure
```
Dude-Coders-Ideathon/
├── inference.py          # Main inference pipeline code
├── train.py              # Training script (V2 with hard negative mining)
├── utils.py              # Geometry and visualization utilities
├── image_retriever.py    # Google Maps Image Downloader
├── evaluate_results.py   # Script to evaluate inference against ground truth
├── requirements.txt      # Python dependencies
├── python_version.txt    # Python version information
├── run_inference.sh      # AUTOMATION SCRIPT (Setup + Run)
├── best.pt               # Trained Model Weights
├── EI_train_data.xlsx    # Input Dataset
├── .env                  # API Keys (Google Maps)
├── MODEL_CARD.md         # Detailed Model Documentation
├── Model_Card.pdf        # PDF version of Model Card
├── LICENSE               # License file
├── training_logs/        # Logs from model training
└── outputs/              # Directory for inference results and audit images
    ├── results.json      # Structural JSON output with detection details
    └── audit_images/     # Visual audit frames (Green: Accepted, Red: Rejected)
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
# Run on all samples (Auto-install dependencies if needed)
./run_inference.sh
```

### Supported Arguments
The `run_inference.sh` script passes all arguments directly to the python inference engine.

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--limit` | Process only the first N samples | `./run_inference.sh --limit 10` |
| `--samples` | Process specific Sample IDs or ranges | `./run_inference.sh --samples "1-5,20,55-60"` |
| `--input` | Use a custom Excel input file | `./run_inference.sh --input data/my_sites.xlsx` |
| `--initial-conf` | Set initial detection confidence threshold | `./run_inference.sh --initial-conf 0.25` |
| `--fallback-conf` | Set fallback detection confidence threshold | `./run_inference.sh --fallback-conf 0.10` |

### Custom Input Requirements
When using `--input`, your Excel file **must** contain the following columns:
- `sample_id` (or `sampleid`)
- `latitude`
- `longitude`

**Note**: The default input file is `EI_train_data.xlsx`.

To run on a custom Excel input file (must have `sample_id`, `latitude`, `longitude`):
```bash
./run_inference.sh --input path/to/your_data.xlsx
```

## Output
Results are generated in the `outputs/` folder:
-   **`results.json`**: Structural JSON output containing detection status, confidence, estimated area, distance, and metadata.
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


## Training
To retrain the model:
1.  Ensure data is prepared in `data_segmented`.
2.  Run valid training script:
    ```bash
    python train.py
    ```
3.  Logs will be saved to `outputs/training/yolov12_v2`.

## Web Application (Smart Search)
The project includes a React-based frontend and FastAPI backend for the "Smart Search" feature (Solar Analysis Chat).

### 1. Start Backend
Run this in a separate terminal from the root directory:
```bash
# Activate Virtual Environment
source venv/bin/activate

# Start the API Server
python -m backend.main
```
*Server runs at: http://localhost:8000*

### 2. Start Frontend
Run this in a separate terminal:
```bash
cd frontend

# Start the Dev Server
./npm_portable.sh run dev
```
*App runs at: http://localhost:5173*
