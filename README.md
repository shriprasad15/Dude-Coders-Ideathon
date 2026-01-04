# Solar Panel Detection System

## Overview
This repository contains an end-to-end pipeline for detecting solar panels in satellite imagery. It uses a fine-tuned YOLOv12 model with a specialized multi-stage fallback inference strategy to handle difficult cases (small panels, low contrast).

![Audit Example](artefacts/train/1.jpg)

*(Green: Selected Panel, Red: Rejected Candidates)*

## Project Structure
```
├── pipeline_code/           # Core inference and training scripts
│   ├── inference.py         # Main inference pipeline
│   ├── train.py             # Training script with hard negative mining
│   ├── utils.py             # Geometry and visualization utilities
│   ├── image_retriever.py   # Google Maps Image Downloader
│   └── evaluate_results.py  # Evaluation script
├── environment_details/     # Environment configuration
│   ├── requirements.txt     # Python dependencies
│   ├── python_version.txt   # Python version information
│   └── environment.yml      # Conda environment file
├── trained_model/           # Model weights
│   └── model.pt             # Trained YOLOv12 weights
├── model_card/              # Model documentation
│   ├── Model_Card.pdf       # Model Card (PDF)
│   └── Model_card.tex       # Model Card (LaTeX source)
├── prediction_files/        # Inference output JSON files
│   ├── train/               # Training set predictions
│   └── test/                # Test set predictions
├── artefacts/               # Visual audit images
│   ├── train/               # Training set visualizations
│   └── test/                # Test set visualizations
├── training_logs/           # Training metrics and logs
│   └── logs.csv             # Training log CSV
├── backend/                 # FastAPI backend for web app
├── frontend/                # React frontend for web app
├── run_inference.sh         # Automation script (Setup + Run)
├── EI_train_data.xlsx       # Input dataset
├── .env                     # API Keys (Google Maps)
└── README.md                # This file
```

## Quick Start

### 1. Configure API Key
Create a `.env` file in the root directory:
```bash
GOOGLE_MAPS_API_KEY=your_key_here
```

### 2. Run Inference (Automated)
The `run_inference.sh` script handles virtual environment creation, dependency installation, and execution:

```bash
# Run on all samples
./run_inference.sh

# Run on specific samples
./run_inference.sh --samples "1-10,20,55-60"

# Run on first 10 samples only
./run_inference.sh --limit 10

# Use custom input file
./run_inference.sh --input path/to/your_data.xlsx --output path/to/your_output_dir
```

### 3. Run Inference (Manual)
```bash
# Activate virtual environment
source venv/bin/activate

# Run inference script directly
python pipeline_code/inference.py \
    --input EI_train_data.xlsx \
    --model trained_model/model.pt \
    --samples "1-5"
```

## Command Line Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--input` | Input Excel file path | `--input data/sites.xlsx` |
| `--model` | Model weights path | `--model trained_model/model.pt` |
| `--limit` | Process only first N samples | `--limit 10` |
| `--samples` | Specific Sample IDs or ranges | `--samples "1-5,20,55-60"` |
| `--initial-conf` | Initial confidence threshold | `--initial-conf 0.20` |
| `--fallback-conf` | Fallback confidence threshold | `--fallback-conf 0.15` |
| `--output` | Output directory for results JSON | `--output path/to/your_output_dir` |

Default output directory is `artefacts/test`

### Custom Input Requirements
Your Excel file **must** contain these columns:
- `sample_id` (or `sampleid`)
- `latitude`
- `longitude`

## Output
Results are generated in the `outputs/` folder:
- **`results.json`**: JSON with detection status, confidence, estimated area, and metadata
- **`audit_images/`**: Visual audit frames (Green: Accepted, Red: Rejected)

## Technical Methodology

### Multi-Stage Fallback Strategy
To handle small panels and low-contrast imagery, we use a 6-stage fallback mechanism:

1. **Initial Check (1200 sqft)**: Standard inference on small buffer
2. **Saturated Check (1200 sqft)**: HSV saturation boost (+50%)
3. **Cropped Check (1200 sqft)**: Crop to buffer region, re-run inference
4. **Saturated Crop Check (1200 sqft)**: Combine cropping and saturation
5. **Expanded Check (2400 sqft)**: Initial check on larger buffer
6. **Expanded Saturated Check (2400 sqft)**: Saturation on larger buffer

### QC Status Logic
- **VERIFIABLE**: High confidence (>70%) detection OR verified absent
- **NOT_VERIFIABLE**: Low confidence (≤70%) detection

## Training
```bash
# Activate environment
source venv/bin/activate

# Run training
python pipeline_code/train.py
```

## Web Application

### Start Backend
```bash
source venv/bin/activate
python -m backend.main
```
*Server runs at: http://localhost:8000*

### Start Frontend
```bash
cd frontend
npm run dev
```
*App runs at: http://localhost:5173*

## License
See [LICENSE](LICENSE) for details.
