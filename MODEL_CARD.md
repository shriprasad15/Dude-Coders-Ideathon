# Model Card: Solar Panel Detection System

## Model Details
- **Architecture**: YOLOv12x (Fine-tuned)
- **Framework**: Ultralytics YOLO
- **Task**: Object Detection (Solar Panels)
- **Input**: Satellite chunks (Google Maps Static API)
- **Output**: Bounding boxes, confidence scores, and estimated area (sqm).
- **Date**: December 2025
- **Version**: 2.0 (Enhanced with Hard Negative Mining)

## Intended Use
- **Primary Use Case**: Automated detection of rooftop solar panels from satellite imagery to verify installation reports or estimate solar potential.
- **Target Users**: Energy auditors, grid operators, urban planners.
- **Out of Scope**: Ground-level imagery, thermal imagery analysis.

## Training Data
- **Dataset Size**: 2,649 images (augmented during training).
- **Source**: aggregated from multiple open-source solar panel datasets:
    1.  [Alfred Weber Institute - Custom Workflow](https://universe.roboflow.com/alfred-weber-institute-of-economics/custom-workflow-object-detection-tgnqc)
    2.  [ProjectSolarPanel - lsgi547-project](https://universe.roboflow.com/projectsolarpanel/lsgi547-project)
    3.  [Piscinas y Tenistable - Solar Panels](https://universe.roboflow.com/piscinas-y-tenistable/solar-panels-ba8ty)
    4.  Google Maps Static API (Satellite View) - Custom collected samples.
- **Preprocessing**: 
    - 1024x1024 resolution.
    - Data Augmentation: Mosaic, Mixup, HSV Color Jitter, Geometric transforms (Rotation, Scale, Shear).
    - **Hard Negative Mining**: Specific retraining on false positives/negatives from initial runs to reduce confusion with shiny rooftops, water tanks, and white concrete.

## Inference Logic (Multi-Stage Fallback)
The model uses a robust "Fallback Strategy" to maximize recall while maintaining precision:
1.  **Initial Scan**: Run inference on the full image (1024x1024).
2.  **Buffer Check (1200/2400 sqft)**: Check if a panel falls within specific buffer zones of the target coordinate.
3.  **Fallback 1 - Saturation**: If nothing found in buffers, enhance image saturation (to pop blue panels) and re-run.
4.  **Fallback 2 - Cropping**: If still nothing, crop the image to the buffer region (zoomed view) and re-run with lower thresholds.
5.  **Selection**: The "Best" panel is selected based on overlap with the buffer zone (Green Box), other candidates are marked as rejections (Red Boxes).

## Performance
- **Metrics**: Optimized for high Recall on small objects.
- **QC Status**: 
    - `VERIFIABLE`: High confidence detection (>0.7) OR Confirmed Absent (after exhaustive search).
    - `NOT_VERIFIABLE`: Low confidence detection (<=0.7).

## Limitations & Bias
- **Resolution Dependent**: Performance degrades significantly on low-resolution imagery (<20 zoom).
- **Cloud/Shadow**: Heavy cloud cover or deep shadows (skyscrapers) can occlude panels.
- **Geographic Bias**: Trained primarily on urban residential rooftops in [Target Region]; may generalize poorly to industrial solar farms or different architectural styles.
- **Look-alikes**: Blue water tanks and skylights remain the primary sources of False Positives.

## Retraining Guidance
To update the model:
1.  Add new samples to `EI_train_data.xlsx`.
2.  Run `train.py` (which implements the V2 training pipeline with hard negative mining).
3.  Replace `best.pt` with the new weights.
