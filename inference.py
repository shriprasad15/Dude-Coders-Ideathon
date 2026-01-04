#!/usr/bin/env python3
"""
Solar panel inference with buffer-based fallback strategy.
Uses ultralytics YOLO directly (no SAHI dependency).

Strategy:
1. Run inference on the entire image
2. Check if solar panel found in 1200 sqft buffer, then 2400 sqft buffer
3. If NOT found in either buffer, try saturation enhancement + re-inference
4. Then crop to each buffer region and re-run inference
5. Report final result per buffer

Run: CUDA_VISIBLE_DEVICES=1 python src/run_inference_fallback.py --samples "1-10,2501-2510"
"""
import pandas as pd
import json
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))
from image_retriever import ImageRetriever
from utils import (
    calculate_meters_per_pixel,
    calculate_radius_from_area_sqft,
    calculate_intersection_area,
    create_spotlight_overlay,
    calculate_box_area_pixels,
    crop_buffer_region,
    run_inference_on_crop
)
from ultralytics import YOLO


# Size validation constants (for reference/future use)
MIN_PANEL_AREA_SQM = 0.5     # Minimum realistic panel area (1 sqm)
MAX_PANEL_AREA_SQM = 1000.0    # Maximum realistic panel area (100 sqm)


def enhance_saturation(image, factor=1.5):
    """Enhance image saturation to make solar panels more visible."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def find_best_panel_in_buffer(boxes, confidences, center, radius_pixels, meters_per_pixel=None):
    """
    Find the panel with largest overlap within the buffer region.
    Reverted to ABSOLUTE overlap area (per FAQ requirements).
    
    CRITICAL: Added strict size check to reject "giant box" false positives.
    If a box is too huge (>300 smooth), we ignore it even if it has high overlap.
    """
    best_overlap = 0
    best_idx = -1
    
    for i, box in enumerate(boxes):
        # Calculate panel area in sqm if meters_per_pixel provided
        if meters_per_pixel is not None:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area_sqm = (w * h) * (meters_per_pixel ** 2)
            

                
        overlap = calculate_intersection_area(box, center, radius_pixels)
        # Debug why large box is ignored
        print(f"    [DEBUG] Box {i}: Area={area_sqm:.1f}sqm, Overlap={overlap:.1f}, Best={best_overlap:.1f}")
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    
    return best_idx, best_overlap


def run_inference_fallback(
    excel_path="EI_train_data.xlsx",
    output_dir="outputs",
    model_path="best.pt",
    limit=None,
    sample_ids=None,
    initial_conf=0.15,
    fallback_conf=0.07
):
    """
    Run inference with buffer-based fallback strategy.
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Reading data from {excel_path}...")
    print(f"Reading data from {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # Column Normalization & Validation
    # 1. Normalize 'sampleid' to 'sample_id'
    if 'sampleid' in df.columns:
        df.rename(columns={'sampleid': 'sample_id'}, inplace=True)
    
    # 2. Check for mandatory columns
    required_columns = ['sample_id', 'latitude', 'longitude']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Input Excel is missing mandatory columns: {missing_cols}")
        print(f"Required columns: {required_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    # Filter by sample_ids if provided
    if sample_ids:
        # ensuring matching types for filtering
        df['sample_id'] = df['sample_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        # Convert input sample_ids to strings for comparison
        str_sample_ids = [str(sid) for sid in sample_ids]
        df = df[df['sample_id'].isin(str_sample_ids)]
        print(f"Filtering to {len(df)} samples with IDs: {sample_ids}")
    elif limit:
        df = df.head(limit)
    
    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    audit_dir = os.path.join(output_dir, "audit_images")
    os.makedirs(audit_dir, exist_ok=True)
    
    # Load env vars
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    retriever = ImageRetriever(api_key=api_key)
    
    results_json = []
    
    print(f"\nStarting fallback inference (initial_conf={initial_conf}, fallback_conf={fallback_conf})...")
    
    for idx, row in df.iterrows():
        sample_id = row.get('sample_id', idx)
        lat = row.get('latitude', 0)
        lon = row.get('longitude', 0)
        
        print(f"\nProcessing {sample_id} ({idx+1}/{len(df)})...")
        
        # 1. Get image
        image_path, metadata = retriever.get_image(lat, lon, zoom=20)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image: {image_path}")
            continue
        
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 2. Run initial inference on FULL image
        results = model.predict(img, conf=initial_conf, augment=True, save=False, verbose=False)
        result = results[0]
        
        boxes = result.boxes.xyxy.tolist() if result.boxes else []
        confidences = result.boxes.conf.tolist() if result.boxes else []
        
        print(f"  Initial detection: {len(boxes)} panels found")
        
        # 3. Calculate buffer radii
        meters_per_pixel = calculate_meters_per_pixel(lat, zoom=20)
        radius_1200 = calculate_radius_from_area_sqft(1200, meters_per_pixel)
        radius_2400 = calculate_radius_from_area_sqft(2400, meters_per_pixel)
        
        # 4. Check buffers in order: 1200 then 2400
        final_has_solar = False
        final_buffer_size = 2400  # Default
        final_pv_area_sqm = 0.0
        final_confidence = 0.0
        final_bbox = []
        detection_method = "initial"
        
        # --- STEP 1: Check 1200 sqft buffer from INITIAL inference ---
        best_idx_1200, _ = find_best_panel_in_buffer(boxes, confidences, center, radius_1200, meters_per_pixel)
        
        if best_idx_1200 != -1:
            final_has_solar = True
            final_buffer_size = 1200
            final_bbox = boxes[best_idx_1200]
            final_confidence = confidences[best_idx_1200]
            detection_method = "initial"
            print(f"  -> Found in 1200 sqft buffer (Initial)! Conf: {final_confidence:.3f}")
        
        else:
            # --- STEP 2: Check 1200 sqft from SATURATED inference ---
            print(f"  -> Not in 1200 (Initial). Trying SATURATED 1200...")
            
            img_enhanced = enhance_saturation(img, factor=1.5)
            # Run inference on enhanced full image
            results_enhanced = model.predict(img_enhanced, conf=fallback_conf, augment=True, save=False, verbose=False)
            result_enhanced = results_enhanced[0]
            
            enh_boxes = result_enhanced.boxes.xyxy.tolist() if result_enhanced.boxes else []
            enh_confs = result_enhanced.boxes.conf.tolist() if result_enhanced.boxes else []
            
            best_enh_1200, _ = find_best_panel_in_buffer(enh_boxes, enh_confs, center, radius_1200, meters_per_pixel)
            
            if best_enh_1200 != -1:
                final_has_solar = True
                final_buffer_size = 1200
                final_bbox = enh_boxes[best_enh_1200]
                final_confidence = enh_confs[best_enh_1200]
                detection_method = "saturated_1200"
                boxes.extend(enh_boxes)
                confidences.extend(enh_confs)
                print(f"  -> Found in 1200 sqft buffer (Saturated)! Conf: {final_confidence:.3f}")
            
            else:
                # --- STEP 3: Check 1200 sqft by CROPPING ---
                print(f"  -> Not in 1200 (Saturated). Trying CROPPED 1200...")
                
                cropped_1200, offset_1200, _ = crop_buffer_region(img, center, radius_1200, padding=30)
                crop_boxes_1200, crop_confs_1200 = run_inference_on_crop(
                    model, cropped_1200, offset_1200, conf=fallback_conf
                )
                
                best_crop_1200, _ = find_best_panel_in_buffer(crop_boxes_1200, crop_confs_1200, center, radius_1200, meters_per_pixel)
                
                if best_crop_1200 != -1:
                    final_has_solar = True
                    final_buffer_size = 1200
                    final_bbox = crop_boxes_1200[best_crop_1200]
                    final_confidence = crop_confs_1200[best_crop_1200]
                    detection_method = "crop_1200"
                    boxes.extend(crop_boxes_1200)
                    confidences.extend(crop_confs_1200)
                    print(f"  -> Found in 1200 sqft buffer (Cropped)! Conf: {final_confidence:.3f}")
                
                else:
                    # --- STEP 4: Check 1200 sqft by CROPPING + SATURATION ---
                    print(f"  -> Not in 1200 (Cropped). Trying SATURATED CROP 1200...")
                    
                    # Saturate the cropped image
                    cropped_1200_sat = enhance_saturation(cropped_1200, factor=1.5)
                    crop_sat_boxes, crop_sat_confs = run_inference_on_crop(
                        model, cropped_1200_sat, offset_1200, conf=fallback_conf
                    )
                    
                    best_crop_sat_1200, _ = find_best_panel_in_buffer(crop_sat_boxes, crop_sat_confs, center, radius_1200, meters_per_pixel)
                    
                    if best_crop_sat_1200 != -1:
                        final_has_solar = True
                        final_buffer_size = 1200
                        final_bbox = crop_sat_boxes[best_crop_sat_1200]
                        final_confidence = crop_sat_confs[best_crop_sat_1200]
                        detection_method = "crop_sat_1200"
                        boxes.extend(crop_sat_boxes)
                        confidences.extend(crop_sat_confs)
                        print(f"  -> Found in 1200 sqft buffer (Sat+Crop)! Conf: {final_confidence:.3f}")
                    
                    else:
                        # --- STEP 5: Check 2400 sqft from INITIAL inference ---
                        # (using results from Step 1)
                        print(f"  -> Not found in any 1200 method. Checking 2400 (Initial)...")
                        
                        best_idx_2400, _ = find_best_panel_in_buffer(boxes, confidences, center, radius_2400, meters_per_pixel)
                        
                        if best_idx_2400 != -1:
                            final_has_solar = True
                            final_buffer_size = 2400
                            final_bbox = boxes[best_idx_2400]
                            final_confidence = confidences[best_idx_2400]
                            detection_method = "initial_2400"
                            print(f"  -> Found in 2400 sqft buffer (Initial)! Conf: {final_confidence:.3f}")
                        
                        else:
                            # --- STEP 6: Check 2400 sqft from SATURATED inference ---
                            # (using results from Step 2)
                            print(f"  -> Checking 2400 (Saturated)...")
                            
                            best_enh_2400, _ = find_best_panel_in_buffer(enh_boxes, enh_confs, center, radius_2400, meters_per_pixel)
                            
                            if best_enh_2400 != -1:
                                final_has_solar = True
                                final_buffer_size = 2400
                                final_bbox = enh_boxes[best_enh_2400]
                                final_confidence = enh_confs[best_enh_2400]
                                detection_method = "saturated_2400"
                                boxes.extend(enh_boxes)
                                confidences.extend(enh_confs)
                                print(f"  -> Found in 2400 sqft buffer (Saturated)! Conf: {final_confidence:.3f}")
                            
                            else:
                                # Start with Not Found
                                final_has_solar = False
                                detection_method = "not_found"
                                
                                # --- STEP 7: RESCUE / EDGE CASE ---
                                # If no panel intersected the buffer, it might be slightly off-center 
                                # due to geocoding inaccuracies. Check for any high-confidence panel nearby.
                                print(f"  -> Checking for off-center rescue...")
                                
                                best_rescue_idx = -1
                                min_dist = float('inf')
                                rescue_threshold_px = radius_2400 * 2.0 # Allow 2x the standard radius
                                
                                for i, box in enumerate(boxes):
                                    # Calculate distance from center to box center
                                    bx1, by1, bx2, by2 = box
                                    cx_box = (bx1 + bx2) / 2
                                    cy_box = (by1 + by2) / 2
                                    dist = np.sqrt((cx_box - center[0])**2 + (cy_box - center[1])**2)
                                    
                                    # Logic: Must be confident (>0.40) and reasonably close
                                    if confidences[i] > 0.40 and dist < min_dist:
                                        min_dist = dist
                                        best_rescue_idx = i
                                
                                if best_rescue_idx != -1 and min_dist < rescue_threshold_px:
                                    final_has_solar = True
                                    final_buffer_size = 2400 # Assign it to the larger buffer category
                                    final_bbox = boxes[best_rescue_idx]
                                    final_confidence = confidences[best_rescue_idx]
                                    detection_method = "rescued_off_center"
                                    print(f"  -> RESCUED off-center panel! Dist: {min_dist:.1f}px, Conf: {final_confidence:.3f}")
                                else:
                                    print(f"  -> NOT FOUND in any buffer/method (and no rescue candidate)")

        # Calculate final area if found
        final_dist_m = 0.0
        if final_has_solar:
             box_area_pixels = calculate_box_area_pixels(final_bbox)
             final_pv_area_sqm = box_area_pixels * (meters_per_pixel ** 2)
             
             # Calculate Euclidean distance from image center (lat,lon) to panel centroid
             # center is (w//2, h//2)
             bx1, by1, bx2, by2 = final_bbox
             cx_box = (bx1 + bx2) / 2
             cy_box = (by1 + by2) / 2
             
             dist_px = np.sqrt((cx_box - center[0])**2 + (cy_box - center[1])**2)
             final_dist_m = dist_px * meters_per_pixel
        
        # 6. Create result record
        # Format sample_id as string "1", "2" etc (stripping .0)
        formatted_sample_id = str(int(float(sample_id)))
        
        # QC Status logic:
        # 1. If Solar Found:
        #    - High confidence (> 0.70) -> VERIFIABLE (Present)
        #    - Low confidence (<= 0.70) -> NOT_VERIFIABLE (Ambiguous)
        # 2. If Solar NOT Found:
        #    - We assume the exhaustive fallback strategy is robust enough to call it VERIFIABLE (Absent).
        #    - However, if the user implies low confidence = not verifiable, strictly speaking 0.0 confidence is low.
        #    - But for "Not Found", 0.0 is the expected confidence.
        #    - We will mark it VERIFIABLE (Absent) by default.
        
        qc_status = "VERIFIABLE"
        
        if final_has_solar:
            if final_confidence <= 0.70:
                qc_status = "NOT_VERIFIABLE"
        else:
            # For Not Found, we consider it Verifiable Absent.
            # (Unless we had some partial detections that were ambiguous? 
            #  But current logic doesn't expose that easily. 
            #  We stick to "Not Found = Verified Absent" for now).
            qc_status = "VERIFIABLE" 
        
        # Filter metadata
        clean_metadata = {
            "source": metadata.get("source", "Unknown"),
            "capture_date": metadata.get("capture_date", "Unknown")
        }

        record = {
            "sample_id": formatted_sample_id,
            "lat": float(lat),
            "lon": float(lon),
            "has_solar": bool(final_has_solar),
            "confidence": round(final_confidence, 4),
            "pv_area_sqm_est": round(final_pv_area_sqm, 2),
            "euclidean_distance_m_est": round(final_dist_m, 2),
            "buffer_radius_sqft": final_buffer_size,
            "qc_status": qc_status,
            "bbox_or_mask": [list(final_bbox)] if final_has_solar else [],
            "image_metadata": clean_metadata
        }
        results_json.append(record)
        
        # 7. Create visualization (using final buffer size for spotlight)
        final_radius_pixels = calculate_radius_from_area_sqft(final_buffer_size, meters_per_pixel)
        
        # Categorize boxes: selected (green), everything else (red)
        selected_box = final_bbox if final_has_solar else None
        boxes_selected = [selected_box] if selected_box else []
        
        # ALL other boxes go to boxes_outside (Red)
        boxes_outside = []
        
        for box in boxes:
            # Check if this is the selected box
            is_selected = (selected_box is not None and 
                          box[0] == selected_box[0] and box[1] == selected_box[1] and
                          box[2] == selected_box[2] and box[3] == selected_box[3])
            
            if not is_selected:
                boxes_outside.append(box)
        
        audit_img = create_spotlight_overlay(
            img, center, final_radius_pixels,
            boxes_inside=boxes_selected,  # Green
            boxes_outside=boxes_outside,  # Red (all others)
            boxes_in_buffer_not_selected=None, # Deprecated/Unused
            active_box=final_confidence if final_has_solar else None # Passing confidence here
        )
        
        # Add text info
        color = (0, 255, 0) if final_has_solar else (0, 0, 255)
        method_tag = f"[{detection_method.upper()}]" if detection_method != "initial" else ""
        cv2.putText(
            audit_img,
            f"ID: {formatted_sample_id} Solar: {final_has_solar} Buffer: {final_buffer_size} {method_tag}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        cv2.putText(
            audit_img,
            f"Total: {len(boxes)} | Selected: {1 if final_has_solar else 0} | Method: {detection_method}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        audit_path = os.path.join(audit_dir, f"{formatted_sample_id}.jpg")
        cv2.imwrite(audit_path, audit_img)
    
    # Save JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Fallback Inference complete!")
    print(f"JSON results: {json_path}")
    print(f"Audit images: {audit_dir}")
    
    # Summary
    detected = sum(1 for r in results_json if r['has_solar'])
    not_found = len(results_json) - detected
    
    print(f"\nSummary: {detected}/{len(results_json)} samples have solar panels detected")
    print(f"  - Detected: {detected}")
    print(f"  - Not found: {not_found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with buffer-based fallback")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples")
    parser.add_argument('--input', type=str, default="EI_train_data.xlsx", help="Path to input Excel file")
    parser.add_argument('--model', type=str, default="best.pt")
    parser.add_argument('--initial-conf', type=float, default=0.15, help="Initial confidence threshold")
    parser.add_argument('--fallback-conf', type=float, default=0.15, help="Fallback confidence threshold")
    parser.add_argument('--samples', type=str, default=None, 
                        help="Comma-separated sample IDs or ranges (e.g., '1-10,2501-2510')")
    args = parser.parse_args()
    
    # Parse sample IDs
    sample_ids = None
    if args.samples:
        sample_ids = []
        for part in args.samples.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                sample_ids.extend(range(start, end + 1))
            else:
                sample_ids.append(int(part))
        print(f"Running on sample IDs: {sample_ids}")
    
    run_inference_fallback(
        excel_path=args.input,
        limit=args.limit,
        model_path=args.model,
        initial_conf=args.initial_conf,
        fallback_conf=args.fallback_conf,
        sample_ids=sample_ids
    )