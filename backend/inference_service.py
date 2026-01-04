import cv2
import numpy as np
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# Ensure we can import from root
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    calculate_meters_per_pixel,
    calculate_radius_from_area_sqft,
    calculate_intersection_area,
    create_spotlight_overlay,
    crop_buffer_region,
    run_inference_on_crop,
    calculate_box_area_pixels
)

class InferenceService:
    def __init__(self, model_path="best.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
    
    def enhance_saturation(self, image, factor=1.5):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def find_best_panel_in_buffer(self, boxes, confidences, center, radius_pixels, meters_per_pixel=None):
        """
        Find the panel with largest overlap within the buffer region.
        """
        best_overlap = 0
        best_idx = -1
        
        for i, box in enumerate(boxes):
            # Calculate panel area in sqm if meters_per_pixel provided (for debug/filtering)
            if meters_per_pixel is not None:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area_sqm = (w * h) * (meters_per_pixel ** 2)
                # Could add size check here if needed, consistent with inference.py

            overlap = calculate_intersection_area(box, center, radius_pixels)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        
        return best_idx, best_overlap

    def process_single_image(self, image_path, lat, lon):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        initial_conf = 0.15
        fallback_conf = 0.05
        
        # 1. Initial Inference on FULL image
        results = self.model.predict(img, conf=initial_conf, augment=True, save=False, verbose=False)
        result = results[0]
        boxes = result.boxes.xyxy.tolist() if result.boxes else []
        confidences = result.boxes.conf.tolist() if result.boxes else []
        
        meters_per_pixel = calculate_meters_per_pixel(lat, zoom=20)
        radius_1200 = calculate_radius_from_area_sqft(1200, meters_per_pixel)
        radius_2400 = calculate_radius_from_area_sqft(2400, meters_per_pixel)
        
        final_has_solar = False
        final_buffer_size = 2400
        final_bbox = []
        final_confidence = 0.0
        detection_method = "initial"

        # --- STEP 1: Check 1200 sqft buffer from INITIAL inference ---
        best_idx_1200, _ = self.find_best_panel_in_buffer(boxes, confidences, center, radius_1200, meters_per_pixel)
        
        if best_idx_1200 != -1:
            final_has_solar = True
            final_buffer_size = 1200
            final_bbox = boxes[best_idx_1200]
            final_confidence = confidences[best_idx_1200]
            detection_method = "initial"
        
        else:
            # --- STEP 2: Check 1200 sqft from SATURATED inference ---
            img_enhanced = self.enhance_saturation(img, factor=1.5)
            results_enhanced = self.model.predict(img_enhanced, conf=fallback_conf, augment=True, save=False, verbose=False)
            result_enhanced = results_enhanced[0]
            
            enh_boxes = result_enhanced.boxes.xyxy.tolist() if result_enhanced.boxes else []
            enh_confs = result_enhanced.boxes.conf.tolist() if result_enhanced.boxes else []
            
            best_enh_1200, _ = self.find_best_panel_in_buffer(enh_boxes, enh_confs, center, radius_1200, meters_per_pixel)
            
            if best_enh_1200 != -1:
                final_has_solar = True
                final_buffer_size = 1200
                final_bbox = enh_boxes[best_enh_1200]
                final_confidence = enh_confs[best_enh_1200]
                detection_method = "saturated_1200"
                boxes.extend(enh_boxes) # For visualization
            
            else:
                # --- STEP 3: Check 1200 sqft by CROPPING ---
                cropped_1200, offset_1200, _ = crop_buffer_region(img, center, radius_1200, padding=30)
                crop_boxes_1200, crop_confs_1200 = run_inference_on_crop(
                    self.model, cropped_1200, offset_1200, conf=fallback_conf
                )
                
                best_crop_1200, _ = self.find_best_panel_in_buffer(crop_boxes_1200, crop_confs_1200, center, radius_1200, meters_per_pixel)
                
                if best_crop_1200 != -1:
                    final_has_solar = True
                    final_buffer_size = 1200
                    final_bbox = crop_boxes_1200[best_crop_1200]
                    final_confidence = crop_confs_1200[best_crop_1200]
                    detection_method = "crop_1200"
                    boxes.extend(crop_boxes_1200)
                
                else:
                    # --- STEP 4: Check 1200 sqft by CROPPING + SATURATION ---
                    cropped_1200_sat = self.enhance_saturation(cropped_1200, factor=1.5)
                    crop_sat_boxes, crop_sat_confs = run_inference_on_crop(
                        self.model, cropped_1200_sat, offset_1200, conf=fallback_conf
                    )
                    
                    best_crop_sat_1200, _ = self.find_best_panel_in_buffer(crop_sat_boxes, crop_sat_confs, center, radius_1200, meters_per_pixel)
                    
                    if best_crop_sat_1200 != -1:
                        final_has_solar = True
                        final_buffer_size = 1200
                        final_bbox = crop_sat_boxes[best_crop_sat_1200]
                        final_confidence = crop_sat_confs[best_crop_sat_1200]
                        detection_method = "crop_sat_1200"
                        boxes.extend(crop_sat_boxes)
                    
                    else:
                        # --- STEP 5: Check 2400 sqft from INITIAL inference ---
                        best_idx_2400, _ = self.find_best_panel_in_buffer(boxes, confidences, center, radius_2400, meters_per_pixel)
                        
                        if best_idx_2400 != -1:
                            final_has_solar = True
                            final_buffer_size = 2400
                            final_bbox = boxes[best_idx_2400]
                            final_confidence = confidences[best_idx_2400]
                            detection_method = "initial_2400"
                        
                        else:
                            # --- STEP 6: Check 2400 sqft from SATURATED inference ---
                            best_enh_2400, _ = self.find_best_panel_in_buffer(enh_boxes, enh_confs, center, radius_2400, meters_per_pixel)
                            
                            if best_enh_2400 != -1:
                                final_has_solar = True
                                final_buffer_size = 2400
                                final_bbox = enh_boxes[best_enh_2400]
                                final_confidence = enh_confs[best_enh_2400]
                                detection_method = "saturated_2400"
                                boxes.extend(enh_boxes)
                            
                            else:
                                # Not found in any method
                                detection_method = "not_found"

        # --- Rescue Step / Off-center check ---
        if not final_has_solar:
             best_rescue_idx = -1
             min_dist = float('inf')
             rescue_threshold_px = radius_2400 * 2.0 
             
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
                 final_buffer_size = 2400 
                 final_bbox = boxes[best_rescue_idx]
                 final_confidence = confidences[best_rescue_idx]
                 detection_method = "rescued_off_center"
        
        # Calculate final metrics if found
        final_pv_area_sqm = 0.0
        final_dist_m = 0.0
        
        if final_has_solar:
             box_area_pixels = calculate_box_area_pixels(final_bbox)
             final_pv_area_sqm = box_area_pixels * (meters_per_pixel ** 2)
             
             # Calculate Euclidean distance from image center to panel centroid
             bx1, by1, bx2, by2 = final_bbox
             cx_box = (bx1 + bx2) / 2
             cy_box = (by1 + by2) / 2
             
             dist_px = np.sqrt((cx_box - center[0])**2 + (cy_box - center[1])**2)
             final_dist_m = dist_px * meters_per_pixel
        
        # Visualization
        final_radius = calculate_radius_from_area_sqft(final_buffer_size, meters_per_pixel)
        selected_box = final_bbox if final_has_solar else None
        
        vis_img = create_spotlight_overlay(
            img, center, final_radius,
            boxes_inside=[selected_box] if selected_box else [],
            boxes_outside=[b for b in boxes if b != selected_box], # simplified
            active_box=final_confidence if final_has_solar else None
        )
        
        return {
            "has_solar": final_has_solar,
            "confidence": final_confidence,
            "bbox": final_bbox,
            "vis_image": vis_img,
            "buffer_size": final_buffer_size,
            "pv_area_sqm": final_pv_area_sqm,
            "euclidean_distance": final_dist_m,
            "detection_method": detection_method
        }
