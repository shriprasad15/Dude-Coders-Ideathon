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
        best_overlap = 0
        best_idx = -1
        
        for i, box in enumerate(boxes):
            if meters_per_pixel is not None:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area_sqm = (w * h) * (meters_per_pixel ** 2)
                if area_sqm > 300.0:
                    continue
                    
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
        
        # 1. Initial Inference
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
        
        # Strategy Implementation
        # Try 1200 Initial
        best_idx, _ = self.find_best_panel_in_buffer(boxes, confidences, center, radius_1200, meters_per_pixel)
        
        if best_idx != -1:
            final_has_solar = True
            final_buffer_size = 1200
            final_bbox = boxes[best_idx]
            final_confidence = confidences[best_idx]
        else:
            # Try 1200 Saturated
            img_sat = self.enhance_saturation(img)
            res_sat = self.model.predict(img_sat, conf=fallback_conf, augment=True, save=False, verbose=False)[0]
            sat_boxes = res_sat.boxes.xyxy.tolist() if res_sat.boxes else []
            sat_confs = res_sat.boxes.conf.tolist() if res_sat.boxes else []
            
            best_sat, _ = self.find_best_panel_in_buffer(sat_boxes, sat_confs, center, radius_1200, meters_per_pixel)
            
            if best_sat != -1:
                final_has_solar = True
                final_buffer_size = 1200
                final_bbox = sat_boxes[best_sat]
                final_confidence = sat_confs[best_sat]
                boxes = sat_boxes # For visualization
            else:
                # Try 1200 Crop
                cropped, offset, _ = crop_buffer_region(img, center, radius_1200, padding=30)
                c_boxes, c_confs = run_inference_on_crop(self.model, cropped, offset, conf=fallback_conf)
                best_crop, _ = self.find_best_panel_in_buffer(c_boxes, c_confs, center, radius_1200, meters_per_pixel)
                
                if best_crop != -1:
                    final_has_solar = True
                    final_buffer_size = 1200
                    final_bbox = c_boxes[best_crop]
                    final_confidence = c_confs[best_crop]
                    boxes = c_boxes
                else:
                    # Try 2400 Initial (Fallback to larger buffer if 1200 fails entirely)
                     best_2400, _ = self.find_best_panel_in_buffer(boxes, confidences, center, radius_2400, meters_per_pixel)
                     if best_2400 != -1:
                        final_has_solar = True
                        final_buffer_size = 2400
                        final_bbox = boxes[best_2400]
                        final_confidence = confidences[best_2400]
        
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
            "buffer_size": final_buffer_size
        }
