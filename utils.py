import numpy as np
import cv2
import math

def calculate_meters_per_pixel(lat, zoom):
    """
    Calculate meters per pixel at a given latitude and zoom level.
    Based on Google Maps Mercator projection.
    """
    return 156543.03392 * np.cos(np.radians(lat)) / (2**zoom)

def calculate_radius_from_area_sqft(area_sqft, meters_per_pixel):
    """
    Calculate the radius in pixels for a given area in square feet.
    """
    # Convert sqft to sqm
    area_sqm = area_sqft * 0.092903
    
    # Area = pi * r^2  =>  r = sqrt(Area / pi) (in meters)
    radius_meters = np.sqrt(area_sqm / np.pi)
    
    # Convert to pixels
    radius_pixels = radius_meters / meters_per_pixel
    return radius_pixels

def calculate_box_area_pixels(box):
    """
    Calculate area of a bounding box [x1, y1, x2, y2].
    """
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h

def calculate_intersection_area(box, circle_center, circle_radius):
    """
    Calculate the approximate intersection area between a box and a circle.
    Uses a mask-based approach for simplicity and accuracy with irregular shapes if needed later.
    """
    # Create a blank mask for the box
    # We need a canvas large enough to hold both. 
    # Since we are working in image coordinates, we can just use a localized mask or the full image size if known.
    # For efficiency, we'll compute this on a localized grid.
    
    x1, y1, x2, y2 = map(int, box)
    cx, cy = map(int, circle_center)
    r = int(circle_radius)
    
    # Define bounds of the computation
    min_x = min(x1, cx - r)
    min_y = min(y1, cy - r)
    max_x = max(x2, cx + r)
    max_y = max(y2, cy + r)
    
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    
    if w <= 0 or h <= 0:
        return 0
        
    # Create masks
    box_mask = np.zeros((h, w), dtype=np.uint8)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw box (offset by min_x, min_y)
    cv2.rectangle(box_mask, (x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y), 1, -1)
    
    # Draw circle
    cv2.circle(circle_mask, (cx - min_x, cy - min_y), r, 1, -1)
    
    # Intersection
    intersection = cv2.bitwise_and(box_mask, circle_mask)
    return np.sum(intersection)

def create_spotlight_overlay(image, center, radius, boxes_inside=None, boxes_outside=None, 
                               boxes_in_buffer_not_selected=None, active_box=None):
    """
    Create a spotlight effect: darken everything outside the circle.
    Draw solar panels detected:
    - Green for the SELECTED panel (largest overlap with buffer)
    - Yellow for panels inside buffer but NOT selected
    - Red for panels outside the buffer radius
    
    Args:
        image: Input image
        center: Center point (x, y) of the buffer circle
        radius: Radius of the buffer circle in pixels
        boxes_inside: List of SELECTED bounding boxes (drawn in green) - should be 0 or 1
        boxes_outside: List of bounding boxes outside the buffer (drawn in red)
        boxes_in_buffer_not_selected: List of boxes in buffer but not selected (drawn in yellow)
        active_box: (deprecated) Single active box for backward compatibility
    """
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    cx, cy = map(int, center)
    r = int(radius)
    
    # White circle on black background
    cv2.circle(mask, (cx, cy), r, 255, -1)
    
    # Darken outside the circle
    # Invert mask to get the "outside" area
    inverse_mask = cv2.bitwise_not(mask)
    
    # Darken the image where the inverse mask is white
    # We blend the original image with black
    darkened = cv2.addWeighted(overlay, 0.3, np.zeros_like(overlay), 0.7, 0)
    
    # Combine: Inside circle uses original image, outside uses darkened
    # mask is 255 inside, 0 outside
    # darkened is dark everywhere
    # We want: (image & mask) + (darkened & inverse_mask)
    
    img_masked = cv2.bitwise_and(image, image, mask=mask)
    dark_masked = cv2.bitwise_and(darkened, darkened, mask=inverse_mask)
    
    final_image = cv2.add(img_masked, dark_masked)
    
    # Draw the buffer circle outline
    cv2.circle(final_image, (cx, cy), r, (0, 255, 255), 2) # Yellow
    
    # Combined logic: 
    # 1. Draw NON-SELECTED boxes in RED (from boxes_outside list)
    # 2. Draw SELECTED box in GREEN (from boxes_inside list) with confidence
    
    # Draw all boxes outside/not-selected in RED
    if boxes_outside:
        for box in boxes_outside:
            x1, y1, x2, y2 = map(int, box)
            # Draw filled semi-transparent rectangle
            overlay_box = final_image.copy()
            cv2.rectangle(overlay_box, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red fill
            cv2.addWeighted(overlay_box, 0.3, final_image, 0.7, 0, final_image)
            # Draw solid border
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red border
            # No label needed for "not selected" as per request, but can keep "OUT" or remove it.
            # User said "red which is not". Let's keep it simple.
    
    # Draw SELECTED boxes (largest overlap) in GREEN
    if boxes_inside:
        for i, box in enumerate(boxes_inside):
            x1, y1, x2, y2 = map(int, box)
            # Draw filled semi-transparent rectangle
            overlay_box = final_image.copy()
            cv2.rectangle(overlay_box, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green fill
            cv2.addWeighted(overlay_box, 0.3, final_image, 0.7, 0, final_image)
            # Draw solid border
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green border (thicker)
            
            # Add label with confidence if available
            label = "SELECTED"
            if active_box and isinstance(active_box, (float, np.floating)): # Hack: overload active_box for confidence
                 label = f"SOLAR: {active_box:.2f}"
            
            cv2.putText(final_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    return final_image


def crop_buffer_region(image, center, radius, padding=10):
    """
    Crop the image to the buffer region (square bounding box around the circle).
    
    Args:
        image: Input image (numpy array)
        center: Center point (x, y) of the buffer circle
        radius: Radius of the buffer circle in pixels
        padding: Extra padding around the circle (in pixels)
    
    Returns:
        cropped_image: Cropped image region
        crop_offset: (offset_x, offset_y) - top-left corner of crop in original image
        crop_bounds: (x1, y1, x2, y2) - actual crop bounds in original image
    """
    h, w = image.shape[:2]
    cx, cy = map(int, center)
    r = int(radius) + padding
    
    # Calculate crop bounds (square region around circle)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    
    # Crop the image
    cropped_image = image[y1:y2, x1:x2].copy()
    
    return cropped_image, (x1, y1), (x1, y1, x2, y2)


def map_boxes_to_original(boxes, crop_offset):
    """
    Map bounding boxes from cropped image coordinates back to original image coordinates.
    
    Args:
        boxes: List of bounding boxes in cropped image coords [x1, y1, x2, y2]
        crop_offset: (offset_x, offset_y) - top-left corner of crop in original image
    
    Returns:
        List of bounding boxes in original image coordinates
    """
    offset_x, offset_y = crop_offset
    mapped_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        mapped_boxes.append([
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y
        ])
    return mapped_boxes


def run_inference_on_crop(model, cropped_image, crop_offset, conf=0.10):
    """
    Run inference on a cropped image and return boxes mapped to original coordinates.
    Uses lower confidence threshold and augmentation for better detection.
    
    Args:
        model: YOLO model instance
        cropped_image: Cropped image region
        crop_offset: (offset_x, offset_y) for mapping back to original
        conf: Confidence threshold (default lower for crop-based detection)
    
    Returns:
        mapped_boxes: List of bboxes in original image coordinates
        confidences: List of confidence scores
    """
    import tempfile
    import os
    
    # Save cropped image temporarily for YOLO inference
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        cv2.imwrite(tmp_path, cropped_image)
    
    try:
        # Run inference with lower threshold and TTA on cropped region
        results = model.predict(tmp_path, conf=conf, augment=True, save=False, verbose=False)
        result = results[0]
        
        # Get boxes and confidences
        crop_boxes = result.boxes.xyxy.tolist() if result.boxes else []
        confidences = result.boxes.conf.tolist() if result.boxes else []
        
        # Map boxes back to original image coordinates
        mapped_boxes = map_boxes_to_original(crop_boxes, crop_offset)
        
        return mapped_boxes, confidences
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def generate_tiles(image, tile_size=256, overlap=64):
    """
    Generate overlapping tiles from an image for tiled inference.
    
    Args:
        image: Input image (numpy array)
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles in pixels
    
    Yields:
        (tile_image, offset_x, offset_y) for each tile
    """
    h, w = image.shape[:2]
    step = tile_size - overlap
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # Calculate actual tile bounds
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # Extract tile
            tile = image[y1:y2, x1:x2]
            
            # Pad if tile is smaller than tile_size (at edges)
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            yield tile, x1, y1


def run_tiled_inference(model, image, tile_size=256, overlap=64, conf=0.05):
    """
    Run inference on overlapping tiles and merge results.
    
    Args:
        model: YOLO model instance
        image: Full input image (numpy array)
        tile_size: Size of each tile
        overlap: Overlap between tiles
        conf: Confidence threshold
    
    Returns:
        all_boxes: List of [x1, y1, x2, y2] in original image coordinates
        all_confidences: List of confidence scores
    """
    import tempfile
    import os
    
    all_boxes = []
    all_confidences = []
    
    for tile, offset_x, offset_y in generate_tiles(image, tile_size, overlap):
        # Save tile temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, tile)
        
        try:
            # Run inference on tile
            results = model.predict(tmp_path, conf=conf, augment=True, verbose=False)
            result = results[0]
            
            if result.boxes:
                boxes = result.boxes.xyxy.tolist()
                confs = result.boxes.conf.tolist()
                
                # Map boxes back to original image coordinates
                for box, c in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    # Adjust for tile offset
                    mapped_box = [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y
                    ]
                    all_boxes.append(mapped_box)
                    all_confidences.append(c)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    # Apply NMS to remove duplicate detections from overlapping tiles
    if all_boxes:
        all_boxes, all_confidences = apply_nms(all_boxes, all_confidences, iou_threshold=0.5)
    
    return all_boxes, all_confidences


def apply_nms(boxes, confidences, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: List of [x1, y1, x2, y2]
        confidences: List of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        filtered_boxes, filtered_confidences
    """
    if not boxes:
        return [], []
    
    # Convert to numpy arrays
    boxes_np = np.array(boxes)
    confs_np = np.array(confidences)
    
    # Sort by confidence (descending)
    indices = np.argsort(confs_np)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the highest confidence box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = []
        
        for idx in remaining:
            iou = calculate_iou(boxes_np[current], boxes_np[idx])
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Keep boxes with IoU below threshold
        indices = remaining[ious < iou_threshold]
    
    filtered_boxes = [boxes[i] for i in keep]
    filtered_confidences = [confidences[i] for i in keep]
    
    return filtered_boxes, filtered_confidences


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

