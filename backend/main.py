from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import os
import sys
import base64
import cv2
import pandas as pd
import io
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware



# Ensure imports work
sys.path.append(str(Path(__file__).parent.parent))
from backend.inference_service import InferenceService
from image_retriever import ImageRetriever
# Import shared services
from backend.services import inference_service as service, image_retriever as retriever

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service (load model once)
# Services are now initialized in backend.services

class AnalyzeRequest(BaseModel):
    lat: float
    lon: float

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    try:
        print(f"Analyzing location: {req.lat}, {req.lon}")
        
        # 1. Get Image
        image_path, metadata = retriever.get_image(req.lat, req.lon, zoom=20)
        
        # 2. Run Inference
        result = service.process_single_image(image_path, req.lat, req.lon)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # 3. Encode Result Image to Base64
        vis_img = result["vis_image"]
        _, buffer = cv2.imencode('.jpg', vis_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "has_solar": result["has_solar"],
            "confidence": float(result["confidence"]),
            "image_base64": img_str,
            "image_base64": img_str,
            "metadata": metadata,
            "pv_area_sqm_est": float(result.get("pv_area_sqm", 0)),
            "euclidean_distance_m_est": float(result.get("euclidean_distance", 0)),
            "buffer_size": result.get("buffer_size", 2400),
            "detection_method": result.get("detection_method", "unknown")
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk-analyze")
async def bulk_analyze(file: UploadFile = File(...)):
    try:
        print(f"Processing bulk file: {file.filename}")
        
        # Read file based on extension
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload CSV or Excel.")

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Check for required columns
        # Allow variations like 'lat', 'latitude', 'lats' etc.
        lat_col = next((col for col in df.columns if col in ['lat', 'latitude', 'lats', 'y']), None)
        lon_col = next((col for col in df.columns if col in ['lon', 'lng', 'longitude', 'longs', 'x']), None)

        if not lat_col or not lon_col:
            raise HTTPException(status_code=400, detail="File must contain 'lat' and 'lon' columns (or similar variations).")

        results = []
        
        # Iterate and Process
        for index, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                # Re-use the logic from analyze endpoint
                # 1. Get Image
                image_path, metadata = retriever.get_image(lat, lon, zoom=20)
                
                # 2. Run Inference
                inf_res = service.process_single_image(image_path, lat, lon)
                
                if "error" in inf_res:
                    print(f"Error processing row {index}: {inf_res['error']}")
                    continue # Skip failed
                
                # 3. Encode Result Image
                # We return the base64 image so the user can verify it in the report
                vis_img = inf_res["vis_image"]
                  # 3. Encode Result Image
                img_str = base64.b64encode(cv2.imencode('.jpg', inf_res["vis_image"])[1]).decode()

                # 4. Get Sample ID if present
                sample_id = None
                id_col = next((col for col in df.columns if col in ['sample_id', 'id', 'sampleid']), None)
                if id_col:
                    sample_id = str(row[id_col])
                else:
                    sample_id = str(index + 1) # Fallback to 1-based index

                results.append({
                    "sample_id": sample_id, # Pass through ID
                    "lat": lat,
                    "lng": lon,
                    "has_solar": inf_res["has_solar"],
                    "confidence": float(inf_res["confidence"]),
                    "image_base64": img_str,
                    "pv_area_sqm_est": inf_res.get("pv_area_sqm", 0),
                    "euclidean_distance_m_est": inf_res.get("euclidean_distance", 0),
                    "buffer_size": inf_res.get("buffer_size", 0),
                    "detection_method": inf_res.get("detection_method", "standard"),
                    "qc_status": "VERIFIABLE", # Placeholder/Static for now
                     # Re-formatting bbox for frontend consistency if needed, strictly passing what inference gave
                    "bbox_or_mask": inf_res.get("bbox", [])
                })
                
            except Exception as row_err:
                print(f"Skipping row {index} due to error: {row_err}")
                continue

        return results

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Bulk Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
