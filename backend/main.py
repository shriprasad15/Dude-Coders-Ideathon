from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import os
import sys
import base64
import cv2
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Ensure imports work
sys.path.append(str(Path(__file__).parent.parent))
from backend.inference_service import InferenceService
from image_retriever import ImageRetriever
from backend.llm_service import llm_service

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
BASE_DIR = Path(__file__).parent.parent
service = InferenceService(model_path=str(BASE_DIR / "best.pt"))
retriever = ImageRetriever(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))

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
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-search")
async def smart_search(query: str = Form(...), file: UploadFile = File(None)):
    print(f"Smart Search Query: {query}")
    result = await llm_service.process_file_query(file, query)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
