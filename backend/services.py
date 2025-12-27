import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can import from root
sys.path.append(str(Path(__file__).parent.parent))

from backend.inference_service import InferenceService
from image_retriever import ImageRetriever

load_dotenv()

# Initialize services
BASE_DIR = Path(__file__).parent.parent
# Look for model in root
MODEL_PATH = BASE_DIR / "best.pt"

print(f"Initializing Service with Model at: {MODEL_PATH}")

inference_service = InferenceService(model_path=str(MODEL_PATH))
image_retriever = ImageRetriever(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))
