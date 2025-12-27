import os
import requests
from langchain_core.tools import tool
from backend.services import inference_service, image_retriever

@tool
def search_places(query: str):
    """
    Search for a place by name or address to get its coordinates.
    Useful when the user asks about a specific location (e.g. "Puzhuthivakkam", "10 Downing Street").
    Returns the formatted address, latitude, and longitude.
    """
    # Use OpenStreetMap (Nominatim) as it doesn't require an API Key for geocoding
    # and is a good fallback when Google Geocoding API is not enabled.
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 3,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "SolarAgent/1.0 (internal-test)"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if response.status_code == 200 and data:
            results = []
            for res in data:
                results.append({
                    "address": res.get("display_name"),
                    "lat": float(res.get("lat")),
                    "lon": float(res.get("lon"))
                })
            return results
        else:
             print(f"OSM Search returned no results or error: {response.status_code}")
             # Return empty list or specific error
             return []
            
    except Exception as e:
        return {"error": str(e)}

@tool
def analyze_solar_potential(lat: float, lon: float):
    """
    Analyze the solar potential for a specific latitude and longitude.
    This tool captures a satellite image and runs the AI detection model.
    Returns whether solar panels are detected, the confidence, and the estimated area.
    """
    try:
        # 1. Get Image
        image_path, metadata = image_retriever.get_image(lat, lon, zoom=20)
        
        # 2. Run Inference
        result = inference_service.process_single_image(image_path, lat, lon)
        
        if "error" in result:
             return {"error": result["error"]}
             
        # Return summary
        return {
            "has_solar": result["has_solar"],
            "confidence": result["confidence"],
            "pv_area_sqm": result.get("pv_area_sqm", 0),
            "detection_method": result.get("detection_method", "unknown"),
             # Important: The frontend might need a signal to re-fetch the analysis result 
             # logic or we pass the data to the frontend via the agent response.
             # For now, we return data that the agent can summarize.
        }
    except Exception as e:
        return {"error": str(e)}
