import requests
import os
from PIL import Image
from io import BytesIO
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ImageRetriever:
    def __init__(self, api_key=None, cache_dir="cache/images", use_mock=False):
        # Prefer provided key, then env var, then None
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.cache_dir = cache_dir
        self.use_mock = use_mock
        self.cache_dir = cache_dir
        self.use_mock = use_mock
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def get_image(self, lat, lon, zoom=20, size="1024x1024", maptype="satellite"):
        """
        Retrieves an image for the given coordinates.
        Checks cache first.
        """
        # Create a unique filename based on parameters
        params_str = f"{lat}_{lon}_{zoom}_{size}_{maptype}"
        filename = hashlib.md5(params_str.encode()).hexdigest() + ".png"
        filepath = os.path.join(self.cache_dir, filename)
        
        # Check cache
        if os.path.exists(filepath):
            # print(f"Loading from cache: {filepath}")
            return filepath, self._get_metadata(filepath, "Cache", datetime.now().strftime("%Y-%m-%d"))

        if self.use_mock:
            return self._generate_mock_image(filepath), self._get_metadata(filepath, "Mock", datetime.now().strftime("%Y-%m-%d"))

        if not self.api_key:
            print("Warning: No API Key provided. Using mock image.")
            return self._generate_mock_image(filepath), self._get_metadata(filepath, "Mock", datetime.now().strftime("%Y-%m-%d"))

        # Fetch from Google Maps Static API
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": size,
            "maptype": maptype,
            "key": self.api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            image.save(filepath)
            return filepath, self._get_metadata(filepath, "Google Maps Static API", datetime.now().strftime("%Y-%m-%d"))
            
        except Exception as e:
            print(f"Error fetching image: {e}")
            # Fallback to mock if API fails
            return self._generate_mock_image(filepath), self._get_metadata(filepath, "Mock (Fallback)", datetime.now().strftime("%Y-%m-%d"))

    def _generate_mock_image(self, filepath):
        """Generates a placeholder image for testing without API key."""
        img = Image.new('RGB', (600, 600), color = (73, 109, 137))
        img.save(filepath)
        return filepath

    def _get_metadata(self, filepath, source, date):
        return {
            "filepath": filepath,
            "source": source,
            "capture_date": date
        }

if __name__ == "__main__":
    # Test
    # Test
    retriever = ImageRetriever()
    path, meta = retriever.get_image(21.5030457, 70.45946829)
    print(f"Image saved to: {path}")
    print(f"Metadata: {meta}")
