import requests

def test_osm_search(query):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "limit": 3,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "SolarAgent/1.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        print(f"Status: {response.status_code}")
        for item in data:
            print(f"\n--- Result ---")
            print(f"Name: {item.get('display_name')}")
            print(f"Lat: {item.get('lat')}")
            print(f"Lon: {item.get('lon')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_osm_search("IIT Madras")
