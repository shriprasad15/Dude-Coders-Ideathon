import requests

URL = "https://sue-asymmetric-nonprogressively.ngrok-free.dev/smart-search"

def test_query(description, query, file_path=None):
    print(f"\n--- Testing: {description} ---")
    data = {"query": query}
    files = {}
    if file_path:
        files = {"file": open(file_path, "rb")}
    
    try:
        response = requests.post(URL, data=data, files=files if file_path else None)
        if response.status_code == 200:
            print("SUCCESS! Response:")
            print(response.json())
        else:
            print(f"FAILED (Status {response.status_code}):")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {e}")

# 1. Test File-less Query
test_query("File-less Query (Chennai)", "Find the coordinates of IIT Madras")

# 2. Test File Query (Create a dummy CSV first)
with open("test.csv", "w") as f:
    f.write("Site Name,Description\nSite A,High solar potential area near the lake\nSite B,Shadow heavy area")

test_query("File Query (Site A)", "Where is Site A?", "test.csv")
