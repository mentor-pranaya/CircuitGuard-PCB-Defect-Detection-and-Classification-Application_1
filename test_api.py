import requests
import os

# Configuration
API_URL = "http://localhost:8000/predict"
TEMPLATE_PATH = r"c:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET\PCB_USED\01.JPG"
TEST_PATH = r"c:\Users\jayza\OneDrive\Desktop\Infosys\PCB_DATASET\PCB_DATASET\images\Missing_hole\01_missing_hole_01.jpg"

def test_api():
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found at {TEMPLATE_PATH}")
        return
    if not os.path.exists(TEST_PATH):
        print(f"Error: Test image not found at {TEST_PATH}")
        return

    print(f"Sending request to {API_URL}...")
    try:
        files = {
            'template': ('template.jpg', open(TEMPLATE_PATH, 'rb'), 'image/jpeg'),
            'test': ('test.jpg', open(TEST_PATH, 'rb'), 'image/jpeg')
        }
        response = requests.post(API_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            print("Success!")
            print("Response:", response.json())
        else:
            print(f"Failed with status code {response.status_code}")
            print("Response:", response.text)
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_api()
