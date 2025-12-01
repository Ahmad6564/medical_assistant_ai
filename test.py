import requests
import json

# API Configuration
API_BASE_URL = "http://localhost:8000"
NER_URL = f"{API_BASE_URL}/api/v1/ner/extract"

# ========== OPTION 1: Disable Authentication (Quickest Fix) ==========
# Edit configs/api_config.yaml and set:
# authentication:
#   required: false

# ========== OPTION 2: Use Authentication (Recommended) ==========

def get_auth_token():
    """Get JWT token by logging in."""
    login_url = f"{API_BASE_URL}/api/v1/auth/token"
    
    # Try default credentials
    credentials = [
        {"username": "admin", "password": "admin123"},
        {"username": "demo", "password": "demo123"}
    ]
    
    for cred in credentials:
        try:
            response = requests.post(login_url, json=cred)
            if response.status_code == 200:
                token_data = response.json()
                print(f"✓ Logged in as: {cred['username']}")
                return token_data["access_token"]
        except:
            continue
    
    return None

# Get authentication token
print("Authenticating...")
token = get_auth_token()

if token:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    print(f"✓ Authentication successful!\n")
else:
    print("⚠ Could not authenticate. Trying without auth...\n")
    headers = {
        "Content-Type": "application/json"
    }

# Test cases
test_cases = [
    {
        "name": "Basic Medical Conditions",
        "payload": {
            "text": "Patient diagnosed with type 2 diabetes mellitus and hypertension.",
            "model_type": "transformer",
            "include_linking": False
        }
    },
    {
        "name": "Anatomy + Problems",
        "payload": {
            "text": "Patient reports severe pain in the left knee and swelling in ankle.",
            "model_type": "transformer",
            "include_linking": False
        }
    },
    {
        "name": "Medical Tests",
        "payload": {
            "text": "Ordered CBC, chest X-ray, and MRI scan of the brain.",
            "model_type": "transformer",
            "include_linking": False
        }
    },
    {
        "name": "Treatments + Medications",
        "payload": {
            "text": "Started patient on metformin 500mg twice daily and prescribed physical therapy for back pain.",
            "model_type": "transformer",
            "include_linking": False
        }
    },
    {
        "name": "Complex Clinical Note",
        "payload": {
            "text": "Patient with history of coronary artery disease presents with chest pain radiating to left arm. EKG shows ST elevation.",
            "model_type": "transformer",
            "include_linking": False
        }
    }
]

# Run tests
print("="*70)
print("MEDICAL AI ASSISTANT - NER API TESTS")
print("="*70)
print(f"API Base URL: {API_BASE_URL}")
print(f"NER Endpoint: {NER_URL}")
print("="*70)

for test in test_cases:
    print(f"\n{'='*70}")
    print(f"Test: {test['name']}")
    print(f"{'='*70}")
    print(f"Input: {test['payload']['text'][:80]}...")
    
    try:
        response = requests.post(NER_URL, json=test['payload'], headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            entities = result.get('entities', [])
            print(f"✓ Success! Found {len(entities)} entities:")
            if entities:
                for entity in entities:
                    confidence = entity.get('confidence', entity.get('score', 'N/A'))
                    if isinstance(confidence, float):
                        print(f"  - '{entity['text']}' → {entity['label']} (confidence: {confidence:.2f})")
                    else:
                        print(f"  - '{entity['text']}' → {entity['label']}")
            else:
                print("  (No entities found)")
        elif response.status_code == 403:
            print(f"✗ Error: 403 - Not Authenticated")
            print("   Fix: Run with authentication or disable auth in configs/api_config.yaml")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")

print("\n" + "="*70)
print("Testing Complete!")
print("="*70)