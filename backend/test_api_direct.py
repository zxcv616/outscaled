#!/usr/bin/env python3
"""
Direct API test to debug the statistical analysis endpoint
"""

import requests
import json

def test_api_endpoints():
    """Test the API endpoints directly"""
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test data
    test_data = {
        "player_name": "iBo",
        "prop_type": "kills", 
        "prop_value": 4.5,
        "opponent": "",
        "tournament": "",
        "map_range": [1]
    }
    
    print("🧪 Testing API Endpoints")
    print("=" * 40)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test regular prediction endpoint
    print("\n2. Testing prediction endpoint...")
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test statistical insights endpoint
    print("\n3. Testing statistical insights endpoint...")
    try:
        response = requests.post(f"{base_url}/statistics/insights", json=test_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success! Got insights data")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test probability distribution endpoint
    print("\n4. Testing probability distribution endpoint...")
    try:
        response = requests.post(f"{base_url}/statistics/probability-distribution", json=test_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success! Got distribution data")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test comprehensive endpoint
    print("\n5. Testing comprehensive endpoint...")
    try:
        response = requests.post(f"{base_url}/statistics/comprehensive", json=test_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success! Got comprehensive data")
            print(f"   Summary stats: {data.get('summary_stats', {}).get('mean_recent', 'N/A')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_api_endpoints() 