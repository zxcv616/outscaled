#!/usr/bin/env python3
"""
Simple test runner for Outscaled.gg API
Run this script to verify all endpoints are working correctly.
"""

import requests
import json
import time
import sys

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_player_search():
    """Test player search functionality"""
    print("Testing player search...")
    try:
        response = requests.get(f"{API_BASE}/players/search?query=pat&limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "players" in data and len(data["players"]) > 0:
                print(f"✅ Player search passed - found {len(data['players'])} players")
                return True
            else:
                print("❌ Player search failed - no players found")
                return False
        else:
            print(f"❌ Player search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Player search error: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("Testing prediction...")
    try:
        prediction_data = {
            "player_name": "PatkicaA",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "confidence" in data:
                print(f"✅ Prediction passed - {data['prediction']} with {data['confidence']:.1f}% confidence")
                return True
            else:
                print("❌ Prediction failed - missing required fields")
                return False
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_extreme_values():
    """Test prediction with extreme values"""
    print("Testing extreme value handling...")
    try:
        prediction_data = {
            "player_name": "PatkicaA",
            "prop_type": "kills",
            "prop_value": 1000,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["prediction"] == "LESS" and data["confidence"] == 99.9:
                print("✅ Extreme value handling passed")
                return True
            else:
                print("❌ Extreme value handling failed")
                return False
        else:
            print(f"❌ Extreme value test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Extreme value test error: {e}")
        return False

def test_statistical_reasoning():
    """Test that reasoning includes statistical analysis"""
    print("Testing statistical reasoning...")
    try:
        prediction_data = {
            "player_name": "Burdol",
            "prop_type": "kills",
            "prop_value": 6.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            reasoning = data["reasoning"].lower()
            
            # Check for statistical analysis keywords
            statistical_indicators = [
                "trend", "volatility", "standard deviation", 
                "average", "recent form", "season average"
            ]
            
            if any(indicator in reasoning for indicator in statistical_indicators):
                print("✅ Statistical reasoning passed")
                return True
            else:
                print("❌ Statistical reasoning failed - no statistical indicators found")
                return False
        else:
            print(f"❌ Statistical reasoning test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Statistical reasoning test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Outscaled.gg API Test Suite")
    print("=" * 40)
    
    # Wait for service to be ready
    print("Waiting for service to be ready...")
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_player_search,
        test_prediction,
        test_extreme_values,
        test_statistical_reasoning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the API service.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 