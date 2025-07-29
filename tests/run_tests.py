#!/usr/bin/env python3
"""
Comprehensive test runner for Outscaled.gg API
Updated for Docker deployment and new features including map-range support.
"""

import requests
import json
import time
import sys
import subprocess
import os

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

def wait_for_service():
    """Wait for the Docker service to be ready"""
    print("🐳 Waiting for Docker service to be ready...")
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        attempt += 1
        time.sleep(2)
        if attempt % 5 == 0:
            print(f"⏳ Still waiting... ({attempt}/{max_attempts})")
    
    print("❌ Service failed to start within timeout")
    return False

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

def test_teams_endpoint():
    """Test teams endpoint for opponent selection"""
    print("Testing teams endpoint...")
    try:
        response = requests.get(f"{API_BASE}/teams", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "teams" in data and len(data["teams"]) > 0:
                print(f"✅ Teams endpoint passed - found {len(data['teams'])} teams")
                return True
            else:
                print("❌ Teams endpoint failed - no teams found")
                return False
        else:
            print(f"❌ Teams endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Teams endpoint error: {e}")
        return False

def test_basic_prediction():
    """Test basic prediction functionality"""
    print("Testing basic prediction...")
    try:
        prediction_data = {
            "player_name": "Smash",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "T1",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data and "confidence" in data and "reasoning" in data:
                print(f"✅ Basic prediction passed - {data['prediction']} with {data['confidence']:.1f}% confidence")
                return True
            else:
                print("❌ Basic prediction failed - missing required fields")
                return False
        else:
            print(f"❌ Basic prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Basic prediction error: {e}")
        return False

def test_map_range_prediction():
    """Test map-range specific prediction (PrizePicks style)"""
    print("Testing map-range prediction...")
    try:
        prediction_data = {
            "player_name": "Smash",
            "prop_type": "kills",
            "prop_value": 6.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            reasoning = data.get("reasoning", "").lower()
            
            # Check for map-range indicators
            if "maps 1-2" in reasoning or "map range" in reasoning:
                print("✅ Map-range prediction passed")
                return True
            else:
                print("❌ Map-range prediction failed - no map range indicators")
                return False
        else:
            print(f"❌ Map-range prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Map-range prediction error: {e}")
        return False

def test_extreme_values():
    """Test prediction with extreme values"""
    print("Testing extreme value handling...")
    try:
        prediction_data = {
            "player_name": "Smash",
            "prop_type": "kills",
            "prop_value": 999,
            "opponent": "T1",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data["prediction"] == "LESS" and data["confidence"] >= 99.0:
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

def test_different_prop_types():
    """Test different prop types"""
    print("Testing different prop types...")
    prop_types = ["kills", "assists", "cs", "deaths", "gold", "damage"]
    passed = 0
    
    for prop_type in prop_types:
        try:
            prediction_data = {
                "player_name": "Smash",
                "prop_type": prop_type,
                "prop_value": 5.0,
                "opponent": "T1",
                "tournament": "LCS",
                "map_range": [1, 2],
                "start_map": 1,
                "end_map": 2
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
                    passed += 1
                else:
                    print(f"❌ {prop_type} prop type failed")
            else:
                print(f"❌ {prop_type} prop type failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {prop_type} prop type error: {e}")
    
    if passed == len(prop_types):
        print(f"✅ All prop types passed ({passed}/{len(prop_types)})")
        return True
    else:
        print(f"❌ Some prop types failed ({passed}/{len(prop_types)})")
        return False

def test_statistical_reasoning():
    """Test that reasoning includes statistical analysis"""
    print("Testing statistical reasoning...")
    try:
        prediction_data = {
            "player_name": "Smash",
            "prop_type": "kills",
            "prop_value": 6.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
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
                "average", "recent form", "season average",
                "confidence", "prediction based"
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

def test_model_info():
    """Test model information endpoints"""
    print("Testing model info endpoints...")
    try:
        # Test model info
        response = requests.get(f"{API_BASE}/model/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "model_type" in data:
                print("✅ Model info endpoint passed")
            else:
                print("❌ Model info endpoint failed - missing model_type")
                return False
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
        
        # Test feature importance
        response = requests.get(f"{API_BASE}/model/features", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "feature_importance" in data:
                print("✅ Feature importance endpoint passed")
                return True
            else:
                print("❌ Feature importance endpoint failed - missing feature_importance")
                return False
        else:
            print(f"❌ Feature importance endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info endpoints error: {e}")
        return False

def test_deterministic_predictions():
    """Test that predictions are deterministic (same inputs = same outputs)"""
    print("Testing deterministic predictions...")
    try:
        prediction_data = {
            "player_name": "Smash",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "T1",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        # Make two identical requests
        response1 = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        response2 = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            if (data1["prediction"] == data2["prediction"] and 
                abs(data1["confidence"] - data2["confidence"]) < 0.1):
                print("✅ Deterministic predictions passed")
                return True
            else:
                print("❌ Deterministic predictions failed - different results")
                return False
        else:
            print("❌ Deterministic predictions failed - request errors")
            return False
            
    except Exception as e:
        print(f"❌ Deterministic predictions error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Outscaled.gg API Test Suite (Updated)")
    print("=" * 50)
    
    # Check if Docker is running
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if "outscaled" not in result.stdout:
            print("🐳 Starting Docker containers...")
            subprocess.run(["docker-compose", "up", "-d"], check=True)
    except Exception as e:
        print(f"❌ Docker error: {e}")
        return 1
    
    # Wait for service to be ready
    if not wait_for_service():
        return 1
    
    tests = [
        test_health_check,
        test_player_search,
        test_teams_endpoint,
        test_basic_prediction,
        test_map_range_prediction,
        test_extreme_values,
        test_different_prop_types,
        test_statistical_reasoning,
        test_model_info,
        test_deterministic_predictions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        print("✨ Features tested:")
        print("   - Health check and service readiness")
        print("   - Player search with autocomplete")
        print("   - Teams endpoint for opponent selection")
        print("   - Basic predictions with confidence")
        print("   - Map-range support (PrizePicks style)")
        print("   - Extreme value handling")
        print("   - All prop types (kills, assists, cs, deaths, gold, damage)")
        print("   - Statistical reasoning analysis")
        print("   - Model information and feature importance")
        print("   - Deterministic predictions")
        return 0
    else:
        print("❌ Some tests failed. Please check the API service.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 