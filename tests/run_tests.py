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

def test_map_range_aggregation():
    """Test that map range aggregation produces different results than single map"""
    print("Testing map range aggregation...")
    try:
        # Test single map prediction
        prediction_data_single = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1],
            "start_map": 1,
            "end_map": 1
        }
        
        response_single = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data_single,
            timeout=10
        )
        
        if response_single.status_code != 200:
            print(f"❌ Single map prediction failed: {response_single.status_code}")
            return False
        
        data_single = response_single.json()
        single_avg_kills = data_single.get("player_stats", {}).get("avg_kills", 0)
        single_recent_kills = data_single.get("player_stats", {}).get("recent_kills_avg", 0)
        
        # Test map range prediction
        prediction_data_range = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response_range = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data_range,
            timeout=10
        )
        
        if response_range.status_code != 200:
            print(f"❌ Map range prediction failed: {response_range.status_code}")
            return False
        
        data_range = response_range.json()
        range_avg_kills = data_range.get("player_stats", {}).get("avg_kills", 0)
        range_recent_kills = data_range.get("player_stats", {}).get("recent_kills_avg", 0)
        
        # Check if we're getting actual data
        if single_avg_kills == 0 or range_avg_kills == 0:
            print("❌ Map range aggregation failed - getting 0.0 values")
            return False
        
        # Check if the values are different
        kills_diff = abs(single_avg_kills - range_avg_kills)
        recent_kills_diff = abs(single_recent_kills - range_recent_kills)
        
        print(f"Single map avg kills: {single_avg_kills:.2f}")
        print(f"Map range avg kills: {range_avg_kills:.2f}")
        print(f"Kills difference: {kills_diff:.2f}")
        
        if kills_diff > 0.01 or recent_kills_diff > 0.01:
            print("✅ Map range aggregation is working!")
            return True
        else:
            print("❌ Map range aggregation not working - values are identical")
            return False
            
    except Exception as e:
        print(f"❌ Map range aggregation test error: {e}")
        return False

def test_verbose_prediction():
    """Test verbose prediction mode with detailed reasoning"""
    print("Testing verbose prediction mode...")
    try:
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        # Test with verbose=True
        response = requests.post(
            f"{API_BASE}/predict?verbose=true",
            headers={"Content-Type": "application/json"},
            json=prediction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            reasoning = data.get("reasoning", "").lower()
            
            # Check for verbose indicators
            verbose_indicators = [
                "coefficient of variation", "analysis based on", 
                "using", "model uses", "engineered features"
            ]
            
            if any(indicator in reasoning for indicator in verbose_indicators):
                print("✅ Verbose prediction mode passed")
                return True
            else:
                print("❌ Verbose prediction mode failed - no verbose indicators")
                return False
        else:
            print(f"❌ Verbose prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Verbose prediction test error: {e}")
        return False

def test_model_transparency():
    """Test model transparency features (model_mode, rule_override, scaler_status)"""
    print("Testing model transparency features...")
    try:
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
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
            
            # Check for transparency fields
            transparency_fields = ["model_mode", "rule_override", "scaler_status"]
            missing_fields = [field for field in transparency_fields if field not in data]
            
            if not missing_fields:
                print("✅ Model transparency features present")
                print(f"  Model mode: {data.get('model_mode', 'unknown')}")
                print(f"  Rule override: {data.get('rule_override', False)}")
                print(f"  Scaler status: {data.get('scaler_status', 'unknown')}")
                return True
            else:
                print(f"❌ Model transparency failed - missing fields: {missing_fields}")
                return False
        else:
            print(f"❌ Model transparency test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model transparency test error: {e}")
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
            # Updated: Current implementation uses statistical analysis and caps confidence at 95.0
            if data["prediction"] == "LESS" and data["confidence"] >= 90.0:
                print("✅ Extreme value handling passed")
                return True
            else:
                print(f"❌ Extreme value handling failed - got {data['prediction']} with {data['confidence']:.1f}% confidence")
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
                "player_name": "Garden",
                "prop_type": prop_type,
                "prop_value": 5.0,
                "opponent": "Hanwha Life Esports Challengers",
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
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 6.5,
            "opponent": "Hanwha Life Esports Challengers",
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
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "Hanwha Life Esports Challengers",
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
        test_map_range_aggregation,  # NEW: Test map range aggregation
        test_verbose_prediction,      # NEW: Test verbose mode
        test_model_transparency,      # NEW: Test transparency features
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
        print("   - Map range aggregation (NEW)")
        print("   - Verbose prediction mode (NEW)")
        print("   - Model transparency features (NEW)")
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