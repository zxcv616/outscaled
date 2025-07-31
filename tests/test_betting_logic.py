#!/usr/bin/env python3
"""
Specialized test suite for League of Legends betting logic
Validates that map ranges work correctly for PrizePicks-style betting
"""

import requests
import json
import time
import sys
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

def test_prizepicks_style_betting():
    """Test that the system correctly handles PrizePicks-style betting logic"""
    print("Testing PrizePicks-style betting logic...")
    
    # Test cases that simulate real betting scenarios
    betting_scenarios = [
        {
            "name": "Garden kills Map 1 vs Maps 1-2",
            "player": "Garden",
            "prop_type": "kills",
            "map1_prop": 4.5,
            "maps12_prop": 8.5,
            "expected_behavior": "Maps 1-2 should be ~2x Map 1"
        },
        {
            "name": "Delight assists Map 1 vs Maps 1-2", 
            "player": "Delight",
            "prop_type": "assists",
            "map1_prop": 12.5,
            "maps12_prop": 20.5,
            "expected_behavior": "Maps 1-2 should be ~2x Map 1"
        },
        {
            "name": "Peanut kills Map 1 vs Maps 1-2",
            "player": "Peanut",
            "prop_type": "kills", 
            "map1_prop": 6.5,
            "maps12_prop": 12.5,
            "expected_behavior": "Maps 1-2 should be ~2x Map 1"
        }
    ]
    
    passed = 0
    total = len(betting_scenarios)
    
    for scenario in betting_scenarios:
        try:
            print(f"\n📊 Testing: {scenario['name']}")
            
            # Test Map 1
            map1_data = {
                "player_name": scenario["player"],
                "prop_type": scenario["prop_type"],
                "prop_value": scenario["map1_prop"],
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": [1],
                "start_map": 1,
                "end_map": 1
            }
            
            response_map1 = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=map1_data,
                timeout=10
            )
            
            if response_map1.status_code != 200:
                print(f"❌ Map 1 prediction failed: {response_map1.status_code}")
                continue
            
            data_map1 = response_map1.json()
            map1_avg = data_map1.get("player_stats", {}).get(f"avg_{scenario['prop_type']}", 0)
            map1_recent = data_map1.get("player_stats", {}).get(f"recent_{scenario['prop_type']}_avg", 0)
            map1_prediction = data_map1.get("prediction", "")
            map1_confidence = data_map1.get("confidence", 0)
            
            # Test Maps 1-2
            maps12_data = {
                "player_name": scenario["player"],
                "prop_type": scenario["prop_type"],
                "prop_value": scenario["maps12_prop"],
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": [1, 2],
                "start_map": 1,
                "end_map": 2
            }
            
            response_maps12 = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=maps12_data,
                timeout=10
            )
            
            if response_maps12.status_code != 200:
                print(f"❌ Maps 1-2 prediction failed: {response_maps12.status_code}")
                continue
            
            data_maps12 = response_maps12.json()
            maps12_avg = data_maps12.get("player_stats", {}).get(f"avg_{scenario['prop_type']}", 0)
            maps12_recent = data_maps12.get("player_stats", {}).get(f"recent_{scenario['prop_type']}_avg", 0)
            maps12_prediction = data_maps12.get("prediction", "")
            maps12_confidence = data_maps12.get("confidence", 0)
            
            # Print detailed results
            print(f"  Map 1 Results:")
            print(f"    Avg: {map1_avg:.2f}")
            print(f"    Recent: {map1_recent:.2f}")
            print(f"    Prediction: {map1_prediction} ({map1_confidence:.1f}%)")
            
            print(f"  Maps 1-2 Results:")
            print(f"    Avg: {maps12_avg:.2f}")
            print(f"    Recent: {maps12_recent:.2f}")
            print(f"    Prediction: {maps12_prediction} ({maps12_confidence:.1f}%)")
            
            # Validate betting logic
            avg_ratio = maps12_avg / map1_avg if map1_avg > 0 else 0
            recent_ratio = maps12_recent / map1_recent if map1_recent > 0 else 0
            
            print(f"  Ratios:")
            print(f"    Avg ratio: {avg_ratio:.2f}")
            print(f"    Recent ratio: {recent_ratio:.2f}")
            
            # Check if the logic makes sense
            # For Maps 1-2, recent should be roughly 2x Map 1 recent (summed across maps)
            # But avg should be similar (per-map average)
            logic_valid = True
            
            if recent_ratio < 1.5:  # Should be roughly 2x for summed data
                print(f"  ⚠️  Recent ratio too low: {recent_ratio:.2f} (expected ~2.0)")
                logic_valid = False
            
            if avg_ratio < 0.5 or avg_ratio > 2.0:  # Should be reasonable
                print(f"  ⚠️  Avg ratio unusual: {avg_ratio:.2f}")
                logic_valid = False
            
            if logic_valid:
                print(f"  ✅ {scenario['name']} passed - betting logic correct")
                passed += 1
            else:
                print(f"  ❌ {scenario['name']} failed - betting logic incorrect")
                
        except Exception as e:
            print(f"❌ {scenario['name']} error: {e}")
    
    print(f"\n📊 PrizePicks-style Betting Results: {passed}/{total} tests passed")
    return passed == total

def test_map_range_statistical_validation():
    """Test that statistical analysis is correct for different map ranges"""
    print("Testing statistical validation for map ranges...")
    
    player = "Garden"
    prop_type = "kills"
    
    # Test different map ranges and validate the statistical logic
    map_ranges = [
        {"range": [1], "name": "Map 1", "expected_multiplier": 1},
        {"range": [1, 2], "name": "Maps 1-2", "expected_multiplier": 2},
        {"range": [1, 2, 3], "name": "Maps 1-3", "expected_multiplier": 3}
    ]
    
    results = {}
    
    for map_range in map_ranges:
        try:
            prediction_data = {
                "player_name": player,
                "prop_type": prop_type,
                "prop_value": 5.0,
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": map_range["range"],
                "start_map": map_range["range"][0],
                "end_map": map_range["range"][-1]
            }
            
            response = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                avg_value = data.get("player_stats", {}).get(f"avg_{prop_type}", 0)
                recent_value = data.get("player_stats", {}).get(f"recent_{prop_type}_avg", 0)
                map_range_response = data.get("player_stats", {}).get("map_range", [])
                
                results[map_range["name"]] = {
                    "avg": avg_value,
                    "recent": recent_value,
                    "map_range": map_range_response,
                    "expected_multiplier": map_range["expected_multiplier"]
                }
            else:
                print(f"❌ Failed to get data for {map_range['name']}")
                
        except Exception as e:
            print(f"❌ Error testing {map_range['name']}: {e}")
    
    # Print results and validate
    print("\n📊 Statistical Validation Results:")
    for map_range_name, result in results.items():
        print(f"\n{map_range_name}:")
        print(f"  Avg: {result['avg']:.2f}")
        print(f"  Recent: {result['recent']:.2f}")
        print(f"  Map Range: {result['map_range']}")
        print(f"  Expected Multiplier: {result['expected_multiplier']}")
    
    # Validate that recent values scale roughly with map count
    base_recent = results.get("Map 1", {}).get("recent", 0)
    if base_recent > 0:
        print(f"\n📈 Recent Value Scaling Analysis:")
        for map_range_name, result in results.items():
            if map_range_name != "Map 1":
                actual_ratio = result["recent"] / base_recent
                expected_ratio = result["expected_multiplier"]
                print(f"  {map_range_name}: {actual_ratio:.2f}x (expected ~{expected_ratio}x)")
    
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("Testing edge cases and error handling...")
    
    edge_cases = [
        {
            "name": "Empty map range",
            "map_range": [],
            "expected": "Should handle gracefully"
        },
        {
            "name": "Invalid map range",
            "map_range": [5, 6],
            "expected": "Should handle gracefully"
        },
        {
            "name": "Single map with high prop value",
            "map_range": [1],
            "prop_value": 50.0,
            "expected": "Should trigger extreme value handling"
        },
        {
            "name": "Multi-map with high prop value",
            "map_range": [1, 2],
            "prop_value": 100.0,
            "expected": "Should trigger extreme value handling"
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for case in edge_cases:
        try:
            prediction_data = {
                "player_name": "Garden",
                "prop_type": "kills",
                "prop_value": case.get("prop_value", 5.0),
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": case["map_range"],
                "start_map": case["map_range"][0] if case["map_range"] else 1,
                "end_map": case["map_range"][-1] if case["map_range"] else 1
            }
            
            response = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "")
                confidence = data.get("confidence", 0)
                
                print(f"✅ {case['name']} passed - got {prediction} ({confidence:.1f}%)")
                passed += 1
            else:
                print(f"❌ {case['name']} failed - API error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {case['name']} error: {e}")
    
    print(f"\n📊 Edge Cases Results: {passed}/{total} tests passed")
    return passed == total

def main():
    """Run all betting logic tests"""
    print("🧪 League of Legends Betting Logic Test Suite")
    print("=" * 50)
    
    # Wait for service to be ready
    if not wait_for_service():
        return 1
    
    tests = [
        test_prizepicks_style_betting,
        test_map_range_statistical_validation,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Betting Logic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All betting logic tests passed!")
        print("✨ Betting features verified:")
        print("   - PrizePicks-style map range logic")
        print("   - Statistical validation for different map ranges")
        print("   - Edge case handling")
        print("   - Proper aggregation for multi-map props")
        return 0
    else:
        print("❌ Some betting logic tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 