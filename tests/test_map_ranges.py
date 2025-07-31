#!/usr/bin/env python3
"""
Comprehensive test suite for map range functionality
Tests the betting logic for different map ranges (Map 1, Maps 1-2, Maps 1-3)
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

def test_single_map_vs_multi_map_differences():
    """Test that single map and multi-map predictions produce different results"""
    print("Testing single map vs multi-map differences...")
    
    test_cases = [
        {
            "player": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "description": "Garden kills test"
        },
        {
            "player": "Delight", 
            "prop_type": "assists",
            "prop_value": 15.5,
            "description": "Delight assists test"
        },
        {
            "player": "Peanut",
            "prop_type": "kills", 
            "prop_value": 8.5,
            "description": "Peanut kills test"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        try:
            # Test single map
            prediction_data_single = {
                "player_name": test_case["player"],
                "prop_type": test_case["prop_type"],
                "prop_value": test_case["prop_value"],
                "opponent": "Test Team",
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
                print(f"❌ Single map prediction failed for {test_case['description']}: {response_single.status_code}")
                continue
            
            data_single = response_single.json()
            single_avg = data_single.get("player_stats", {}).get(f"avg_{test_case['prop_type']}", 0)
            single_recent = data_single.get("player_stats", {}).get(f"recent_{test_case['prop_type']}_avg", 0)
            
            # Test multi-map
            prediction_data_multi = {
                "player_name": test_case["player"],
                "prop_type": test_case["prop_type"],
                "prop_value": test_case["prop_value"],
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": [1, 2],
                "start_map": 1,
                "end_map": 2
            }
            
            response_multi = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data_multi,
                timeout=10
            )
            
            if response_multi.status_code != 200:
                print(f"❌ Multi-map prediction failed for {test_case['description']}: {response_multi.status_code}")
                continue
            
            data_multi = response_multi.json()
            multi_avg = data_multi.get("player_stats", {}).get(f"avg_{test_case['prop_type']}", 0)
            multi_recent = data_multi.get("player_stats", {}).get(f"recent_{test_case['prop_type']}_avg", 0)
            
            # Check if we're getting actual data
            if single_avg == 0 or multi_avg == 0:
                print(f"❌ {test_case['description']} failed - getting 0.0 values")
                continue
            
            # Check for differences
            avg_diff = abs(single_avg - multi_avg)
            recent_diff = abs(single_recent - multi_recent)
            
            print(f"\n{test_case['description']}:")
            print(f"  Single map avg: {single_avg:.2f}")
            print(f"  Multi-map avg: {multi_avg:.2f}")
            print(f"  Single map recent: {single_recent:.2f}")
            print(f"  Multi-map recent: {multi_recent:.2f}")
            print(f"  Avg difference: {avg_diff:.2f}")
            print(f"  Recent difference: {recent_diff:.2f}")
            
            if avg_diff > 0.01 or recent_diff > 0.01:
                print(f"  ✅ {test_case['description']} passed - different values")
                passed += 1
            else:
                print(f"  ❌ {test_case['description']} failed - identical values")
                
        except Exception as e:
            print(f"❌ {test_case['description']} error: {e}")
    
    print(f"\n📊 Single vs Multi-map Results: {passed}/{total} tests passed")
    return passed == total

def test_map_range_consistency():
    """Test that map ranges are consistent across different prop types"""
    print("Testing map range consistency across prop types...")
    
    player = "Garden"
    map_ranges = [
        {"range": [1], "name": "Map 1"},
        {"range": [1, 2], "name": "Maps 1-2"},
        {"range": [1, 2, 3], "name": "Maps 1-3"}
    ]
    
    prop_types = ["kills", "assists", "cs"]
    
    results = {}
    
    for map_range in map_ranges:
        results[map_range["name"]] = {}
        
        for prop_type in prop_types:
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
                    
                    results[map_range["name"]][prop_type] = {
                        "avg": avg_value,
                        "recent": recent_value
                    }
                else:
                    print(f"❌ Failed to get data for {map_range['name']} {prop_type}")
                    
            except Exception as e:
                print(f"❌ Error testing {map_range['name']} {prop_type}: {e}")
    
    # Print results
    print("\n📊 Map Range Consistency Results:")
    for map_range_name, prop_results in results.items():
        print(f"\n{map_range_name}:")
        for prop_type, values in prop_results.items():
            print(f"  {prop_type}: avg={values['avg']:.2f}, recent={values['recent']:.2f}")
    
    # Check for consistency (different values across map ranges)
    consistent = True
    for prop_type in prop_types:
        values = []
        for map_range_name in results.keys():
            if prop_type in results[map_range_name]:
                values.append(results[map_range_name][prop_type]["avg"])
        
        if len(set(values)) < 2:  # All values are the same
            print(f"❌ {prop_type} values are identical across map ranges")
            consistent = False
    
    if consistent:
        print("✅ Map range consistency test passed")
    else:
        print("❌ Map range consistency test failed")
    
    return consistent

def test_betting_scenarios():
    """Test realistic betting scenarios"""
    print("Testing realistic betting scenarios...")
    
    scenarios = [
        {
            "name": "Single Map Kill Prop",
            "player": "Garden",
            "prop_type": "kills",
            "prop_value": 4.5,
            "map_range": [1],
            "expected": "Should use Map 1 data only"
        },
        {
            "name": "Multi-Map Kill Prop", 
            "player": "Garden",
            "prop_type": "kills",
            "prop_value": 8.5,
            "map_range": [1, 2],
            "expected": "Should use combined Maps 1-2 data"
        },
        {
            "name": "Single Map Assist Prop",
            "player": "Delight",
            "prop_type": "assists",
            "prop_value": 12.5,
            "map_range": [1],
            "expected": "Should use Map 1 data only"
        },
        {
            "name": "Multi-Map Assist Prop",
            "player": "Delight", 
            "prop_type": "assists",
            "prop_value": 20.5,
            "map_range": [1, 2],
            "expected": "Should use combined Maps 1-2 data"
        }
    ]
    
    passed = 0
    total = len(scenarios)
    
    for scenario in scenarios:
        try:
            prediction_data = {
                "player_name": scenario["player"],
                "prop_type": scenario["prop_type"],
                "prop_value": scenario["prop_value"],
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": scenario["map_range"],
                "start_map": scenario["map_range"][0],
                "end_map": scenario["map_range"][-1]
            }
            
            response = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check that map_range is correctly set in response
                response_map_range = data.get("player_stats", {}).get("map_range", [])
                if response_map_range == scenario["map_range"]:
                    print(f"✅ {scenario['name']} passed - map range correct")
                    passed += 1
                else:
                    print(f"❌ {scenario['name']} failed - map range mismatch")
                    print(f"  Expected: {scenario['map_range']}, Got: {response_map_range}")
            else:
                print(f"❌ {scenario['name']} failed - API error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {scenario['name']} error: {e}")
    
    print(f"\n📊 Betting Scenarios Results: {passed}/{total} tests passed")
    return passed == total

def test_extreme_map_ranges():
    """Test extreme map ranges and edge cases"""
    print("Testing extreme map ranges and edge cases...")
    
    edge_cases = [
        {
            "name": "Map 1 Only",
            "map_range": [1],
            "expected_behavior": "Single map analysis"
        },
        {
            "name": "Maps 1-2",
            "map_range": [1, 2], 
            "expected_behavior": "Two map aggregation"
        },
        {
            "name": "Maps 1-3",
            "map_range": [1, 2, 3],
            "expected_behavior": "Three map aggregation"
        },
        {
            "name": "Maps 2-3",
            "map_range": [2, 3],
            "expected_behavior": "Later maps only"
        }
    ]
    
    passed = 0
    total = len(edge_cases)
    
    for case in edge_cases:
        try:
            prediction_data = {
                "player_name": "Garden",
                "prop_type": "kills",
                "prop_value": 5.0,
                "opponent": "Test Team",
                "tournament": "LCS",
                "map_range": case["map_range"],
                "start_map": case["map_range"][0],
                "end_map": case["map_range"][-1]
            }
            
            response = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                response_map_range = data.get("player_stats", {}).get("map_range", [])
                
                if response_map_range == case["map_range"]:
                    print(f"✅ {case['name']} passed")
                    passed += 1
                else:
                    print(f"❌ {case['name']} failed - map range mismatch")
            else:
                print(f"❌ {case['name']} failed - API error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {case['name']} error: {e}")
    
    print(f"\n📊 Extreme Map Ranges Results: {passed}/{total} tests passed")
    return passed == total

def main():
    """Run all map range tests"""
    print("🧪 Map Range Functionality Test Suite")
    print("=" * 50)
    
    # Wait for service to be ready
    if not wait_for_service():
        return 1
    
    tests = [
        test_single_map_vs_multi_map_differences,
        test_map_range_consistency,
        test_betting_scenarios,
        test_extreme_map_ranges
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Map Range Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All map range tests passed! Betting logic is working correctly.")
        print("✨ Features verified:")
        print("   - Single map vs multi-map differentiation")
        print("   - Map range consistency across prop types")
        print("   - Realistic betting scenarios")
        print("   - Extreme map range edge cases")
        return 0
    else:
        print("❌ Some map range tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 