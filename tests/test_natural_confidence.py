#!/usr/bin/env python3
"""
Test Natural Confidence Levels Without Thresholds
This test compares confidence levels with and without extreme value handling.
"""

import requests
import json
import time
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def wait_for_service():
    """Wait for the service to be ready"""
    print("🐳 Waiting for Docker service to be ready...")
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            time.sleep(2)
    
    print("❌ Service failed to start")
    return False

def test_natural_confidence():
    """Test if confidence levels emerge naturally without thresholds"""
    print("\n🧪 Testing Natural Confidence Levels")
    print("=" * 60)
    
    test_cases = [
        {"player": "Garden", "prop_value": 0.5, "description": "Very low prop for high-average player"},
        {"player": "Garden", "prop_value": 1.0, "description": "Low prop for high-average player"},
        {"player": "Garden", "prop_value": 3.0, "description": "Average prop for high-average player"},
        {"player": "Garden", "prop_value": 5.0, "description": "High prop for high-average player"},
        {"player": "Garden", "prop_value": 10.0, "description": "Very high prop for high-average player"},
        {"player": "Shadow", "prop_value": 0.5, "description": "Very low prop for low-average player"},
        {"player": "Shadow", "prop_value": 1.0, "description": "Low prop for low-average player"},
        {"player": "Shadow", "prop_value": 2.0, "description": "High prop for low-average player"},
        {"player": "Shadow", "prop_value": 5.0, "description": "Very high prop for low-average player"},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📊 Testing {case['description']}:")
        print(f"  Player: {case['player']}")
        print(f"  Prop Value: {case['prop_value']}")
        
        payload = {
            "player_name": case["player"],
            "prop_type": "kills",
            "prop_value": case["prop_value"],
            "opponent": "Test Team",
            "tournament": "LCS",
            "map_range": [1],
            "start_map": 1,
            "end_map": 1
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/predict",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "UNKNOWN")
                confidence = data.get("confidence", 0.0)
                reasoning = data.get("reasoning", "")
                rule_override = data.get("rule_override", False)
                
                # Extract recent average from reasoning
                recent_avg = None
                if "Recent kills average (" in reasoning:
                    try:
                        start = reasoning.find("Recent kills average (") + 22
                        end = reasoning.find(")", start)
                        recent_avg = float(reasoning[start:end])
                    except:
                        pass
                
                results.append({
                    "player": case["player"],
                    "prop_value": case["prop_value"],
                    "prediction": prediction,
                    "confidence": confidence,
                    "recent_avg": recent_avg,
                    "rule_override": rule_override,
                    "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                })
                
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {confidence:.1f}%")
                print(f"  Recent Avg: {recent_avg}")
                print(f"  Rule Override: {rule_override}")
                print(f"  Reasoning: {reasoning[:80]}...")
                
                # Analyze if confidence makes sense
                if recent_avg:
                    if recent_avg > case["prop_value"]:
                        if prediction == "MORE" and confidence > 70:
                            print("  ✅ High confidence MORE for low prop - NATURAL")
                        elif prediction == "MORE" and confidence < 70:
                            print("  ⚠️  Low confidence MORE for low prop - UNNATURAL")
                        else:
                            print("  ❓ Unexpected prediction")
                    elif recent_avg < case["prop_value"]:
                        if prediction == "LESS" and confidence > 70:
                            print("  ✅ High confidence LESS for high prop - NATURAL")
                        elif prediction == "LESS" and confidence < 70:
                            print("  ⚠️  Low confidence LESS for high prop - UNNATURAL")
                        else:
                            print("  ❓ Unexpected prediction")
                    else:
                        print("  📊 Average prop value - moderate confidence expected")
                
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return results

def analyze_natural_confidence(results):
    """Analyze if confidence levels emerge naturally"""
    print("\n🔍 Natural Confidence Analysis")
    print("=" * 60)
    
    # Group results by player
    player_results = {}
    for result in results:
        player = result["player"]
        if player not in player_results:
            player_results[player] = []
        player_results[player].append(result)
    
    # Analyze each player
    for player, player_data in player_results.items():
        print(f"\n📈 {player} Analysis:")
        
        # Find recent average
        recent_avg = None
        for result in player_data:
            if result.get("recent_avg") is not None:
                recent_avg = result["recent_avg"]
                break
        
        if recent_avg is None:
            print("  ❌ Could not determine recent average")
            continue
        
        print(f"  Recent Average: {recent_avg:.1f} kills")
        
        # Analyze each result for this player
        for result in player_data:
            prop_value = result["prop_value"]
            prediction = result["prediction"]
            confidence = result["confidence"]
            rule_override = result["rule_override"]
            
            print(f"  Prop {prop_value}: {prediction} @ {confidence:.1f}% (override: {rule_override})")
            
            # Check if confidence is natural
            if recent_avg > prop_value:
                # Recent avg above prop value
                if prediction == "MORE" and confidence > 70:
                    print("    ✅ Natural high confidence MORE")
                elif prediction == "MORE" and confidence < 70:
                    print("    ⚠️  Unnatural low confidence MORE")
                else:
                    print("    ❓ Unexpected prediction")
            elif recent_avg < prop_value:
                # Recent avg below prop value
                if prediction == "LESS" and confidence > 70:
                    print("    ✅ Natural high confidence LESS")
                elif prediction == "LESS" and confidence < 70:
                    print("    ⚠️  Unnatural low confidence LESS")
                else:
                    print("    ❓ Unexpected prediction")
    
    # Check rule override frequency
    override_count = sum(1 for r in results if r.get("rule_override", False))
    total_count = len(results)
    
    print(f"\n📊 Rule Override Analysis:")
    print(f"  Total predictions: {total_count}")
    print(f"  Rule overrides: {override_count}")
    print(f"  Override rate: {override_count/total_count*100:.1f}%")
    
    if override_count > total_count * 0.5:
        print("  ⚠️  High override rate - thresholds may be too aggressive")
    else:
        print("  ✅ Reasonable override rate")

def main():
    """Main test function"""
    print("🧪 Natural Confidence Test Suite")
    print("=" * 60)
    
    # Wait for service
    if not wait_for_service():
        return False
    
    # Test natural confidence levels
    results = test_natural_confidence()
    
    # Analyze results
    analyze_natural_confidence(results)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    # Check if confidence levels are natural
    natural_count = 0
    total_count = len(results)
    
    for result in results:
        recent_avg = result.get("recent_avg")
        if recent_avg:
            prop_value = result["prop_value"]
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            if recent_avg > prop_value and prediction == "MORE" and confidence > 70:
                natural_count += 1
            elif recent_avg < prop_value and prediction == "LESS" and confidence > 70:
                natural_count += 1
            elif abs(recent_avg - prop_value) < 0.5 and confidence < 80:
                natural_count += 1
    
    natural_rate = natural_count / total_count if total_count > 0 else 0
    
    if natural_rate > 0.7:
        print("✅ Confidence levels appear to emerge naturally!")
        print("✅ Most predictions have appropriate confidence levels")
        return True
    else:
        print("❌ Confidence levels may be artificially controlled!")
        print("❌ Many predictions have unexpected confidence levels")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 