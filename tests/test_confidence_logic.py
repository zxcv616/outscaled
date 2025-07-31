#!/usr/bin/env python3
"""
Test Confidence Logic for Faker Kills 1-5
This test systematically checks if confidence levels make logical sense.
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

def test_faker_confidence_levels():
    """Test confidence levels for Faker kills 1-5"""
    print("\n🧪 Testing Faker Confidence Levels (Kills 1-5)")
    print("=" * 60)
    
    results = []
    
    for prop_value in range(1, 6):
        print(f"\n📊 Testing Faker kills {prop_value}:")
        
        # Make prediction request
        payload = {
            "player_name": "Faker",
            "prop_type": "kills",
            "prop_value": prop_value,
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
                    "prop_value": prop_value,
                    "prediction": prediction,
                    "confidence": confidence,
                    "recent_avg": recent_avg,
                    "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                })
                
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {confidence:.1f}%")
                print(f"  Recent Avg: {recent_avg}")
                print(f"  Reasoning: {reasoning[:80]}...")
                
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                results.append({
                    "prop_value": prop_value,
                    "error": f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "prop_value": prop_value,
                "error": str(e)
            })
    
    return results

def analyze_confidence_logic(results):
    """Analyze if confidence levels make logical sense"""
    print("\n🔍 Confidence Logic Analysis")
    print("=" * 60)
    
    # Find recent average from results
    recent_avg = None
    for result in results:
        if result.get("recent_avg") is not None:
            recent_avg = result["recent_avg"]
            break
    
    if recent_avg is None:
        print("❌ Could not determine recent average")
        return False
    
    print(f"📈 Faker's Recent Average: {recent_avg:.1f} kills")
    print()
    
    # Analyze each result
    issues_found = []
    
    for result in results:
        if "error" in result:
            continue
            
        prop_value = result["prop_value"]
        prediction = result["prediction"]
        confidence = result["confidence"]
        
        print(f"🎯 Prop Value {prop_value}:")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.1f}%")
        
        # Logic checks
        if recent_avg > prop_value:
            # Recent average is above prop value
            if prediction == "MORE":
                # Should be MORE with higher confidence
                if confidence < 60:
                    issues_found.append(f"Prop {prop_value}: MORE prediction with low confidence ({confidence:.1f}%) when recent avg ({recent_avg:.1f}) > prop value")
                else:
                    print(f"   ✅ MORE prediction with {confidence:.1f}% confidence - LOGICAL")
            else:
                # Should be LESS with lower confidence
                if confidence > 70:
                    issues_found.append(f"Prop {prop_value}: LESS prediction with high confidence ({confidence:.1f}%) when recent avg ({recent_avg:.1f}) > prop value")
                else:
                    print(f"   ✅ LESS prediction with {confidence:.1f}% confidence - LOGICAL")
        elif recent_avg < prop_value:
            # Recent average is below prop value
            if prediction == "LESS":
                # Should be LESS with higher confidence
                if confidence < 60:
                    issues_found.append(f"Prop {prop_value}: LESS prediction with low confidence ({confidence:.1f}%) when recent avg ({recent_avg:.1f}) < prop value")
                else:
                    print(f"   ✅ LESS prediction with {confidence:.1f}% confidence - LOGICAL")
            else:
                # Should be MORE with lower confidence
                if confidence > 70:
                    issues_found.append(f"Prop {prop_value}: MORE prediction with high confidence ({confidence:.1f}%) when recent avg ({recent_avg:.1f}) < prop value")
                else:
                    print(f"   ✅ MORE prediction with {confidence:.1f}% confidence - LOGICAL")
        else:
            # Recent average equals prop value
            if confidence > 80:
                issues_found.append(f"Prop {prop_value}: High confidence ({confidence:.1f}%) when recent avg ({recent_avg:.1f}) = prop value")
            else:
                print(f"   ✅ Moderate confidence ({confidence:.1f}%) when recent avg = prop value - LOGICAL")
        
        print()
    
    # Check for logical progression
    print("📊 Confidence Progression Analysis:")
    confidences = [r["confidence"] for r in results if "error" not in r]
    if len(confidences) >= 2:
        # Check if confidence increases as prop value moves further from recent average
        distance_from_avg = [abs(r["prop_value"] - recent_avg) for r in results if "error" not in r]
        
        # Sort by distance from average
        sorted_data = sorted(zip(distance_from_avg, confidences), key=lambda x: x[0])
        sorted_confidences = [c for _, c in sorted_data]
        
        # Check if confidence generally increases with distance
        increasing_count = sum(1 for i in range(1, len(sorted_confidences)) 
                             if sorted_confidences[i] >= sorted_confidences[i-1])
        
        if increasing_count >= len(sorted_confidences) * 0.6:  # At least 60% should increase
            print("   ✅ Confidence generally increases with distance from recent average")
        else:
            issues_found.append("Confidence does not follow logical progression with distance from recent average")
            print("   ❌ Confidence progression is illogical")
    
    # Report issues
    if issues_found:
        print("\n❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"   • {issue}")
        return False
    else:
        print("\n✅ All confidence levels appear logical!")
        return True

def test_edge_cases():
    """Test edge cases for confidence logic"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        {"prop_value": 0.5, "description": "Very low prop value"},
        {"prop_value": 6, "description": "High prop value"},
        {"prop_value": 10, "description": "Very high prop value"}
    ]
    
    for case in edge_cases:
        print(f"\n📊 Testing {case['description']}:")
        
        payload = {
            "player_name": "Faker",
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
                
                print(f"  Prop Value: {case['prop_value']}")
                print(f"  Prediction: {prediction}")
                print(f"  Confidence: {confidence:.1f}%")
                
                # Check if confidence makes sense for extreme values
                if case["prop_value"] <= 1:
                    if prediction == "MORE" and confidence > 70:
                        print("  ✅ High confidence MORE for low prop value - LOGICAL")
                    else:
                        print("  ⚠️  Unexpected result for low prop value")
                elif case["prop_value"] >= 6:
                    if prediction == "LESS" and confidence > 80:
                        print("  ✅ High confidence LESS for high prop value - LOGICAL")
                    else:
                        print("  ⚠️  Unexpected result for high prop value")
                        
            else:
                print(f"  ❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main test function"""
    print("🧪 Confidence Logic Test Suite")
    print("=" * 60)
    
    # Wait for service
    if not wait_for_service():
        return False
    
    # Test main confidence levels
    results = test_faker_confidence_levels()
    
    # Analyze logic
    logic_ok = analyze_confidence_logic(results)
    
    # Test edge cases
    test_edge_cases()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    if logic_ok:
        print("✅ Confidence logic appears to be working correctly!")
        print("✅ All confidence levels follow logical patterns")
        return True
    else:
        print("❌ Confidence logic issues detected!")
        print("❌ Some confidence levels don't make statistical sense")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 