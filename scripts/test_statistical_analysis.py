#!/usr/bin/env python3
"""
Test script for the new statistical analysis features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.predictor import PropPredictor
from app.services.data_fetcher import DataFetcher

def test_statistical_analysis():
    """Test the new statistical analysis features"""
    
    print("🧪 Testing Statistical Analysis Features")
    print("=" * 50)
    
    # Initialize components
    predictor = PropPredictor()
    data_fetcher = DataFetcher()
    
    # Test with iBo (known player with good data)
    player_name = "iBo"
    prop_type = "kills"
    prop_value = 6.0
    
    print(f"📊 Testing with {player_name} - {prop_type} {prop_value}")
    print()
    
    # Get player stats
    player_stats = data_fetcher.get_player_stats(
        player_name=player_name,
        opponent="",
        tournament="",
        map_range=[1]
    )
    
    if not player_stats:
        print("❌ No data found for player")
        return
    
    # Create prop request
    prop_request = {
        "player_name": player_name,
        "prop_type": prop_type,
        "prop_value": prop_value,
        "opponent": "",
        "tournament": "",
        "map_range": [1]
    }
    
    # Test 1: Probability Distribution
    print("1️⃣ Testing Probability Distribution")
    print("-" * 30)
    
    distribution_result = predictor.calculate_probability_distribution(
        player_stats=player_stats,
        prop_request=prop_request,
        range_std=3.0  # Smaller range for testing
    )
    
    if "error" in distribution_result:
        print(f"❌ Error: {distribution_result['error']}")
    else:
        print(f"✅ Analysis range: {distribution_result['analysis_range']['min_value']}-{distribution_result['analysis_range']['max_value']}")
        print(f"📈 Recent average: {distribution_result['summary_stats']['mean_recent']}")
        print(f"📊 Standard deviation: {distribution_result['summary_stats']['std_recent']}")
        print(f"🎯 Input z-score: {distribution_result['summary_stats']['input_z_score']}")
        print()
        
        # Show some key values
        prob_dist = distribution_result['probability_distribution']
        print("Key probability values:")
        for value in [1, 3, 5, 6, 7, 9, 11]:
            if value in prob_dist:
                data = prob_dist[value]
                print(f"  {value} kills: {data['prediction']} ({data['confidence']:.1f}% confidence)")
                print(f"    - MORE: {data['probability_more']:.1f}% | LESS: {data['probability_less']:.1f}%")
    
    print()
    
    # Test 2: Statistical Insights
    print("2️⃣ Testing Statistical Insights")
    print("-" * 30)
    
    insights_result = predictor.get_statistical_insights(
        player_stats=player_stats,
        prop_request=prop_request
    )
    
    if "error" in insights_result:
        print(f"❌ Error: {insights_result['error']}")
    else:
        stats = insights_result['statistical_measures']
        prob_analysis = insights_result['probability_analysis']
        volatility = insights_result['volatility_metrics']
        
        print(f"📊 Z-Score: {stats['z_score']}")
        print(f"📈 Percentile: {stats['percentile']:.1f}%")
        print(f"🎯 Significance: {stats['significance_description']}")
        print(f"📊 MORE probability: {prob_analysis['probability_more']:.1f}%")
        print(f"📊 LESS probability: {prob_analysis['probability_less']:.1f}%")
        print(f"📊 Recommended: {prob_analysis['recommended_prediction']}")
        print(f"📊 Volatility: {volatility['volatility_percentage']:.1f}% ({'High' if volatility['high_volatility'] else 'Moderate' if volatility['moderate_volatility'] else 'Low'})")
    
    print()
    
    # Test 3: Regular Prediction (for comparison)
    print("3️⃣ Testing Regular Prediction")
    print("-" * 30)
    
    prediction_result = predictor.predict(
        player_stats=player_stats,
        prop_request=prop_request,
        verbose=True
    )
    
    print(f"🎯 Prediction: {prediction_result['prediction']}")
    print(f"📊 Confidence: {prediction_result['confidence']:.1f}%")
    print(f"📝 Reasoning: {prediction_result['reasoning'][:100]}...")
    
    print()
    print("✅ Statistical analysis test completed!")

if __name__ == "__main__":
    test_statistical_analysis() 