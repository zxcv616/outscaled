#!/usr/bin/env python3
"""
Simple test script for statistical analysis features (no database required)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.predictor import PropPredictor

def test_statistical_analysis_simple():
    """Test the statistical analysis features with mock data"""
    
    print("🧪 Testing Statistical Analysis Features (Simple)")
    print("=" * 60)
    
    # Initialize predictor
    predictor = PropPredictor()
    
    # Create mock player stats (similar to iBo's data)
    mock_player_stats = {
        "player_name": "iBo",
        "recent_matches": [
            {"kills": 0, "assists": 5, "cs": 234, "win": False},
            {"kills": 9, "assists": 6, "cs": 275, "win": True},
            {"kills": 3, "assists": 6, "cs": 289, "win": False},
            {"kills": 3, "assists": 9, "cs": 228, "win": False},
            {"kills": 1, "assists": 1, "cs": 163, "win": False},
            {"kills": 5, "assists": 10, "cs": 316, "win": True},
            {"kills": 5, "assists": 3, "cs": 207, "win": False},
            {"kills": 11, "assists": 6, "cs": 249, "win": True},
            {"kills": 4, "assists": 5, "cs": 261, "win": False},
            {"kills": 3, "assists": 3, "cs": 197, "win": False}
        ],
        "avg_kills": 3.8,
        "avg_assists": 5.4,
        "avg_cs": 240.9,
        "recent_kills_avg": 3.2,
        "recent_assists_avg": 5.4,
        "recent_cs_avg": 237.8,
        "win_rate": 0.3,
        "data_source": "oracles_elixir"
    }
    
    # Test prop request
    prop_request = {
        "player_name": "iBo",
        "prop_type": "kills",
        "prop_value": 6.0,
        "opponent": "",
        "tournament": "",
        "map_range": [1]
    }
    
    print(f"📊 Testing with {mock_player_stats['player_name']} - {prop_request['prop_type']} {prop_request['prop_value']}")
    print()
    
    # Test 1: Probability Distribution
    print("1️⃣ Testing Probability Distribution")
    print("-" * 40)
    
    distribution_result = predictor.calculate_probability_distribution(
        player_stats=mock_player_stats,
        prop_request=prop_request,
        range_std=3.0
    )
    
    if "error" in distribution_result:
        print(f"❌ Error: {distribution_result['error']}")
    else:
        print(f"✅ Analysis range: {distribution_result['analysis_range']['min_value']}-{distribution_result['analysis_range']['max_value']}")
        print(f"📈 Recent average: {distribution_result['summary_stats']['mean_recent']}")
        print(f"📊 Standard deviation: {distribution_result['summary_stats']['std_recent']}")
        print(f"🎯 Input z-score: {distribution_result['summary_stats']['input_z_score']}")
        print()
        
        # Show probability distribution for key values
        prob_dist = distribution_result['probability_distribution']
        print("📊 Probability Distribution (key values):")
        for value in [1, 3, 5, 6, 7, 9, 11]:
            if value in prob_dist:
                data = prob_dist[value]
                significance = "🔴" if data['statistical_significance'] == 'high' else "🟡" if data['statistical_significance'] == 'medium' else "🟢"
                print(f"  {significance} {value} kills: {data['prediction']} ({data['confidence']:.1f}% confidence)")
                print(f"     MORE: {data['probability_more']:.1f}% | LESS: {data['probability_less']:.1f}%")
                print(f"     Within 95% CI: {'✅' if data['within_95_ci'] else '❌'}")
    
    print()
    
    # Test 2: Statistical Insights
    print("2️⃣ Testing Statistical Insights")
    print("-" * 40)
    
    insights_result = predictor.get_statistical_insights(
        player_stats=mock_player_stats,
        prop_request=prop_request
    )
    
    if "error" in insights_result:
        print(f"❌ Error: {insights_result['error']}")
    else:
        stats = insights_result['statistical_measures']
        prob_analysis = insights_result['probability_analysis']
        volatility = insights_result['volatility_metrics']
        trend = insights_result['trend_analysis']
        
        print(f"📊 Z-Score: {stats['z_score']}")
        print(f"📈 Percentile: {stats['percentile']:.1f}%")
        print(f"🎯 Significance: {stats['significance_description']}")
        print(f"📊 MORE probability: {prob_analysis['probability_more']:.1f}%")
        print(f"📊 LESS probability: {prob_analysis['probability_less']:.1f}%")
        print(f"📊 Recommended: {prob_analysis['recommended_prediction']}")
        print(f"📊 Confidence level: {prob_analysis['confidence_level']}")
        print(f"📊 Volatility: {volatility['volatility_percentage']:.1f}% ({'High' if volatility['high_volatility'] else 'Moderate' if volatility['moderate_volatility'] else 'Low'})")
        print(f"📈 Trend: {trend['direction']}")
        
        # Show confidence intervals
        ci_95 = insights_result['confidence_intervals']['95_percent']
        print(f"📊 95% CI: [{ci_95['lower']:.1f}, {ci_95['upper']:.1f}] (width: {ci_95['width']:.1f})")
    
    print()
    
    # Test 3: Regular Prediction (for comparison)
    print("3️⃣ Testing Regular Prediction")
    print("-" * 40)
    
    prediction_result = predictor.predict(
        player_stats=mock_player_stats,
        prop_request=prop_request,
        verbose=True
    )
    
    print(f"🎯 Prediction: {prediction_result['prediction']}")
    print(f"📊 Confidence: {prediction_result['confidence']:.1f}%")
    print(f"📝 Reasoning: {prediction_result['reasoning'][:120]}...")
    print(f"🔧 Model mode: {prediction_result.get('model_mode', 'unknown')}")
    print(f"⚡ Rule override: {prediction_result.get('rule_override', False)}")
    
    print()
    print("✅ Statistical analysis test completed!")
    print()
    print("🎯 Key Benefits of Statistical Analysis:")
    print("   • Shows probability for range of values (not just input)")
    print("   • Provides statistical confidence levels")
    print("   • Identifies significant vs non-significant predictions")
    print("   • Shows volatility and trend analysis")
    print("   • Gives percentile rankings")

if __name__ == "__main__":
    test_statistical_analysis_simple() 