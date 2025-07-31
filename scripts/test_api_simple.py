#!/usr/bin/env python3
"""
Simple API test to debug the statistical analysis endpoint
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.predictor import PropPredictor
from app.services.data_fetcher import DataFetcher

def test_api_simple():
    """Test the API components directly"""
    
    print("🧪 Testing API Components")
    print("=" * 40)
    
    # Initialize components
    predictor = PropPredictor()
    data_fetcher = DataFetcher()
    
    # Test player exists
    player_name = "iBo"
    print(f"Testing player: {player_name}")
    
    # Test player_exists
    exists = data_fetcher.player_exists(player_name)
    print(f"Player exists: {exists}")
    
    if not exists:
        print("❌ Player not found")
        return
    
    # Test get_player_stats (without db)
    try:
        # Create a mock db session
        class MockDB:
            def __init__(self):
                pass
        
        mock_db = MockDB()
        player_stats = data_fetcher.get_player_stats(player_name, mock_db, map_range=[1])
        print(f"✅ Player stats loaded: {len(player_stats.get('recent_matches', []))} recent matches")
        
        # Test statistical analysis
        prop_request = {
            "player_name": player_name,
            "prop_type": "kills",
            "prop_value": 6.0,
            "opponent": "",
            "tournament": "",
            "map_range": [1]
        }
        
        # Test probability distribution
        print("\n📊 Testing Probability Distribution...")
        distribution_result = predictor.calculate_probability_distribution(
            player_stats=player_stats,
            prop_request=prop_request,
            range_std=5.0
        )
        
        if "error" in distribution_result:
            print(f"❌ Error: {distribution_result['error']}")
        else:
            print(f"✅ Probability distribution calculated")
            print(f"   Range: {distribution_result['analysis_range']['min_value']}-{distribution_result['analysis_range']['max_value']}")
            print(f"   Values analyzed: {len(distribution_result['probability_distribution'])}")
        
        # Test statistical insights
        print("\n🎯 Testing Statistical Insights...")
        insights_result = predictor.get_statistical_insights(
            player_stats=player_stats,
            prop_request=prop_request
        )
        
        if "error" in insights_result:
            print(f"❌ Error: {insights_result['error']}")
        else:
            print(f"✅ Statistical insights calculated")
            print(f"   Z-Score: {insights_result['statistical_measures']['z_score']}")
            print(f"   Significance: {insights_result['statistical_measures']['significance_description']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_simple() 