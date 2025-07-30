#!/usr/bin/env python3
"""
Test to debug the statistical functions error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_statistical_functions_with_real_data():
    """Test statistical functions with real player data"""
    try:
        from app.ml.predictor import PropPredictor
        from app.services.data_fetcher import DataFetcher
        
        predictor = PropPredictor()
        data_fetcher = DataFetcher()
        
        # Get real player data
        class MockDB:
            def __init__(self):
                pass
        
        mock_db = MockDB()
        player_stats = data_fetcher.get_player_stats(
            player_name="iBo",
            db=mock_db,
            map_range=[1]
        )
        
        print(f"Player stats keys: {list(player_stats.keys())}")
        print(f"Recent matches: {len(player_stats.get('recent_matches', []))}")
        
        prop_request = {
            "player_name": "iBo",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "",
            "tournament": "",
            "map_range": [1]
        }
        
        # Test probability distribution
        print("\nTesting probability distribution...")
        try:
            result = predictor.calculate_probability_distribution(
                player_stats=player_stats,
                prop_request=prop_request,
                range_std=5.0
            )
            print(f"✅ Success: {len(result.get('probability_distribution', {}))} values")
        except Exception as e:
            print(f"❌ Error in probability distribution: {e}")
            import traceback
            traceback.print_exc()
        
        # Test statistical insights
        print("\nTesting statistical insights...")
        try:
            result = predictor.get_statistical_insights(
                player_stats=player_stats,
                prop_request=prop_request
            )
            print(f"✅ Success: {result.get('statistical_measures', {}).get('z_score', 'N/A')}")
        except Exception as e:
            print(f"❌ Error in statistical insights: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Statistical Functions with Real Data")
    print("=" * 50)
    test_statistical_functions_with_real_data() 