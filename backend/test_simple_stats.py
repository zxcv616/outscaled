#!/usr/bin/env python3
"""
Simple test to isolate the statistical endpoint issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direct_import():
    """Test direct imports"""
    try:
        from app.ml.predictor import PropPredictor
        from app.services.data_fetcher import DataFetcher
        print("✅ Direct imports work")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_simple_stats():
    """Test simple statistical functions"""
    try:
        from app.ml.predictor import PropPredictor
        from app.services.data_fetcher import DataFetcher
        
        predictor = PropPredictor()
        data_fetcher = DataFetcher()
        
        # Simple test data
        player_stats = {
            "player_name": "iBo",
            "recent_matches": [{"kills": 3, "assists": 5, "cs": 200, "win": True}],
            "avg_kills": 3.0,
            "avg_assists": 5.0,
            "avg_cs": 200.0,
            "win_rate": 0.5,
            "data_source": "oracles_elixir"
        }
        
        prop_request = {
            "player_name": "iBo",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "",
            "tournament": "",
            "map_range": [1]
        }
        
        # Test probability distribution
        print("Testing probability distribution...")
        result = predictor.calculate_probability_distribution(
            player_stats=player_stats,
            prop_request=prop_request,
            range_std=5.0
        )
        print(f"✅ Probability distribution: {len(result.get('probability_distribution', {}))} values")
        
        # Test statistical insights
        print("Testing statistical insights...")
        result = predictor.get_statistical_insights(
            player_stats=player_stats,
            prop_request=prop_request
        )
        print(f"✅ Statistical insights: {result.get('statistical_measures', {}).get('z_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in simple stats: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Simple Statistical Test")
    print("=" * 30)
    
    if test_direct_import():
        test_simple_stats()
    else:
        print("❌ Import failed") 