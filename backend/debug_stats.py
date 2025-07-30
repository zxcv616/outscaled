#!/usr/bin/env python3
"""
Debug script for statistical analysis functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_scipy_import():
    """Test scipy imports"""
    try:
        from scipy.stats import norm, percentileofscore
        print("✅ Scipy imports working")
        return True
    except Exception as e:
        print(f"❌ Scipy import error: {e}")
        return False

def test_predictor_import():
    """Test predictor import"""
    try:
        from app.ml.predictor import PropPredictor
        print("✅ Predictor import working")
        return True
    except Exception as e:
        print(f"❌ Predictor import error: {e}")
        return False

def test_data_fetcher_import():
    """Test data fetcher import"""
    try:
        from app.services.data_fetcher import DataFetcher
        print("✅ DataFetcher import working")
        return True
    except Exception as e:
        print(f"❌ DataFetcher import error: {e}")
        return False

def test_statistical_functions():
    """Test statistical functions directly"""
    try:
        from app.ml.predictor import PropPredictor
        from app.services.data_fetcher import DataFetcher
        
        predictor = PropPredictor()
        data_fetcher = DataFetcher()
        
        # Create mock data
        mock_player_stats = {
            "player_name": "iBo",
            "recent_matches": [
                {"kills": 0, "assists": 5, "cs": 234, "win": False},
                {"kills": 9, "assists": 6, "cs": 275, "win": True},
                {"kills": 3, "assists": 6, "cs": 289, "win": False},
                {"kills": 3, "assists": 9, "cs": 228, "win": False},
                {"kills": 1, "assists": 1, "cs": 163, "win": False}
            ],
            "avg_kills": 3.2,
            "avg_assists": 5.4,
            "avg_cs": 237.8,
            "win_rate": 0.2,
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
        print("\n📊 Testing Probability Distribution...")
        distribution_result = predictor.calculate_probability_distribution(
            player_stats=mock_player_stats,
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
            player_stats=mock_player_stats,
            prop_request=prop_request
        )
        
        if "error" in insights_result:
            print(f"❌ Error: {insights_result['error']}")
        else:
            print(f"✅ Statistical insights calculated")
            print(f"   Z-Score: {insights_result['statistical_measures']['z_score']}")
            print(f"   Significance: {insights_result['statistical_measures']['significance_description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in statistical functions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Debugging Statistical Analysis")
    print("=" * 40)
    
    # Test imports
    scipy_ok = test_scipy_import()
    predictor_ok = test_predictor_import()
    data_fetcher_ok = test_data_fetcher_import()
    
    if scipy_ok and predictor_ok and data_fetcher_ok:
        print("\n✅ All imports working, testing functions...")
        test_statistical_functions()
    else:
        print("\n❌ Import issues detected") 