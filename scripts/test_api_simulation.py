#!/usr/bin/env python3
"""
Test script to simulate API call for Garden player
"""

import sys
import os
import pandas as pd
sys.path.append('.')

def test_api_simulation():
    """Simulate the actual API call for Garden player"""
    print("Testing API simulation for Garden player...")
    
    try:
        # Simulate the data fetcher without database dependency
        from app.services.data_fetcher import DataFetcher
        
        # Create a mock database session
        class MockDB:
            pass
        
        db = MockDB()
        
        # Test single map (Map 1)
        print("\n=== Testing Map 1 ===")
        stats1 = DataFetcher().get_player_stats("Garden", db, [1])
        print(f"Single map stats:")
        print(f"  Avg kills: {stats1.get('avg_kills', 0):.2f}")
        print(f"  Recent kills avg: {stats1.get('recent_kills_avg', 0):.2f}")
        print(f"  Avg assists: {stats1.get('avg_assists', 0):.2f}")
        print(f"  Recent assists avg: {stats1.get('recent_assists_avg', 0):.2f}")
        print(f"  Avg CS: {stats1.get('avg_cs', 0):.2f}")
        print(f"  Recent CS avg: {stats1.get('recent_cs_avg', 0):.2f}")
        print(f"  Win rate: {stats1.get('win_rate', 0):.2f}")
        print(f"  Total matches: {stats1.get('total_matches_available', 0)}")
        print(f"  Matches in range: {stats1.get('matches_in_range', 0)}")
        print(f"  Data source: {stats1.get('data_source', 'none')}")
        
        # Test map range (Maps 1-2)
        print("\n=== Testing Maps 1-2 ===")
        stats2 = DataFetcher().get_player_stats("Garden", db, [1, 2])
        print(f"Map range stats:")
        print(f"  Avg kills: {stats2.get('avg_kills', 0):.2f}")
        print(f"  Recent kills avg: {stats2.get('recent_kills_avg', 0):.2f}")
        print(f"  Avg assists: {stats2.get('avg_assists', 0):.2f}")
        print(f"  Recent assists avg: {stats2.get('recent_assists_avg', 0):.2f}")
        print(f"  Avg CS: {stats2.get('avg_cs', 0):.2f}")
        print(f"  Recent CS avg: {stats2.get('recent_cs_avg', 0):.2f}")
        print(f"  Win rate: {stats2.get('win_rate', 0):.2f}")
        print(f"  Total matches: {stats2.get('total_matches_available', 0)}")
        print(f"  Matches in range: {stats2.get('matches_in_range', 0)}")
        print(f"  Data source: {stats2.get('data_source', 'none')}")
        
        # Check if the values are different
        single_avg = stats1.get('avg_kills', 0)
        range_avg = stats2.get('avg_kills', 0)
        
        print(f"\n=== Comparison ===")
        print(f"Single map avg kills: {single_avg:.2f}")
        print(f"Map range avg kills: {range_avg:.2f}")
        print(f"Difference: {abs(single_avg - range_avg):.2f}")
        
        if abs(single_avg - range_avg) > 0.01:
            print("✅ SUCCESS: Map range aggregation is working!")
        else:
            print("❌ FAILURE: Map range aggregation not working")
            
        # Check if we're getting actual data
        if single_avg > 0 and range_avg > 0:
            print("✅ SUCCESS: Getting actual player data!")
        else:
            print("❌ FAILURE: Not getting actual player data")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_simulation() 