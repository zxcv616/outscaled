#!/usr/bin/env python3
"""
Test script to verify map range aggregation fix
"""

import sys
import os
sys.path.append('.')

from app.services.data_fetcher import DataFetcher

def test_map_range_aggregation():
    """Test that map range aggregation produces different results than single map"""
    print("Testing map range aggregation fix...")
    
    try:
        # Initialize data fetcher
        df = DataFetcher()
        
        # Test with a known player
        player_name = "Garden"
        
        # Get stats for single map
        stats1 = df.get_player_stats(player_name, None, [1])
        print(f"Single map stats for {player_name}:")
        print(f"  Avg kills: {stats1.get('avg_kills', 0):.2f}")
        print(f"  Recent kills avg: {stats1.get('recent_kills_avg', 0):.2f}")
        print(f"  Total matches: {stats1.get('total_matches_available', 0)}")
        print(f"  Matches in range: {stats1.get('matches_in_range', 0)}")
        
        # Get stats for map range
        stats2 = df.get_player_stats(player_name, None, [1, 2])
        print(f"\nMap range [1,2] stats for {player_name}:")
        print(f"  Avg kills: {stats2.get('avg_kills', 0):.2f}")
        print(f"  Recent kills avg: {stats2.get('recent_kills_avg', 0):.2f}")
        print(f"  Total matches: {stats2.get('total_matches_available', 0)}")
        print(f"  Matches in range: {stats2.get('matches_in_range', 0)}")
        
        # Check if the values are different
        single_avg = stats1.get('avg_kills', 0)
        range_avg = stats2.get('avg_kills', 0)
        
        if abs(single_avg - range_avg) > 0.01:
            print(f"\n✅ SUCCESS: Map range aggregation is working!")
            print(f"   Single map avg: {single_avg:.2f}")
            print(f"   Map range avg: {range_avg:.2f}")
            print(f"   Difference: {abs(single_avg - range_avg):.2f}")
        else:
            print(f"\n❌ FAILURE: Map range aggregation not working")
            print(f"   Single map avg: {single_avg:.2f}")
            print(f"   Map range avg: {range_avg:.2f}")
            print(f"   Difference: {abs(single_avg - range_avg):.2f}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_map_range_aggregation() 