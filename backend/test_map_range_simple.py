#!/usr/bin/env python3
"""
Simple test script to verify map range aggregation fix without database dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('.')

from app.utils.data_utils import create_map_index_column, aggregate_stats_by_map_range

def test_map_range_aggregation():
    """Test that map range aggregation produces different results than single map"""
    print("Testing map range aggregation fix...")
    
    try:
        # Create sample data that mimics Oracle's Elixir format
        sample_data = {
            'gameid': [
                'LOLTMNT03_179647',  # Series 1, Map 1
                'LOLTMNT03_179648',  # Series 1, Map 2  
                'LOLTMNT03_179649',  # Series 1, Map 3
                'LOLTMNT04_179650',  # Series 2, Map 1
                'LOLTMNT04_179651',  # Series 2, Map 2
                'LOLTMNT05_179652',  # Series 3, Map 1
            ],
            'playername': ['Garden'] * 6,
            'kills': [5, 7, 3, 4, 6, 8],  # Different kills per map
            'assists': [3, 5, 2, 4, 3, 7],
            'total cs': [200, 250, 180, 220, 240, 260],
            'result': [1, 0, 1, 1, 0, 1],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03']
        }
        
        df = pd.DataFrame(sample_data)
        print(f"Original data shape: {df.shape}")
        print(f"Sample data:\n{df.head()}")
        
        # Test map index creation
        df_with_index = create_map_index_column(df)
        print(f"\nAfter map index creation:")
        print(f"  match_series: {df_with_index['match_series'].unique()}")
        print(f"  map_index_within_series: {df_with_index['map_index_within_series'].value_counts().sort_index().to_dict()}")
        
        # Test single map filtering
        single_map_data = df_with_index[df_with_index['map_index_within_series'].isin([1])]
        single_map_avg_kills = single_map_data['kills'].mean()
        print(f"\nSingle map (Map 1) average kills: {single_map_avg_kills:.2f}")
        
        # Test map range aggregation
        map_range_data = aggregate_stats_by_map_range(df_with_index, [1, 2])
        print(f"\nMap range [1,2] aggregated data:")
        print(f"  Shape: {map_range_data.shape}")
        print(f"  Data:\n{map_range_data}")
        
        if not map_range_data.empty:
            # Calculate average kills per map from aggregated data
            total_kills = map_range_data['kills'].sum()
            total_maps = map_range_data['map_index_within_series'].sum()
            range_avg_kills = total_kills / total_maps if total_maps > 0 else 0
            print(f"  Total kills across maps: {total_kills}")
            print(f"  Total maps played: {total_maps}")
            print(f"  Average kills per map: {range_avg_kills:.2f}")
            
            # Check if the values are different
            if abs(single_map_avg_kills - range_avg_kills) > 0.01:
                print(f"\n✅ SUCCESS: Map range aggregation is working!")
                print(f"   Single map avg: {single_map_avg_kills:.2f}")
                print(f"   Map range avg: {range_avg_kills:.2f}")
                print(f"   Difference: {abs(single_map_avg_kills - range_avg_kills):.2f}")
            else:
                print(f"\n❌ FAILURE: Map range aggregation not working")
                print(f"   Single map avg: {single_map_avg_kills:.2f}")
                print(f"   Map range avg: {range_avg_kills:.2f}")
                print(f"   Difference: {abs(single_map_avg_kills - range_avg_kills):.2f}")
        else:
            print("❌ FAILURE: No aggregated data returned")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_map_range_aggregation() 