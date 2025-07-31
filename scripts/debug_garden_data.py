#!/usr/bin/env python3
"""
Debug script to check Garden player data in web API context
"""

import sys
import os
import pandas as pd
sys.path.append('.')

def debug_garden_data():
    """Debug Garden player data specifically"""
    print("Debugging Garden player data...")
    
    try:
        # Load the 2025 data
        df = pd.read_csv("2025_LoL_esports_match_data_from_OraclesElixir.csv", low_memory=False)
        print(f"Full dataset shape: {df.shape}")
        
        # Filter for complete data
        df_complete = df[df["datacompleteness"] == "complete"]
        print(f"Complete data shape: {df_complete.shape}")
        
        # Find Garden player data
        garden_data = df_complete[df_complete["playername"].str.lower() == "garden"]
        print(f"Garden player matches found: {len(garden_data)}")
        
        if len(garden_data) > 0:
            # Test map index creation
            from app.utils.data_utils import create_map_index_column, aggregate_stats_by_map_range
            
            garden_with_index = create_map_index_column(garden_data)
            print(f"\nAfter map index creation:")
            print(f"  match_series: {garden_with_index['match_series'].unique()}")
            print(f"  map_index_within_series: {garden_with_index['map_index_within_series'].value_counts().sort_index().to_dict()}")
            
            # Check what map ranges are available
            print(f"\nAvailable map indices: {sorted(garden_with_index['map_index_within_series'].unique())}")
            
            # Test single map filtering
            single_map_data = garden_with_index[garden_with_index['map_index_within_series'].isin([1])]
            print(f"\nSingle map (Map 1) data for Garden:")
            print(f"  Matches: {len(single_map_data)}")
            if len(single_map_data) > 0:
                print(f"  Avg kills: {single_map_data['kills'].mean():.2f}")
                print(f"  Avg assists: {single_map_data['assists'].mean():.2f}")
                print(f"  Avg CS: {single_map_data['total cs'].mean():.2f}")
            
            # Test map range aggregation for [1, 2]
            print(f"\nTesting map range [1, 2] aggregation:")
            map_range_data = aggregate_stats_by_map_range(garden_with_index, [1, 2])
            print(f"  Aggregated data shape: {map_range_data.shape}")
            
            if not map_range_data.empty:
                print(f"  Aggregated data:\n{map_range_data}")
                total_kills = map_range_data['kills'].sum()
                total_maps = map_range_data['map_index_within_series'].sum()
                avg_kills = total_kills / total_maps if total_maps > 0 else 0
                print(f"  Total kills: {total_kills}")
                print(f"  Total maps: {total_maps}")
                print(f"  Average kills per map: {avg_kills:.2f}")
            else:
                print("  ❌ No aggregated data found for [1, 2]")
                
                # Check what map ranges are actually available
                print(f"\nChecking available map ranges:")
                available_maps = sorted(garden_with_index['map_index_within_series'].unique())
                print(f"  Available map indices: {available_maps}")
                
                # Test with available map ranges
                for map_range in [[1], [1, 2], [1, 2, 3]]:
                    test_data = aggregate_stats_by_map_range(garden_with_index, map_range)
                    print(f"  Map range {map_range}: {len(test_data)} series found")
                
        else:
            print("❌ No Garden player data found!")
            
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_garden_data() 