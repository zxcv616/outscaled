#!/usr/bin/env python3
"""
Test script to check Garden player data specifically
"""

import sys
import os
import pandas as pd
sys.path.append('.')

def test_garden_player():
    """Test Garden player data specifically"""
    print("Testing Garden player data...")
    
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
            print(f"Garden data shape: {garden_data.shape}")
            print(f"Sample Garden matches:")
            print(garden_data[['gameid', 'playername', 'kills', 'assists', 'total cs', 'result', 'date']].head(10))
            
            # Check for map series
            print(f"\nUnique gameids for Garden: {garden_data['gameid'].nunique()}")
            print(f"Sample gameids: {garden_data['gameid'].head(5).tolist()}")
            
            # Test map index creation
            from app.utils.data_utils import create_map_index_column
            garden_with_index = create_map_index_column(garden_data)
            print(f"\nAfter map index creation:")
            print(f"  match_series: {garden_with_index['match_series'].unique()}")
            print(f"  map_index_within_series: {garden_with_index['map_index_within_series'].value_counts().sort_index().to_dict()}")
            
            # Test single map filtering
            single_map_data = garden_with_index[garden_with_index['map_index_within_series'].isin([1])]
            print(f"\nSingle map (Map 1) data for Garden:")
            print(f"  Matches: {len(single_map_data)}")
            if len(single_map_data) > 0:
                print(f"  Avg kills: {single_map_data['kills'].mean():.2f}")
                print(f"  Avg assists: {single_map_data['assists'].mean():.2f}")
                print(f"  Avg CS: {single_map_data['total cs'].mean():.2f}")
            
            # Test map range aggregation
            from app.utils.data_utils import aggregate_stats_by_map_range
            map_range_data = aggregate_stats_by_map_range(garden_with_index, [1, 2])
            print(f"\nMap range [1,2] aggregated data for Garden:")
            print(f"  Shape: {map_range_data.shape}")
            if not map_range_data.empty:
                print(f"  Data:\n{map_range_data}")
                total_kills = map_range_data['kills'].sum()
                total_maps = map_range_data['map_index_within_series'].sum()
                avg_kills = total_kills / total_maps if total_maps > 0 else 0
                print(f"  Total kills: {total_kills}")
                print(f"  Total maps: {total_maps}")
                print(f"  Average kills per map: {avg_kills:.2f}")
            else:
                print("  No aggregated data found")
                
        else:
            print("❌ No Garden player data found!")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_garden_player() 