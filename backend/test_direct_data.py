#!/usr/bin/env python3
"""
Test script to directly test data processing without database dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('.')

def test_direct_data_processing():
    """Test data processing directly without database dependencies"""
    print("Testing direct data processing...")
    
    try:
        # Load the 2025 data directly
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
            
            # Test single map filtering
            single_map_data = garden_with_index[garden_with_index['map_index_within_series'].isin([1])]
            single_map_avg_kills = single_map_data['kills'].mean()
            single_map_avg_assists = single_map_data['assists'].mean()
            single_map_avg_cs = single_map_data['total cs'].mean()
            print(f"\nSingle map (Map 1) averages:")
            print(f"  Avg kills: {single_map_avg_kills:.2f}")
            print(f"  Avg assists: {single_map_avg_assists:.2f}")
            print(f"  Avg CS: {single_map_avg_cs:.2f}")
            
            # Test map range aggregation
            map_range_data = aggregate_stats_by_map_range(garden_with_index, [1, 2])
            print(f"\nMap range [1,2] aggregated data:")
            print(f"  Shape: {map_range_data.shape}")
            if not map_range_data.empty:
                total_kills = map_range_data['kills'].sum()
                total_assists = map_range_data['assists'].sum()
                total_cs = map_range_data['total cs'].sum()
                total_maps = map_range_data['map_index_within_series'].sum()
                
                range_avg_kills = total_kills / total_maps if total_maps > 0 else 0
                range_avg_assists = total_assists / total_maps if total_maps > 0 else 0
                range_avg_cs = total_cs / total_maps if total_maps > 0 else 0
                
                print(f"  Total kills: {total_kills}")
                print(f"  Total assists: {total_assists}")
                print(f"  Total CS: {total_cs}")
                print(f"  Total maps: {total_maps}")
                print(f"  Average kills per map: {range_avg_kills:.2f}")
                print(f"  Average assists per map: {range_avg_assists:.2f}")
                print(f"  Average CS per map: {range_avg_cs:.2f}")
                
                # Check if the values are different
                kills_diff = abs(single_map_avg_kills - range_avg_kills)
                assists_diff = abs(single_map_avg_assists - range_avg_assists)
                cs_diff = abs(single_map_avg_cs - range_avg_cs)
                
                print(f"\n=== Comparison ===")
                print(f"Single map avg kills: {single_map_avg_kills:.2f}")
                print(f"Map range avg kills: {range_avg_kills:.2f}")
                print(f"Kills difference: {kills_diff:.2f}")
                
                print(f"Single map avg assists: {single_map_avg_assists:.2f}")
                print(f"Map range avg assists: {range_avg_assists:.2f}")
                print(f"Assists difference: {assists_diff:.2f}")
                
                print(f"Single map avg CS: {single_map_avg_cs:.2f}")
                print(f"Map range avg CS: {range_avg_cs:.2f}")
                print(f"CS difference: {cs_diff:.2f}")
                
                if kills_diff > 0.01 or assists_diff > 0.01 or cs_diff > 0.01:
                    print("✅ SUCCESS: Map range aggregation is working!")
                else:
                    print("❌ FAILURE: Map range aggregation not working")
                    
                # Check if we're getting actual data
                if single_map_avg_kills > 0 and range_avg_kills > 0:
                    print("✅ SUCCESS: Getting actual player data!")
                else:
                    print("❌ FAILURE: Not getting actual player data")
            else:
                print("❌ FAILURE: No aggregated data returned")
                
        else:
            print("❌ No Garden player data found!")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_data_processing() 