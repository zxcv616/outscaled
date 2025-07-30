#!/usr/bin/env python3
"""
Test script to check data loading and player availability
"""

import sys
import os
import pandas as pd
sys.path.append('.')

def test_data_loading():
    """Test if the CSV data is loading correctly"""
    print("Testing data loading...")
    
    try:
        # Check if CSV files exist
        csv_files = [
            "2025_LoL_esports_match_data_from_OraclesElixir.csv",
            "2024_LoL_esports_match_data_from_OraclesElixir.csv"
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"✅ Found {csv_file}")
                # Try to load a small sample
                df = pd.read_csv(csv_file, nrows=1000)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {df.columns.tolist()}")
                
                # Check for playername column
                if 'playername' in df.columns:
                    players = df['playername'].dropna().unique()
                    print(f"   Sample players: {players[:10].tolist()}")
                    
                    # Look for Garden
                    garden_players = [p for p in players if 'garden' in str(p).lower()]
                    print(f"   Garden players found: {garden_players}")
                else:
                    print(f"   ❌ No 'playername' column found")
            else:
                print(f"❌ {csv_file} not found")
        
        # Try to load full dataset
        print("\nTrying to load full dataset...")
        try:
            df_2025 = pd.read_csv("2025_LoL_esports_match_data_from_OraclesElixir.csv")
            print(f"2025 data shape: {df_2025.shape}")
            
            if 'playername' in df_2025.columns:
                all_players_2025 = df_2025['playername'].dropna().unique()
                garden_players_2025 = [p for p in all_players_2025 if 'garden' in str(p).lower()]
                print(f"Garden players in 2025: {garden_players_2025}")
                
                # Show some sample players
                print(f"Sample players from 2025: {all_players_2025[:20].tolist()}")
            else:
                print("No playername column in 2025 data")
                
        except Exception as e:
            print(f"Error loading 2025 data: {e}")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading() 