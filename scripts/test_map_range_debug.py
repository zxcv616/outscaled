#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.data_utils import create_map_index_column, aggregate_stats_by_map_range, filter_by_map_range
from app.services.data_fetcher import DataFetcher

def test_map_range_aggregation():
    """Test map range aggregation for YoungJae data"""
    
    print("=== MAP RANGE AGGREGATION DEBUG ===\n")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Get player data for YoungJae
    player_name = "YoungJae"
    
    # Get all matches for YoungJae
    player_matches = data_fetcher.df[data_fetcher.df["playername"].str.lower() == player_name.lower()]
    
    if player_matches.empty:
        print(f"No data found for player: {player_name}")
        return
    
    print(f"Found {len(player_matches)} total matches for {player_name}")
    
    # Create map index
    player_matches_with_index = create_map_index_column(player_matches)
    
    print("\n=== MAP INDEX CREATION ===")
    print("Sample of map indices:")
    sample_data = player_matches_with_index[['gameid', 'playername', 'kills', 'map_index_within_series', 'match_series']].head(10)
    print(sample_data.to_string())
    
    print(f"\nMap index distribution:")
    print(player_matches_with_index['map_index_within_series'].value_counts().sort_index())
    
    print(f"\nMatch series distribution:")
    print(player_matches_with_index['match_series'].value_counts().head())
    
    # Test Map 1 filtering
    print("\n=== MAP 1 FILTERING ===")
    map1_data = filter_by_map_range(player_matches_with_index, [1])
    print(f"Map 1 matches: {len(map1_data)}")
    print("Sample Map 1 data:")
    print(map1_data[['gameid', 'playername', 'kills', 'map_index_within_series']].head().to_string())
    
    # Test Maps 1-2 aggregation
    print("\n=== MAPS 1-2 AGGREGATION ===")
    maps12_aggregated = aggregate_stats_by_map_range(player_matches_with_index, [1, 2])
    print(f"Maps 1-2 aggregated series: {len(maps12_aggregated)}")
    print("Sample aggregated data:")
    print(maps12_aggregated[['playername', 'match_series', 'kills', 'map_index_within_series']].head().to_string())
    
    # Test Maps 1-3 aggregation
    print("\n=== MAPS 1-3 AGGREGATION ===")
    maps13_aggregated = aggregate_stats_by_map_range(player_matches_with_index, [1, 2, 3])
    print(f"Maps 1-3 aggregated series: {len(maps13_aggregated)}")
    print("Sample aggregated data:")
    print(maps13_aggregated[['playername', 'match_series', 'kills', 'map_index_within_series']].head().to_string())
    
    # Compare the data
    print("\n=== COMPARISON ===")
    print(f"Map 1 only: {len(map1_data)} matches")
    print(f"Maps 1-2: {len(maps12_aggregated)} series")
    print(f"Maps 1-3: {len(maps13_aggregated)} series")
    
    # Check if Maps 1-2 includes Map 3 data
    print("\n=== CHECKING FOR MAP 3 CONTAMINATION ===")
    maps12_filtered = filter_by_map_range(player_matches_with_index, [1, 2])
    print(f"Maps 1-2 filtered (before aggregation): {len(maps12_filtered)} matches")
    print("Map indices in Maps 1-2 filtered data:")
    print(maps12_filtered['map_index_within_series'].value_counts().sort_index())
    
    # Check recent matches for Maps 1-2
    print("\n=== RECENT MATCHES FOR MAPS 1-2 ===")
    recent_maps12 = maps12_aggregated.sort_values("date", ascending=False).head(5)
    print("Recent Maps 1-2 aggregated data:")
    print(recent_maps12[['playername', 'match_series', 'kills', 'assists', 'map_index_within_series']].to_string())
    
    # Check individual map data for recent series
    print("\n=== INDIVIDUAL MAP DATA FOR RECENT SERIES ===")
    recent_series = recent_maps12['match_series'].tolist()
    individual_maps = player_matches_with_index[player_matches_with_index['match_series'].isin(recent_series)]
    individual_maps = individual_maps.sort_values(['match_series', 'map_index_within_series'])
    print("Individual map data for recent series:")
    print(individual_maps[['match_series', 'map_index_within_series', 'kills', 'assists']].to_string())

if __name__ == "__main__":
    test_map_range_aggregation() 