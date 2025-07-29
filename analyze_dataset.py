#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the benefits of combining 2024 and 2025 Oracle's Elixir datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_datasets():
    """Analyze the benefits of combining 2024 and 2025 datasets"""
    
    print("=== Oracle's Elixir Dataset Analysis ===\n")
    
    # Load both datasets
    datasets = {}
    for year in [2024, 2025]:
        filename = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        try:
            df = pd.read_csv(filename, low_memory=False)
            # Filter for complete data only
            df = df[df["datacompleteness"] == "complete"]
            df['data_year'] = str(year)
            datasets[year] = df
            print(f"✓ Loaded {year} dataset: {len(df)} complete matches")
        except FileNotFoundError:
            print(f"✗ {filename} not found")
            datasets[year] = pd.DataFrame()
    
    # Analyze individual datasets
    print("\n--- Individual Dataset Analysis ---")
    for year, df in datasets.items():
        if not df.empty:
            print(f"\n{year} Dataset:")
            print(f"  Total matches: {len(df):,}")
            print(f"  Unique players: {df['playername'].nunique():,}")
            print(f"  Unique teams: {df['teamname'].nunique():,}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Leagues: {df['league'].nunique()} unique leagues")
    
    # Combine datasets
    print("\n--- Combined Dataset Analysis ---")
    all_dataframes = [df for df in datasets.values() if not df.empty]
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Remove duplicates based on gameid and playername
        combined_df = combined_df.drop_duplicates(subset=['gameid', 'playername'], keep='first')
        
        print(f"\nCombined Dataset:")
        print(f"  Total matches: {len(combined_df):,}")
        print(f"  Unique players: {combined_df['playername'].nunique():,}")
        print(f"  Unique teams: {combined_df['teamname'].nunique():,}")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"  Leagues: {combined_df['league'].nunique()} unique leagues")
        
        # Data year distribution
        year_dist = combined_df['data_year'].value_counts().sort_index()
        print(f"  Data year distribution:")
        for year, count in year_dist.items():
            print(f"    {year}: {count:,} matches")
        
        # Player overlap analysis
        print(f"\n--- Player Overlap Analysis ---")
        players_2024 = set(datasets[2024]['playername'].unique()) if not datasets[2024].empty else set()
        players_2025 = set(datasets[2025]['playername'].unique()) if not datasets[2025].empty else set()
        
        overlap = players_2024.intersection(players_2025)
        only_2024 = players_2024 - players_2025
        only_2025 = players_2025 - players_2024
        
        print(f"  Players in both years: {len(overlap):,}")
        print(f"  Players only in 2024: {len(only_2024):,}")
        print(f"  Players only in 2025: {len(only_2025):,}")
        
        # Team overlap analysis
        print(f"\n--- Team Overlap Analysis ---")
        teams_2024 = set(datasets[2024]['teamname'].unique()) if not datasets[2024].empty else set()
        teams_2025 = set(datasets[2025]['teamname'].unique()) if not datasets[2025].empty else set()
        
        team_overlap = teams_2024.intersection(teams_2025)
        only_teams_2024 = teams_2024 - teams_2025
        only_teams_2025 = teams_2025 - teams_2024
        
        print(f"  Teams in both years: {len(team_overlap):,}")
        print(f"  Teams only in 2024: {len(only_teams_2024):,}")
        print(f"  Teams only in 2025: {len(only_teams_2025):,}")
        
        # Sample player analysis
        print(f"\n--- Sample Player Analysis ---")
        sample_players = list(overlap)[:5] if overlap else list(combined_df['playername'].unique())[:5]
        
        for player in sample_players:
            player_data = combined_df[combined_df['playername'] == player]
            years = player_data['data_year'].value_counts().sort_index()
            print(f"  {player}: {len(player_data)} total matches")
            for year, count in years.items():
                print(f"    {year}: {count} matches")
        
        # Benefits summary
        print(f"\n--- Benefits Summary ---")
        print(f"✓ Increased dataset size: {len(combined_df):,} total matches")
        print(f"✓ Broader player coverage: {combined_df['playername'].nunique():,} unique players")
        print(f"✓ More team diversity: {combined_df['teamname'].nunique():,} unique teams")
        print(f"✓ Extended temporal coverage: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"✓ Cross-year player tracking: {len(overlap):,} players with data in both years")
        print(f"✓ Enhanced model training: More diverse and comprehensive training data")
        print(f"✓ Better prediction accuracy: Historical performance patterns across years")
        
        return combined_df
    else:
        print("No datasets available for analysis")
        return None

if __name__ == "__main__":
    analyze_datasets() 