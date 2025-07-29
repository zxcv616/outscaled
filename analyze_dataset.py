#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the benefits of combining 2024 and 2025 Oracle's Elixir data
"""

import pandas as pd
import numpy as np
import os

def analyze_datasets():
    """Analyze the benefits of combining multiple datasets"""
    print("📊 Oracle's Elixir Dataset Analysis")
    print("=" * 50)
    
    # Check which files exist
    csv_files = {
        "2025": "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        "2024": "2024_LoL_esports_match_data_from_OraclesElixir.csv"
    }
    
    datasets = {}
    total_matches = 0
    total_players = set()
    total_teams = set()
    
    for year, filename in csv_files.items():
        if os.path.exists(filename):
            print(f"\n📈 Loading {year} dataset...")
            df = pd.read_csv(filename, low_memory=False)
            
            # Filter for complete data
            df_complete = df[df["datacompleteness"] == "complete"]
            
            # Basic stats
            matches = len(df_complete)
            players = len(df_complete["playername"].unique())
            teams = len(df_complete["teamname"].unique())
            
            datasets[year] = {
                "matches": matches,
                "players": players,
                "teams": teams,
                "dataframe": df_complete
            }
            
            total_matches += matches
            total_players.update(df_complete["playername"].unique())
            total_teams.update(df_complete["teamname"].unique())
            
            print(f"   ✅ {year}: {matches:,} matches, {players:,} players, {teams:,} teams")
        else:
            print(f"   ❌ {year}: File not found")
    
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total matches: {total_matches:,}")
    print(f"   Total unique players: {len(total_players):,}")
    print(f"   Total unique teams: {len(total_teams):,}")
    
    # Analyze overlap
    if len(datasets) >= 2:
        print(f"\n🔄 Dataset Overlap Analysis:")
        
        # Player overlap
        players_2024 = set(datasets["2024"]["dataframe"]["playername"].unique()) if "2024" in datasets else set()
        players_2025 = set(datasets["2025"]["dataframe"]["playername"].unique()) if "2025" in datasets else set()
        
        overlap_players = players_2024.intersection(players_2025)
        unique_2024 = players_2024 - players_2025
        unique_2025 = players_2025 - players_2024
        
        print(f"   Players in both years: {len(overlap_players):,}")
        print(f"   Players only in 2024: {len(unique_2024):,}")
        print(f"   Players only in 2025: {len(unique_2025):,}")
        
        # Team overlap
        teams_2024 = set(datasets["2024"]["dataframe"]["teamname"].unique()) if "2024" in datasets else set()
        teams_2025 = set(datasets["2025"]["dataframe"]["teamname"].unique()) if "2025" in datasets else set()
        
        overlap_teams = teams_2024.intersection(teams_2025)
        unique_teams_2024 = teams_2024 - teams_2025
        unique_teams_2025 = teams_2025 - teams_2024
        
        print(f"   Teams in both years: {len(overlap_teams):,}")
        print(f"   Teams only in 2024: {len(unique_teams_2024):,}")
        print(f"   Teams only in 2025: {len(unique_teams_2025):,}")
    
    # Benefits analysis
    print(f"\n🎯 Benefits of Combined Dataset:")
    print(f"   📈 Data Volume: {total_matches:,} total matches vs individual year averages")
    print(f"   👥 Player Coverage: {len(total_players):,} unique players")
    print(f"   🏆 Team Coverage: {len(total_teams):,} unique teams")
    print(f"   ⏰ Temporal Coverage: Multi-year meta evolution")
    print(f"   🔄 Player Continuity: Track player performance across years")
    print(f"   📊 Statistical Power: Larger sample sizes for ML training")
    
    # Sample player analysis
    if len(datasets) >= 2 and len(overlap_players) > 0:
        print(f"\n👤 Sample Player Analysis (players in both years):")
        sample_players = list(overlap_players)[:5]
        
        for player in sample_players:
            matches_2024 = len(datasets["2024"]["dataframe"][datasets["2024"]["dataframe"]["playername"] == player])
            matches_2025 = len(datasets["2025"]["dataframe"][datasets["2025"]["dataframe"]["playername"] == player])
            total_matches = matches_2024 + matches_2025
            
            print(f"   {player}: {matches_2024} (2024) + {matches_2025} (2025) = {total_matches} total")
    
    print(f"\n✅ Recommendation: Use combined dataset for maximum coverage and accuracy!")
    return datasets

if __name__ == "__main__":
    analyze_datasets() 