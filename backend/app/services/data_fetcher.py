import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
from app.core.config import settings
from app.models.database import Player, PlayerMatch, Match
from sqlalchemy.orm import Session

class DataFetcher:
    def __init__(self):
        self.csv_path = "2025_LoL_esports_match_data_from_OraclesElixir.csv"
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the Oracle's Elixir CSV data"""
        try:
            print(f"Loading Oracle's Elixir data from {self.csv_path}...")
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            
            # Filter for complete data only
            self.df = self.df[self.df["datacompleteness"] == "complete"]
            
            # Convert column types - use correct column names
            numeric_columns = ["kills", "deaths", "assists", "dpm", "earnedgoldshare", "total cs", "earnedgold"]
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Add KDA column
            self.df["kda"] = (self.df["kills"] + self.df["assists"]) / self.df["deaths"].replace(0, 1)
            
            # Add GPM (Gold Per Minute) column - use earnedgold and gamelength
            self.df["gpm"] = self.df["earnedgold"] / (self.df["gamelength"] / 60)
            
            # Add KP% (Kill Participation) column
            self.df["kp_percent"] = (self.df["kills"] + self.df["assists"]) / self.df["teamkills"]
            
            # Add map index tracking for series
            self.df['match_series'] = self.df['gameid'].str.split('_').str[0]
            self.df['map_index_within_series'] = self.df.groupby('match_series')['gameid'].rank(method='dense').astype(int)
            
            print(f"Loaded {len(self.df)} complete matches with {len(self.df['playername'].unique())} unique players")
            print(f"Map index distribution: {self.df['map_index_within_series'].value_counts().sort_index().to_dict()}")
            
        except Exception as e:
            print(f"Error loading Oracle's Elixir data: {e}")
            self.df = pd.DataFrame()
    
    def get_player_stats(self, player_name: str, db: Session, map_range: List[int] = [1, 2]) -> Dict:
        """Get comprehensive player statistics from Oracle's Elixir dataset with map-range support"""
        if self.df is None or self.df.empty:
            print("No Oracle's Elixir data available")
            return self._get_minimal_stats(player_name)
        
        try:
            # Search for player (case-insensitive)
            player_matches = self.df[self.df["playername"].str.lower() == player_name.lower()]
            
            if player_matches.empty:
                print(f"No data found for player: {player_name}")
                return self._get_minimal_stats(player_name)
            
            print(f"Found {len(player_matches)} total matches for {player_name}")
            
            # Filter for specific map range if provided
            if map_range and map_range != [1]:
                player_matches_filtered = player_matches[player_matches['map_index_within_series'].isin(map_range)]
                print(f"Filtered to {len(player_matches_filtered)} matches for map range {map_range}")
                
                # If no matches in requested range, use all matches but warn
                if player_matches_filtered.empty:
                    print(f"No data found for player {player_name} in map range {map_range}. Using all available data.")
                    player_matches_filtered = player_matches
                    map_range_warning = f" (Note: No data available for maps {map_range}, using all matches)"
                else:
                    player_matches_filtered = player_matches_filtered
                    map_range_warning = ""
            else:
                player_matches_filtered = player_matches
                map_range_warning = ""
            
            if player_matches_filtered.empty:
                print(f"No data found for player {player_name}")
                return self._get_minimal_stats(player_name)
            
            print(f"Using {len(player_matches_filtered)} matches for {player_name}")
            
            # Get recent matches (last 10)
            recent_matches = player_matches_filtered.sort_values("date", ascending=False).head(10)
            
            # Calculate comprehensive statistics (map-range aware)
            maps_played = len(map_range) if map_range and map_range != [1] else 1
            
            stats = {
                "player_name": player_name,
                "recent_matches": self._format_recent_matches(recent_matches),
                "avg_kills": player_matches_filtered["kills"].mean() * maps_played,  # Scale by maps
                "avg_assists": player_matches_filtered["assists"].mean() * maps_played,
                "avg_cs": player_matches_filtered["total cs"].mean() * maps_played,
                "avg_deaths": player_matches_filtered["deaths"].mean() * maps_played,
                "avg_gold": player_matches_filtered["earnedgold"].mean() * maps_played,
                "avg_damage": player_matches_filtered["dpm"].mean() * maps_played if "dpm" in player_matches_filtered.columns else 0.0,
                "avg_vision": player_matches_filtered.get("visionscore", pd.Series([0])).mean() * maps_played,
                "recent_kills_avg": recent_matches["kills"].mean() * maps_played,
                "recent_assists_avg": recent_matches["assists"].mean() * maps_played,
                "recent_cs_avg": recent_matches["total cs"].mean() * maps_played,
                "win_rate": (player_matches_filtered["result"] == 1).mean(),
                "avg_kda": player_matches_filtered["kda"].mean(),
                "avg_gpm": player_matches_filtered["gpm"].mean(),
                "avg_kp_percent": player_matches_filtered["kp_percent"].mean(),
                "data_source": "oracles_elixir",
                "map_range": map_range,
                "maps_played": maps_played,
                "map_range_warning": map_range_warning,
                "total_matches_available": len(player_matches),
                "matches_in_range": len(player_matches_filtered)
            }
            
            # Handle NaN values
            for key, value in stats.items():
                if isinstance(value, float) and np.isnan(value):
                    stats[key] = 0.0
            
            return stats
            
        except Exception as e:
            print(f"Error processing player data: {e}")
            return self._get_minimal_stats(player_name)
    
    def _format_recent_matches(self, recent_matches: pd.DataFrame) -> List[Dict]:
        """Format recent matches for API response"""
        matches = []
        for _, match in recent_matches.iterrows():
            match_data = {
                "match_id": str(match.get("gameid", "")),
                "champion": str(match.get("champion", "")),
                "kills": int(match.get("kills", 0)),
                "deaths": int(match.get("deaths", 0)),
                "assists": int(match.get("assists", 0)),
                "cs": int(match.get("total cs", 0)),
                "gold": int(match.get("earnedgold", 0)),
                "damage_dealt": int(match.get("dpm", 0)) if "dpm" in match else 0,
                "vision_score": int(match.get("visionscore", 0)),
                "win": bool(match.get("result", 0)),
                "team_position": str(match.get("position", "")),
                "game_duration": int(match.get("gamelength", 0)),
                "map_number": 1,
                "side": "blue" if match.get("side", "") == "Blue" else "red",
                "match_date": str(match.get("date", "")),
                "team_name": str(match.get("teamname", "")),
                "opponent": str(match.get("opponent", "")),
                "league": str(match.get("league", ""))
            }
            matches.append(match_data)
        return matches
    
    def _get_minimal_stats(self, player_name: str) -> Dict:
        """Return minimal stats when no data is available"""
        return {
            "player_name": player_name,
            "recent_matches": [],
            "avg_kills": 0.0,
            "avg_assists": 0.0,
            "avg_cs": 0.0,
            "avg_deaths": 0.0,
            "avg_gold": 0.0,
            "avg_damage": 0.0,
            "avg_vision": 0.0,
            "recent_kills_avg": 0.0,
            "recent_assists_avg": 0.0,
            "recent_cs_avg": 0.0,
            "win_rate": 0.0,
            "avg_kda": 0.0,
            "avg_gpm": 0.0,
            "avg_kp_percent": 0.0,
            "data_source": "none"
        }
    
    def get_available_players(self) -> List[str]:
        """Get list of available players in the dataset"""
        if self.df is None or self.df.empty:
            return []
        
        # Get unique player names, drop NaN values, and convert to string
        players = self.df["playername"].dropna().astype(str).unique().tolist()
        
        # Sort alphabetically, case-insensitive
        players.sort(key=lambda x: x.lower())
        
        return players
    
    def get_available_teams(self) -> List[str]:
        """Get list of available teams in the dataset"""
        if self.df is None or self.df.empty:
            return []
        
        # Get unique team names, drop NaN values, and convert to string
        teams = self.df["teamname"].dropna().astype(str).unique().tolist()
        
        # Sort alphabetically, case-insensitive
        teams.sort(key=lambda x: x.lower())
        
        return teams
    
    def get_player_matches_by_league(self, player_name: str, league: str = None) -> List[Dict]:
        """Get player matches filtered by league"""
        if self.df is None or self.df.empty:
            return []
        
        player_matches = self.df[self.df["playername"].str.lower() == player_name.lower()]
        
        if league:
            player_matches = player_matches[player_matches["league"].str.lower() == league.lower()]
        
        return self._format_recent_matches(player_matches.sort_values("date", ascending=False))
    
    def get_team_stats(self, team_name: str) -> Dict:
        """Get team statistics"""
        if self.df is None or self.df.empty:
            return {}
        
        team_matches = self.df[self.df["teamname"].str.lower() == team_name.lower()]
        
        if team_matches.empty:
            return {}
        
        return {
            "team_name": team_name,
            "total_matches": len(team_matches),
            "win_rate": (team_matches["result"] == 1).mean(),
            "avg_kills": team_matches["kills"].mean(),
            "avg_assists": team_matches["assists"].mean(),
            "avg_deaths": team_matches["deaths"].mean(),
            "avg_gold": team_matches["gold"].mean(),
            "data_source": "oracles_elixir"
        } 