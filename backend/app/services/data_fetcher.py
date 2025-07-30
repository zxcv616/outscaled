import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import logging
import re
from app.core.config import settings
# Make database import optional to avoid psycopg2 dependency
try:
    from app.models.database import Player, PlayerMatch, Match
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    Player = None
    PlayerMatch = None
    Match = None
from app.utils.data_utils import (
    create_map_index_column, 
    safe_division, 
    validate_required_columns, 
    extract_year_from_filename,
    aggregate_stats_by_map_range
)
from sqlalchemy.orm import Session

# Set up logging
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        # Support multiple CSV files for comprehensive data coverage
        self.csv_paths = [
            "2025_LoL_esports_match_data_from_OraclesElixir.csv",
            "2024_LoL_esports_match_data_from_OraclesElixir.csv"
        ]
        self.df = None
        self._load_data()
    
    def _validate_csv_schema(self, df: pd.DataFrame, csv_path: str) -> bool:
        """Validate that required columns exist in the CSV"""
        required_columns = [
            'kills', 'deaths', 'assists', 'total cs', 'earnedgold', 
            'gamelength', 'teamkills', 'gameid', 'playername', 'teamname',
            'result', 'date', 'datacompleteness'
        ]
        
        is_valid, missing_columns = validate_required_columns(df, required_columns)
        if not is_valid:
            logger.error(f"Missing required columns in {csv_path}: {missing_columns}")
            return False
        
        logger.info(f"Schema validation passed for {csv_path}")
        return True
    
    def _load_data(self):
        """Load and preprocess multiple Oracle's Elixir CSV datasets"""
        try:
            logger.info("Loading Oracle's Elixir datasets...")
            all_dataframes = []
            
            for csv_path in self.csv_paths:
                if os.path.exists(csv_path):
                    logger.info(f"Loading {csv_path}...")
                    df = pd.read_csv(csv_path, low_memory=False)
                    
                    # Validate schema before processing
                    if not self._validate_csv_schema(df, csv_path):
                        logger.warning(f"Skipping {csv_path} due to schema validation failure")
                        continue
                    
                    # Filter for complete data only
                    df = df[df["datacompleteness"] == "complete"]
                    
                    # Extract year using utility function
                    year = extract_year_from_filename(csv_path)
                    df['data_year'] = year
                    
                    all_dataframes.append(df)
                    logger.info(f"Loaded {len(df)} complete matches from {year}")
                else:
                    logger.warning(f"Warning: {csv_path} not found, skipping...")
            
            if not all_dataframes:
                logger.warning("No CSV files found, creating empty dataframe")
                self.df = pd.DataFrame()
                return
            
            # Combine all dataframes
            self.df = pd.concat(all_dataframes, ignore_index=True)
            
            # Remove duplicates based on gameid and playername (same player in same game)
            self.df = self.df.drop_duplicates(subset=['gameid', 'playername'], keep='first')
            
            # Convert column types - use correct column names
            numeric_columns = ["kills", "deaths", "assists", "dpm", "earnedgoldshare", "total cs", "earnedgold"]
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Add KDA column with safe division
            self.df["kda"] = safe_division(
                self.df["kills"] + self.df["assists"], 
                self.df["deaths"], 
                default_value=1.0
            )
            
            # Add GPM (Gold Per Minute) column - use earnedgold and gamelength
            self.df["gpm"] = safe_division(
                self.df["earnedgold"], 
                self.df["gamelength"] / 60, 
                default_value=0.0
            )
            
            # Add KP% (Kill Participation) column with safe division
            self.df["kp_percent"] = safe_division(
                self.df["kills"] + self.df["assists"], 
                self.df["teamkills"], 
                default_value=0.0
            )
            
            # Add map index tracking for series using utility function
            self.df = create_map_index_column(self.df)
            
            # Sort by date for proper temporal ordering
            self.df = self.df.sort_values('date', ascending=False)
            
            logger.info(f"Combined dataset: {len(self.df)} total matches with {len(self.df['playername'].unique())} unique players")
            logger.info(f"Data years: {self.df['data_year'].value_counts().to_dict()}")
            logger.info(f"Map index distribution: {self.df['map_index_within_series'].value_counts().sort_index().to_dict()}")
            
        except Exception as e:
            logger.error(f"Error loading Oracle's Elixir data: {e}")
            self.df = pd.DataFrame()
    
    def get_player_stats(self, player_name: str, db: Session, map_range: List[int] = [1, 2]) -> Dict:
        """Get comprehensive player statistics from Oracle's Elixir dataset with map-range support"""
        if self.df is None or self.df.empty:
            logger.warning("No Oracle's Elixir data available")
            return self._get_minimal_stats(player_name)
        
        try:
            # Search for player (case-insensitive)
            player_matches = self.df[self.df["playername"].str.lower() == player_name.lower()]
            
            if player_matches.empty:
                logger.warning(f"No data found for player: {player_name}")
                return self._get_minimal_stats(player_name)
            
            logger.info(f"Found {len(player_matches)} total matches for {player_name}")
            
            # Create map index for the player matches
            player_matches_with_index = create_map_index_column(player_matches)
            
            # Handle map range aggregation
            if map_range and map_range != [1]:
                # Use the aggregation utility to properly sum stats across map range
                
                # Aggregate stats across the map range
                aggregated_stats = aggregate_stats_by_map_range(player_matches_with_index, map_range)
                
                if aggregated_stats.empty:
                    logger.warning(f"No data found for player {player_name} in map range {map_range}. Using all available data.")
                    player_matches_filtered = player_matches
                    map_range_warning = f" (Note: No data available for maps {map_range}, using all matches)"
                else:
                    # For aggregated data, we need to calculate averages from the sums
                    # The aggregated data contains summed stats across map ranges
                    player_matches_filtered = aggregated_stats
                    map_range_warning = ""
                    logger.info(f"Aggregated {len(aggregated_stats)} match series for map range {map_range}")
            else:
                # Single map - use original filtering
                player_matches_filtered = player_matches_with_index[player_matches_with_index['map_index_within_series'].isin(map_range)]
                map_range_warning = ""
            
            if player_matches_filtered.empty:
                logger.warning(f"No data found for player {player_name}")
                return self._get_minimal_stats(player_name)
            
            logger.info(f"Using {len(player_matches_filtered)} matches/series for {player_name}")
            
            # Get recent matches (last 10) - filter by map range
            if map_range and map_range != [1]:
                # For multi-map ranges, use the aggregated data for recent matches
                recent_matches = player_matches_filtered.sort_values("date", ascending=False).head(10)
            else:
                # For single map, use filtered data
                recent_matches = player_matches_filtered.sort_values("date", ascending=False).head(10)
            
            # DEBUG: Check recent matches data
            logger.debug(f"Recent matches shape: {recent_matches.shape}")
            logger.debug(f"Recent matches columns: {recent_matches.columns.tolist()}")
            if not recent_matches.empty:
                logger.debug(f"Recent kills sample: {recent_matches['kills'].head().tolist()}")
                logger.debug(f"Recent kills mean: {recent_matches['kills'].mean()}")
                logger.debug(f"Recent kills isna: {recent_matches['kills'].isna().sum()}")
            else:
                logger.warning(f"Recent matches is empty for {player_name}")
            
            # Calculate comprehensive statistics
            maps_played = len(map_range) if map_range and map_range != [1] else 1
            
            # Get data year distribution for this player (use original data)
            data_years = player_matches['data_year'].value_counts().to_dict()
            data_years_str = ", ".join([f"{year} ({count} matches)" for year, count in data_years.items()])
            
            logger.debug(f"data_years calculation for {player_name}:")
            logger.debug(f"  data_years dict: {data_years}")
            logger.debug(f"  data_years_str: {data_years_str}")
            
            # Calculate statistics based on whether we're using aggregated or individual map data
            if map_range and map_range != [1]:
                # For aggregated data, the stats are already summed across the map range
                # We need to calculate averages from the sums
                total_maps = player_matches_filtered['map_index_within_series'].sum()  # Count of maps played
                
                # For recent averages, use the filtered recent_matches
                recent_kills_avg = recent_matches["kills"].head(5).mean() if not recent_matches.empty else 0.0
                recent_assists_avg = recent_matches["assists"].head(5).mean() if not recent_matches.empty else 0.0
                recent_cs_avg = recent_matches["total cs"].head(5).mean() if not recent_matches.empty else 0.0
                
                # For win rate and other stats that need the original data structure
                win_rate = (player_matches["result"] == 1).mean()
                avg_kda = player_matches["kda"].mean()
                avg_gpm = player_matches["gpm"].mean()
                avg_kp_percent = player_matches["kp_percent"].mean()
                
                stats = {
                    "player_name": player_name,
                    "recent_matches": self._format_recent_matches(recent_matches),
                    "avg_kills": player_matches_filtered["kills"].sum() / max(total_maps, 1),  # Average per map
                    "avg_assists": player_matches_filtered["assists"].sum() / max(total_maps, 1),
                    "avg_cs": player_matches_filtered["total cs"].sum() / max(total_maps, 1),
                    "avg_deaths": player_matches_filtered["deaths"].sum() / max(total_maps, 1),
                    "avg_gold": player_matches_filtered["earnedgold"].sum() / max(total_maps, 1),
                    "avg_damage": player_matches.get("dpm", pd.Series([0.0])).mean(),  # Use original data for dpm
                    "avg_vision": player_matches.get("visionscore", pd.Series([0.0])).mean(),  # Use original data for vision
                    "recent_kills_avg": recent_kills_avg,
                    "recent_assists_avg": recent_assists_avg,
                    "recent_cs_avg": recent_cs_avg,
                    "win_rate": win_rate,
                    "avg_kda": avg_kda,
                    "avg_gpm": avg_gpm,
                    "avg_kp_percent": avg_kp_percent,
                    "data_source": "oracles_elixir",
                    "data_years": data_years_str,
                    "map_range": map_range,
                    "maps_played": maps_played,
                    "map_range_warning": map_range_warning,
                    "total_matches_available": len(player_matches),
                    "matches_in_range": len(player_matches_filtered)
                }
            else:
                # For single map data, use per-map averages
                # For recent averages, use the filtered recent_matches
                recent_kills_avg = recent_matches["kills"].head(5).mean() if not recent_matches.empty else 0.0
                recent_assists_avg = recent_matches["assists"].head(5).mean() if not recent_matches.empty else 0.0
                recent_cs_avg = recent_matches["total cs"].head(5).mean() if not recent_matches.empty else 0.0
                
                stats = {
                    "player_name": player_name,
                    "recent_matches": self._format_recent_matches(recent_matches),
                    "avg_kills": player_matches_filtered["kills"].mean(),
                    "avg_assists": player_matches_filtered["assists"].mean(),
                    "avg_cs": player_matches_filtered["total cs"].mean(),
                    "avg_deaths": player_matches_filtered["deaths"].mean(),
                    "avg_gold": player_matches_filtered["earnedgold"].mean(),
                    "avg_damage": player_matches_filtered.get("dpm", pd.Series([0.0])).mean(),
                    "avg_vision": player_matches_filtered.get("visionscore", pd.Series([0.0])).mean(),
                    "recent_kills_avg": recent_kills_avg,
                    "recent_assists_avg": recent_assists_avg,
                    "recent_cs_avg": recent_cs_avg,
                    "win_rate": (player_matches_filtered["result"] == 1).mean(),
                    "avg_kda": player_matches_filtered["kda"].mean(),
                    "avg_gpm": player_matches_filtered["gpm"].mean(),
                    "avg_kp_percent": player_matches_filtered["kp_percent"].mean(),
                    "data_source": "oracles_elixir",
                    "data_years": data_years_str,
                    "map_range": map_range,
                    "maps_played": maps_played,
                    "map_range_warning": map_range_warning,
                    "total_matches_available": len(player_matches),
                    "matches_in_range": len(player_matches_filtered)
                }
            
            # DEBUG: Check the calculated values
            logger.debug(f"Calculated stats for {player_name}:")
            logger.debug(f"  recent_kills_avg: {stats.get('recent_kills_avg')}")
            logger.debug(f"  avg_kills: {stats.get('avg_kills')}")
            logger.debug(f"  win_rate: {stats.get('win_rate')}")
            
            # Handle NaN values
            for key, value in stats.items():
                if isinstance(value, float) and np.isnan(value):
                    logger.warning(f"NaN value found for {key}, setting to 0.0")
                    stats[key] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting player stats for {player_name}: {e}")
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
                "league": str(match.get("league", "")),
                "data_year": str(match.get("data_year", ""))
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
            "data_source": "none",
            "data_years": "No data available",
            "maps_played": 1,
            "map_range": [1],
            "map_range_warning": "",
            "total_matches_available": 0,
            "matches_in_range": 0
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
        """Get all available teams from the dataset"""
        try:
            if self.df is None or self.df.empty:
                return []
            
            teams = self.df['teamname'].dropna().unique().tolist()
            return sorted(teams)
        except Exception as e:
            logger.error(f"Error getting available teams: {e}")
            return []
    
    def player_exists(self, player_name: str) -> bool:
        """Check if a player exists in the dataset"""
        try:
            if self.df is None or self.df.empty:
                return False
            
            # Check if player exists (case-insensitive)
            player_exists = self.df['playername'].str.lower().str.contains(player_name.lower(), na=False).any()
            return player_exists
        except Exception as e:
            logger.error(f"Error checking if player exists: {e}")
            return False
    
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
            "avg_gold": team_matches["earnedgold"].mean(),  # Fixed: use earnedgold instead of gold
            "data_source": "oracles_elixir"
        } 