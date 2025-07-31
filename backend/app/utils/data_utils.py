"""
Data utility functions for reusable data processing logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_map_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create map_index_within_series column for map-range support
    
    Oracle's Elixir data structure:
    - Each gameid represents a complete series (not individual maps)
    - For betting simulation, we'll assign map numbers to each series
    - We'll use chronological order to simulate map numbers within series
    
    Args:
        df: DataFrame with 'gameid' column
        
    Returns:
        DataFrame with added 'match_series' and 'map_index_within_series' columns
    """
    if 'gameid' not in df.columns:
        raise ValueError("DataFrame must contain 'gameid' column")
    
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Extract tournament series identifier (e.g., 'LOLTMNT03' from 'LOLTMNT03_179647')
    df_copy['match_series'] = df_copy['gameid'].str.split('_').str[0]
    
    # For betting simulation, we'll assign map numbers to each series
    # We'll use chronological order within each tournament to simulate map numbers
    
    # Sort by date within each tournament series
    df_copy = df_copy.sort_values(['match_series', 'date'])
    
    # Assign map numbers based on chronological order within each series
    # This simulates individual maps for betting purposes
    df_copy['map_index_within_series'] = df_copy.groupby('match_series').cumcount() + 1
    
    # Cap the map numbers at a reasonable maximum (e.g., 5 maps per series)
    df_copy['map_index_within_series'] = df_copy['map_index_within_series'].clip(upper=5)
    
    return df_copy

def filter_by_map_range(df: pd.DataFrame, map_range: List[int]) -> pd.DataFrame:
    """
    Filter DataFrame by map range
    
    Args:
        df: DataFrame with 'map_index_within_series' column
        map_range: List of map indices to include (e.g., [1, 2] for Maps 1-2)
        
    Returns:
        Filtered DataFrame
    """
    if 'map_index_within_series' not in df.columns:
        raise ValueError("DataFrame must contain 'map_index_within_series' column")
    
    return df[df['map_index_within_series'].isin(map_range)]

def aggregate_stats_by_map_range(df: pd.DataFrame, map_range: List[int], 
                               group_cols: List[str] = None) -> pd.DataFrame:
    """
    Aggregate statistics across a map range for betting simulation
    
    For betting purposes:
    - Map 1: Use only the first map of each series
    - Maps 1-2: Sum stats from the first two maps of each series
    - Maps 1-3: Sum stats from the first three maps of each series
    
    Args:
        df: DataFrame with player/match data
        map_range: List of map indices to aggregate (e.g., [1, 2] for Maps 1-2)
        group_cols: Columns to group by (default: ['playername', 'match_series'])
        
    Returns:
        Aggregated DataFrame with summed stats
    """
    if group_cols is None:
        group_cols = ['playername', 'match_series']
    
    # Ensure map index column exists
    if 'map_index_within_series' not in df.columns:
        df = create_map_index_column(df)
    
    # Filter by map range - only include maps that are in the requested range
    filtered_df = filter_by_map_range(df, map_range)
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Define columns to sum for betting stats
    sum_columns = ['kills', 'deaths', 'assists', 'total cs', 'earnedgold']
    available_columns = [col for col in sum_columns if col in filtered_df.columns]
    
    # For betting simulation, we need to group by player and series
    # Then sum the stats across the requested map range
    agg_dict = {col: 'sum' for col in available_columns}
    
    # Add count of maps played in this range
    agg_dict['map_index_within_series'] = 'count'
    
    # Add date column (take the most recent date from the series)
    if 'date' in filtered_df.columns:
        agg_dict['date'] = 'max'  # Use the most recent date from the series
    
    # Group by player and series to get aggregated stats
    aggregated = filtered_df.groupby(group_cols).agg(agg_dict).reset_index()
    
    return aggregated

def safe_division(numerator: pd.Series, denominator: pd.Series, 
                 default_value: float = 1.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        default_value: Value to use when denominator is zero
        
    Returns:
        Series with safe division results
    """
    return np.where(denominator != 0, numerator / denominator, default_value)

def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all required columns exist, False otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns

def extract_year_from_filename(filename: str) -> str:
    """
    Extract year from filename using regex
    
    Args:
        filename: Filename to extract year from
        
    Returns:
        Year as string
    """
    import re
    match = re.search(r'(\d{4})_LoL_esports_match_data', filename)
    if match:
        return match.group(1)
    else:
        # Fallback to filename parsing
        import os
        return os.path.basename(filename).split("_")[0] 