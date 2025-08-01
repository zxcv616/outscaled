from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class PropRequest(BaseModel):
    player_name: str
    prop_type: str  # kills, assists, cs, deaths, gold, damage
    prop_value: float
    opponent: Optional[str] = ""
    tournament: Optional[str] = ""
    map_range: Optional[List[int]] = [1, 2]  # Default to Maps 1-2
    start_map: Optional[int] = 1
    end_map: Optional[int] = 2
    map_number: Optional[int] = 1  # Legacy field for backward compatibility

class PredictionResponse(BaseModel):
    prediction: str  # "MORE" or "LESS"
    confidence: float  # 0-100
    reasoning: str
    player_stats: Dict[str, Any]
    prop_request: PropRequest

class PlayerStatsResponse(BaseModel):
    player_name: str
    recent_matches: List[Dict[str, Any]]
    avg_kills: float
    avg_assists: float
    avg_cs: float
    avg_deaths: float
    win_rate: float
    data_source: str
    data_years: Optional[str] = "No data available"

class ErrorResponse(BaseModel):
    detail: str 