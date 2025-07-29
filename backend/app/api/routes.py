from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict
from app.models.database import get_db
from app.schemas.pydantic_models import PropRequest, PredictionResponse, PlayerStatsResponse
from app.services.data_fetcher import DataFetcher
from app.ml.predictor import PropPredictor

router = APIRouter()

# Initialize services
data_fetcher = DataFetcher()
predictor = PropPredictor()

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Outscaled.gg API is running"}

@router.get("/players/search")
def search_players(query: str = "", limit: int = 10, db: Session = Depends(get_db)):
    """Search for players in the dataset"""
    try:
        data_fetcher = DataFetcher()
        
        if not query or len(query) < 2:
            return {"players": [], "message": "Please provide a search query with at least 2 characters"}
        
        # Get all available players
        all_players = data_fetcher.get_available_players()
        
        # Filter players based on query (case-insensitive)
        query_lower = query.lower()
        matching_players = [
            player for player in all_players 
            if query_lower in player.lower()
        ]
        
        # Sort by relevance (exact matches first, then alphabetical)
        def sort_key(player):
            player_lower = player.lower()
            if player_lower.startswith(query_lower):
                return (0, player_lower)  # Exact prefix match
            elif query_lower in player_lower:
                return (1, player_lower)  # Contains match
            else:
                return (2, player_lower)  # Fallback
        
        matching_players.sort(key=sort_key)
        
        # Limit results
        results = matching_players[:limit]
        
        return {
            "players": results,
            "total_found": len(matching_players),
            "query": query,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching players: {str(e)}")

@router.get("/teams")
def get_available_teams(db: Session = Depends(get_db)):
    """Get all available teams from the dataset"""
    try:
        data_fetcher = DataFetcher()
        
        # Get all available teams from the dataset
        all_teams = data_fetcher.get_available_teams()
        
        return {
            "teams": all_teams,
            "total_teams": len(all_teams)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting teams: {str(e)}")

@router.get("/players/popular")
def get_popular_players(limit: int = 20, db: Session = Depends(get_db)):
    """Get popular players (players with most matches)"""
    try:
        data_fetcher = DataFetcher()
        
        # Get all players and their match counts
        all_players = data_fetcher.get_available_players()
        
        # For now, return first N players as "popular"
        # In a real implementation, you'd sort by match count
        popular_players = all_players[:limit]
        
        return {
            "players": popular_players,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting popular players: {str(e)}")

@router.get("/players/validate/{player_name}")
def validate_player(player_name: str, db: Session = Depends(get_db)):
    """Validate if a player exists in the dataset"""
    try:
        data_fetcher = DataFetcher()
        
        # Get all available players
        all_players = data_fetcher.get_available_players()
        
        # Check if player exists (case-insensitive)
        player_exists = any(
            player.lower() == player_name.lower() 
            for player in all_players
        )
        
        if player_exists:
            # Find the exact player name (preserve case)
            exact_name = next(
                player for player in all_players 
                if player.lower() == player_name.lower()
            )
            
            return {
                "valid": True,
                "player_name": exact_name,
                "message": "Player found in dataset"
            }
        else:
            # Find similar players for suggestions
            similar_players = []
            player_name_lower = player_name.lower()
            
            for player in all_players:
                if (player_name_lower in player.lower() or 
                    player.lower() in player_name_lower):
                    similar_players.append(player)
            
            return {
                "valid": False,
                "player_name": player_name,
                "suggestions": similar_players[:5],
                "message": "Player not found. Did you mean one of these?"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating player: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
def predict_prop(request: PropRequest, db: Session = Depends(get_db)):
    """Make a prediction for a player prop with map-range support"""
    try:
        # Validate player exists first
        data_fetcher = DataFetcher()
        all_players = data_fetcher.get_available_players()
        
        player_exists = any(
            player.lower() == request.player_name.lower() 
            for player in all_players
        )
        
        if not player_exists:
            # Find similar players for suggestions
            similar_players = []
            player_name_lower = request.player_name.lower()
            
            for player in all_players:
                if (player_name_lower in player.lower() or 
                    player.lower() in player_name_lower):
                    similar_players.append(player)
            
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Player not found in dataset",
                    "player_name": request.player_name,
                    "suggestions": similar_players[:5],
                    "message": "Please use a valid player name from the dataset"
                }
            )
        
        # Get map range from request
        map_range = getattr(request, 'map_range', [1, 2])  # Default to Maps 1-2
        if not map_range:
            map_range = [1, 2]  # Fallback default
        
        # Get player stats with map range
        player_stats = data_fetcher.get_player_stats(request.player_name, db, map_range)
        
        # Add map range to prop request
        prop_request_dict = request.dict()
        prop_request_dict['map_range'] = map_range
        
        # Make prediction
        predictor = PropPredictor()
        prediction_result = predictor.predict(player_stats, prop_request_dict)
        
        return PredictionResponse(
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            reasoning=prediction_result["reasoning"],
            player_stats=player_stats,
            prop_request=prop_request_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.get("/playerstats/{player_name}", response_model=PlayerStatsResponse)
def get_player_stats(player_name: str, db: Session = Depends(get_db)):
    """Get player statistics"""
    try:
        data_fetcher = DataFetcher()
        player_stats = data_fetcher.get_player_stats(player_name, db)
        
        return PlayerStatsResponse(
            player_name=player_stats["player_name"],
            recent_matches=player_stats["recent_matches"],
            avg_kills=player_stats["avg_kills"],
            avg_assists=player_stats["avg_assists"],
            avg_cs=player_stats["avg_cs"],
            avg_deaths=player_stats["avg_deaths"],
            win_rate=player_stats["win_rate"],
            data_source=player_stats["data_source"],
            data_years=player_stats.get("data_years", "No data available")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting player stats: {str(e)}")

@router.get("/model/info")
def get_model_info():
    """Get information about the current ML model"""
    try:
        predictor = PropPredictor()
        return predictor.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/model/features")
def get_feature_importance():
    """Get feature importance scores"""
    try:
        predictor = PropPredictor()
        feature_importance = predictor.get_feature_importance()
        return {"feature_importance": feature_importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}") 