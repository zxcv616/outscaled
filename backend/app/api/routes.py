from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.database import get_db
from app.schemas.pydantic_models import PropRequest, PlayerStatsResponse
from app.services.data_fetcher import DataFetcher
from app.ml.predictor import PropPredictor
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize components lazily
_data_fetcher = None
_predictor = None

def get_data_fetcher():
    global _data_fetcher
    if _data_fetcher is None:
        try:
            _data_fetcher = DataFetcher()
            logger.info("DataFetcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DataFetcher: {e}")
            raise HTTPException(status_code=500, detail="Data service unavailable")
    return _data_fetcher

def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            _predictor = PropPredictor()
            logger.info("PropPredictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PropPredictor: {e}")
            raise HTTPException(status_code=500, detail="Prediction service unavailable")
    return _predictor

router = APIRouter()

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Outscaled.gg API is running"}

@router.get("/players/search")
def search_players(query: str = "", limit: int = 10, db: Session = Depends(get_db)):
    """Search for players in the dataset"""
    try:
        if not query or len(query) < 2:
            return {"players": [], "message": "Please provide a search query with at least 2 characters"}
        
        # Get all available players
        all_players = get_data_fetcher().get_available_players()
        
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
        # Get all available teams from the dataset
        all_teams = get_data_fetcher().get_available_teams()
        
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
        # Get all players and their match counts
        all_players = get_data_fetcher().get_available_players()
        
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
        # Get all available players
        all_players = get_data_fetcher().get_available_players()
        
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

@router.post("/predict")
async def predict_prop(prop_request: PropRequest, verbose: bool = False, db: Session = Depends(get_db)):
    """Make a prediction for a prop"""
    try:
        # Validate player exists first
        all_players = get_data_fetcher().get_available_players()
        
        player_exists = any(
            player.lower() == prop_request.player_name.lower() 
            for player in all_players
        )
        
        if not player_exists:
            # Find similar players for suggestions
            similar_players = []
            player_name_lower = prop_request.player_name.lower()
            
            for player in all_players:
                if (player_name_lower in player.lower() or 
                    player.lower() in player_name_lower):
                    similar_players.append(player)
            
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Player not found in dataset",
                    "player_name": prop_request.player_name,
                    "suggestions": similar_players[:5],
                    "message": "Please use a valid player name from the dataset"
                }
            )
        
        # Extract map range from request
        map_range = prop_request.map_range if prop_request.map_range else [1]
        
        # Get player stats with map range
        player_stats = get_data_fetcher().get_player_stats(
            prop_request.player_name, 
            db, 
            map_range=map_range
        )
        
        # Convert to dict for predictor
        prop_request_dict = prop_request.dict()
        prop_request_dict["map_range"] = map_range
        
        # Make prediction with verbose parameter
        prediction_result = get_predictor().predict(player_stats, prop_request_dict, verbose=verbose)
        
        # Add player stats to response
        prediction_result["player_stats"] = player_stats
        prediction_result["prop_request"] = prop_request_dict
        
        # Return with no-cache headers
        return JSONResponse(
            content=prediction_result,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/playerstats/{player_name}")
async def get_player_stats(player_name: str, db: Session = Depends(get_db)):
    """Get player statistics"""
    try:
        player_stats = get_data_fetcher().get_player_stats(player_name, db)
        
        # Create response with explicit fields
        response = PlayerStatsResponse(
            player_name=player_stats.get("player_name", player_name),
            recent_matches=player_stats.get("recent_matches", []),
            avg_kills=player_stats.get("avg_kills", 0.0),
            avg_assists=player_stats.get("avg_assists", 0.0),
            avg_cs=player_stats.get("avg_cs", 0.0),
            avg_deaths=player_stats.get("avg_deaths", 0.0),
            win_rate=player_stats.get("win_rate", 0.0),
            data_source=player_stats.get("data_source", "none"),
            data_years=player_stats.get("data_years", "No data available")
        )
        
        # Return with no-cache headers
        return JSONResponse(
            content=response.dict(),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )
        
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/info")
def get_model_info():
    """Get information about the current ML model"""
    try:
        return get_predictor().get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@router.get("/model/features")
def get_feature_importance():
    """Get feature importance scores from the model"""
    try:
        feature_importance = get_predictor().get_feature_importance()
        return {
            "feature_importance": feature_importance,
            "total_features": len(feature_importance)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

@router.post("/statistics/probability-distribution")
async def get_probability_distribution(prop_request: PropRequest, range_std: float = 5.0):
    """Get probability distribution for a range of values around the prop value"""
    try:
        # Validate player exists
        player_name = prop_request.player_name
        if not get_data_fetcher().player_exists(player_name):
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in dataset")
        
        # Get player stats - use mock db to avoid database dependency
        class MockDB:
            def __init__(self):
                pass
        
        mock_db = MockDB()
        player_stats = get_data_fetcher().get_player_stats(
            player_name=player_name,
            db=mock_db,
            map_range=prop_request.map_range
        )
        
        if not player_stats:
            raise HTTPException(status_code=404, detail=f"No data found for player '{player_name}'")
        
        # Calculate probability distribution
        distribution_result = get_predictor().calculate_probability_distribution(
            player_stats=player_stats,
            prop_request=prop_request.dict(),
            range_std=range_std
        )
        
        if "error" in distribution_result:
            raise HTTPException(status_code=400, detail=distribution_result["error"])
        
        return {
            "player_name": player_name,
            "prop_request": prop_request.dict(),
            "probability_distribution": distribution_result["probability_distribution"],
            "summary_stats": distribution_result["summary_stats"],
            "analysis_range": distribution_result["analysis_range"],
            "range_std": range_std
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating probability distribution: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating probability distribution: {str(e)}")

@router.post("/statistics/insights")
async def get_statistical_insights(prop_request: PropRequest):
    """Get detailed statistical insights for the current prediction"""
    try:
        # Validate player exists
        player_name = prop_request.player_name
        if not get_data_fetcher().player_exists(player_name):
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in dataset")
        
        # Get player stats - use mock db to avoid database dependency
        class MockDB:
            def __init__(self):
                pass
        
        mock_db = MockDB()
        player_stats = get_data_fetcher().get_player_stats(
            player_name=player_name,
            db=mock_db,
            map_range=prop_request.map_range
        )
        
        if not player_stats:
            raise HTTPException(status_code=404, detail=f"No data found for player '{player_name}'")
        
        # Get statistical insights
        insights_result = get_predictor().get_statistical_insights(
            player_stats=player_stats,
            prop_request=prop_request.dict()
        )
        
        if "error" in insights_result:
            raise HTTPException(status_code=400, detail=insights_result["error"])
        
        return {
            "player_name": player_name,
            "prop_request": prop_request.dict(),
            "statistical_measures": insights_result["statistical_measures"],
            "probability_analysis": insights_result["probability_analysis"],
            "confidence_intervals": insights_result["confidence_intervals"],
            "trend_analysis": insights_result["trend_analysis"],
            "volatility_metrics": insights_result["volatility_metrics"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistical insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistical insights: {str(e)}")

@router.post("/statistics/comprehensive")
async def get_comprehensive_statistics(prop_request: PropRequest, range_std: float = 5.0):
    """Get comprehensive statistical analysis including prediction, probability distribution, and insights"""
    try:
        # Validate player exists
        player_name = prop_request.player_name
        if not get_data_fetcher().player_exists(player_name):
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found in dataset")
        
        # Get player stats - use mock db to avoid database dependency
        class MockDB:
            def __init__(self):
                pass
        
        mock_db = MockDB()
        player_stats = get_data_fetcher().get_player_stats(
            player_name=player_name,
            db=mock_db,
            map_range=prop_request.map_range
        )
        
        if not player_stats:
            raise HTTPException(status_code=404, detail=f"No data found for player '{player_name}'")
        
        # Get prediction
        prediction_result = get_predictor().predict(
            player_stats=player_stats,
            prop_request=prop_request.dict(),
            verbose=True
        )
        
        # Get probability distribution
        distribution_result = get_predictor().calculate_probability_distribution(
            player_stats=player_stats,
            prop_request=prop_request.dict(),
            range_std=range_std
        )
        
        # Get statistical insights
        insights_result = get_predictor().get_statistical_insights(
            player_stats=player_stats,
            prop_request=prop_request.dict()
        )
        
        return {
            "player_name": player_name,
            "prop_request": prop_request.dict(),
            "prediction": prediction_result,
            "probability_distribution": distribution_result.get("probability_distribution", {}),
            "statistical_insights": insights_result,
            "summary_stats": distribution_result.get("summary_stats", {}),
            "analysis_range": distribution_result.get("analysis_range", {}),
            "range_std": range_std
        }
        
    except Exception as e:
        logger.error(f"Error getting comprehensive statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting comprehensive statistics: {str(e)}") 