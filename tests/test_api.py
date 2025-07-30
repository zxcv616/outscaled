import pytest
import requests
import json
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

class TestOutscaledAPI:
    """Test suite for Outscaled.gg API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_player_search(self):
        """Test player search functionality"""
        # Test valid search
        response = requests.get(f"{API_BASE}/players/search?query=pat&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "players" in data
        assert isinstance(data["players"], list)
        assert "total_found" in data
        
        # Test empty query
        response = requests.get(f"{API_BASE}/players/search?query=&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert data["players"] == []
    
    def test_player_validation(self):
        """Test player validation endpoint"""
        # Test valid player
        response = requests.get(f"{API_BASE}/players/validate/PatkicaA")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True
        assert "player_name" in data
        
        # Test invalid player
        response = requests.get(f"{API_BASE}/players/validate/InvalidPlayer123")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == False
        assert "suggestions" in data
    
    def test_prediction_basic(self):
        """Test basic prediction functionality"""
        prediction_data = {
            "player_name": "PatkicaA",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "prediction" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "player_stats" in data
        assert "prop_request" in data
        
        # Check prediction values
        assert data["prediction"] in ["MORE", "LESS"]
        assert 0 <= data["confidence"] <= 100
        
        # Check player stats
        stats = data["player_stats"]
        assert "player_name" in stats
        assert "avg_kills" in stats
        assert "recent_matches" in stats
    
    def test_prediction_extreme_values(self):
        """Test prediction with extreme/unrealistic values"""
        # Test extremely high value
        prediction_data = {
            "player_name": "PatkicaA",
            "prop_type": "kills",
            "prop_value": 1000,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "LESS"
        assert data["confidence"] == 99.9
        assert "unrealistically high" in data["reasoning"]
        
        # Test negative value
        prediction_data["prop_value"] = -5
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "MORE"
        assert data["confidence"] == 99.9
        assert "negative" in data["reasoning"]
    
    def test_prediction_invalid_player(self):
        """Test prediction with invalid player name"""
        prediction_data = {
            "player_name": "InvalidPlayer123",
            "prop_type": "kills",
            "prop_value": 4.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data["detail"]
        assert "Player not found" in data["detail"]["error"]
        assert "suggestions" in data["detail"]
    
    def test_player_stats(self):
        """Test player stats endpoint"""
        response = requests.get(f"{API_BASE}/playerstats/PatkicaA")
        assert response.status_code == 200
        data = response.json()
        
        assert "player_name" in data
        assert "recent_matches" in data
        assert "avg_kills" in data
        assert "win_rate" in data
        assert "data_source" in data
    
    def test_model_info(self):
        """Test model information endpoint"""
        response = requests.get(f"{API_BASE}/model/info")
        assert response.status_code == 200
        data = response.json()
        
        assert "model_type" in data
        assert "n_features" in data
        assert "feature_names" in data
        assert "status" in data
    
    def test_feature_importance(self):
        """Test feature importance endpoint"""
        response = requests.get(f"{API_BASE}/model/features")
        assert response.status_code == 200
        data = response.json()
        
        # Should return feature importance dictionary
        assert isinstance(data, dict)
        if data:  # If model has feature importance
            assert len(data) > 0
    
    def test_different_prop_types(self):
        """Test predictions with different prop types"""
        prop_types = ["kills", "assists", "cs", "deaths", "gold", "damage"]
        
        for prop_type in prop_types:
            prediction_data = {
                "player_name": "PatkicaA",
                "prop_type": prop_type,
                "prop_value": 5.0,
                "opponent": "Gen.G",
                "tournament": "LCS",
                "map_number": 1
            }
            
            response = requests.post(
                f"{API_BASE}/predict",
                headers={"Content-Type": "application/json"},
                json=prediction_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "reasoning" in data
    
    def test_statistical_reasoning(self):
        """Test that reasoning includes statistical analysis"""
        prediction_data = {
            "player_name": "Burdol",
            "prop_type": "kills",
            "prop_value": 6.5,
            "opponent": "Gen.G",
            "tournament": "LCS",
            "map_number": 1
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        reasoning = data["reasoning"]
        
        # Check for statistical analysis keywords
        statistical_indicators = [
            "trend", "volatility", "standard deviation", 
            "average", "recent form", "season average"
        ]
        
        # At least one statistical indicator should be present
        assert any(indicator in reasoning.lower() for indicator in statistical_indicators)

    def test_map_range_aggregation(self):
        """Test that map range aggregation produces different results than single map"""
        # Test single map prediction
        prediction_data_single = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1],
            "start_map": 1,
            "end_map": 1
        }
        
        response_single = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data_single
        )
        
        assert response_single.status_code == 200
        data_single = response_single.json()
        single_avg_kills = data_single.get("player_stats", {}).get("avg_kills", 0)
        single_recent_kills = data_single.get("player_stats", {}).get("recent_kills_avg", 0)
        
        # Test map range prediction
        prediction_data_range = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response_range = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data_range
        )
        
        assert response_range.status_code == 200
        data_range = response_range.json()
        range_avg_kills = data_range.get("player_stats", {}).get("avg_kills", 0)
        range_recent_kills = data_range.get("player_stats", {}).get("recent_kills_avg", 0)
        
        # Check if we're getting actual data
        assert single_avg_kills > 0, "Single map should have non-zero avg kills"
        assert range_avg_kills > 0, "Map range should have non-zero avg kills"
        
        # Check if the values are different
        kills_diff = abs(single_avg_kills - range_avg_kills)
        recent_kills_diff = abs(single_recent_kills - range_recent_kills)
        
        assert kills_diff > 0.01 or recent_kills_diff > 0.01, "Map range aggregation should produce different values"

    def test_verbose_prediction(self):
        """Test verbose prediction mode with detailed reasoning"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        # Test with verbose=True
        response = requests.post(
            f"{API_BASE}/predict?verbose=true",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        reasoning = data.get("reasoning", "").lower()
        
        # Check for verbose indicators
        verbose_indicators = [
            "coefficient of variation", "analysis based on", 
            "using", "model uses", "engineered features"
        ]
        
        assert any(indicator in reasoning for indicator in verbose_indicators), "Verbose mode should include detailed analysis"

    def test_model_transparency(self):
        """Test model transparency features (model_mode, rule_override, scaler_status)"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for transparency fields
        transparency_fields = ["model_mode", "rule_override", "scaler_status"]
        for field in transparency_fields:
            assert field in data, f"Missing transparency field: {field}"
        
        # Check field types
        assert isinstance(data["model_mode"], str), "model_mode should be string"
        assert isinstance(data["rule_override"], bool), "rule_override should be boolean"
        assert isinstance(data["scaler_status"], str), "scaler_status should be string"

    def test_feature_vector_alignment(self):
        """Test that feature count matches model expectations"""
        # Test model info endpoint
        response = requests.get(f"{API_BASE}/model/info")
        assert response.status_code == 200
        data = response.json()
        
        assert "n_features" in data, "Model info should include n_features"
        assert "feature_names" in data, "Model info should include feature_names"
        
        # Test feature importance endpoint
        response = requests.get(f"{API_BASE}/model/features")
        assert response.status_code == 200
        data = response.json()
        
        if "feature_importance" in data and data["feature_importance"]:
            # Check that feature importance has the expected number of features
            feature_count = len(data["feature_importance"])
            assert feature_count > 0, "Feature importance should have features"

    def test_map_range_reasoning(self):
        """Test that map range is mentioned in reasoning"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        reasoning = data.get("reasoning", "").lower()
        
        # Check for map range indicators
        map_indicators = ["maps 1-2", "map range", "maps 1-3", "maps 1-5"]
        assert any(indicator in reasoning for indicator in map_indicators), "Reasoning should mention map range"

    def test_confidence_calibration(self):
        """Test that confidence values are properly calibrated"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check confidence bounds
        confidence = data.get("confidence", 0)
        assert 10 <= confidence <= 95, f"Confidence {confidence} should be between 10 and 95"

    def test_data_source_indicators(self):
        """Test that data source is properly indicated"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check data source field
        assert "data_source" in data, "Response should include data_source"
        data_source = data["data_source"]
        assert data_source in ["oracles_elixir", "model_prediction", "extreme_value_check", "statistical_check"], f"Unexpected data_source: {data_source}"

    def test_prediction_consistency(self):
        """Test that predictions are consistent for same inputs"""
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills",
            "prop_value": 5.5,
            "opponent": "Hanwha Life Esports Challengers",
            "tournament": "LCS",
            "map_range": [1, 2],
            "start_map": 1,
            "end_map": 2
        }
        
        # Make two identical requests
        response1 = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        response2 = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Predictions should be identical
        assert data1["prediction"] == data2["prediction"], "Predictions should be identical"
        assert abs(data1["confidence"] - data2["confidence"]) < 0.1, "Confidence should be nearly identical"

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        # Test with missing required fields
        prediction_data = {
            "player_name": "Garden",
            "prop_type": "kills"
            # Missing prop_value
        }
        
        response = requests.post(
            f"{API_BASE}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )
        
        # Should return 422 (validation error) or 400 (bad request)
        assert response.status_code in [400, 422], f"Expected 400 or 422, got {response.status_code}"

    def test_player_validation_edge_cases(self):
        """Test player validation with edge cases"""
        # Test case sensitivity
        response = requests.get(f"{API_BASE}/players/validate/garden")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True, "Case-insensitive player validation should work"
        
        # Test with special characters
        response = requests.get(f"{API_BASE}/players/validate/Invalid@Player#123")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == False, "Invalid player should be rejected"
        assert "suggestions" in data, "Should provide suggestions for invalid players"

    def test_teams_endpoint_comprehensive(self):
        """Test teams endpoint comprehensively"""
        response = requests.get(f"{API_BASE}/teams")
        assert response.status_code == 200
        data = response.json()
        
        assert "teams" in data, "Response should include teams array"
        assert "total_teams" in data, "Response should include total_teams count"
        assert isinstance(data["teams"], list), "teams should be a list"
        assert len(data["teams"]) > 0, "Should have at least one team"
        
        # Check that teams are strings and not empty
        for team in data["teams"]:
            assert isinstance(team, str), "Team names should be strings"
            assert len(team.strip()) > 0, "Team names should not be empty"

    def test_player_stats_comprehensive(self):
        """Test player stats endpoint comprehensively"""
        response = requests.get(f"{API_BASE}/playerstats/Garden")
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ["player_name", "recent_matches", "avg_kills", "avg_assists", "avg_cs", "win_rate", "data_source"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data["player_name"], str), "player_name should be string"
        assert isinstance(data["recent_matches"], list), "recent_matches should be list"
        assert isinstance(data["avg_kills"], (int, float)), "avg_kills should be numeric"
        assert isinstance(data["win_rate"], (int, float)), "win_rate should be numeric"
        assert isinstance(data["data_source"], str), "data_source should be string"
        
        # Check value ranges
        assert 0 <= data["win_rate"] <= 1, "win_rate should be between 0 and 1"
        assert data["avg_kills"] >= 0, "avg_kills should be non-negative"
        assert data["avg_assists"] >= 0, "avg_assists should be non-negative"
        assert data["avg_cs"] >= 0, "avg_cs should be non-negative"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 