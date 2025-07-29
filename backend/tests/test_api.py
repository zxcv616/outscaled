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

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 