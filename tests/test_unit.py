#!/usr/bin/env python3
"""
Unit tests for Outscaled.gg ML components
Run this inside the Docker container: docker-compose exec backend python tests/test_unit.py
"""

import sys
import os
import unittest
import numpy as np

# Add the app directory to the path
sys.path.append('/app')

from app.ml.feature_engineering import FeatureEngineer, FeaturePipeline
from app.ml.predictor import PropPredictor

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        self.feature_engineer = FeatureEngineer()
        self.feature_pipeline = FeaturePipeline()
    
    def test_feature_count(self):
        """Test that we have the correct number of features"""
        feature_names = self.feature_engineer.get_feature_names()
        self.assertEqual(len(feature_names), 31, f"Expected 31 features, got {len(feature_names)}")
    
    def test_feature_engineering(self):
        """Test feature engineering with sample data"""
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'avg_deaths': 2.0,
            'avg_gold': 12000.0,
            'avg_damage': 15000.0,
            'avg_vision': 25.0,
            'recent_kills_avg': 6.0,
            'recent_assists_avg': 4.0,
            'recent_cs_avg': 220.0,
            'win_rate': 0.6,
            'avg_kda': 4.0,
            'avg_gpm': 600.0,
            'avg_kp_percent': 65.0,
            'position': 'adc',
            'recent_matches': [
                {'kills': 5, 'assists': 3, 'cs': 200, 'vision': 25, 'champion': 'Jinx'},
                {'kills': 7, 'assists': 5, 'cs': 240, 'vision': 30, 'champion': 'Kai\'Sa'}
            ]
        }
        
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 4.5,
            'map_range': [1, 2],
            'start_map': 1,
            'end_map': 2,
            'opponent': 'T1',
            'tournament': 'LCS'
        }
        
        features = self.feature_engineer.engineer_features(player_stats, prop_request)
        
        self.assertEqual(len(features), 31, "Should have 31 features")
        self.assertTrue(np.all(np.isfinite(features)), "All features should be finite")
    
    def test_map_range_normalization(self):
        """Test that map-range normalization works correctly"""
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        # Test with different map ranges
        prop_request_1 = {'map_range': [1], 'prop_type': 'kills'}
        prop_request_2 = {'map_range': [1, 2], 'prop_type': 'kills'}
        
        features_1 = self.feature_engineer.engineer_features(player_stats, prop_request_1)
        features_2 = self.feature_engineer.engineer_features(player_stats, prop_request_2)
        
        # With normalization, single map should have higher values than multi-map
        # (since we divide by map count, single map = no division, multi-map = division)
        self.assertGreater(features_1[0], features_2[0], "Single map should have higher normalized values")
    
    def test_role_specific_performance(self):
        """Test role-specific performance calculation"""
        player_stats = {
            'position': 'adc',
            'recent_matches': [
                {'kills': 5, 'assists': 3, 'cs': 200, 'vision': 25, 'champion': 'Jinx'},
                {'kills': 7, 'assists': 5, 'cs': 240, 'vision': 30, 'champion': 'Kai\'Sa'}
            ]
        }
        
        prop_request = {'map_range': [1, 2], 'prop_type': 'kills'}
        
        features = self.feature_engineer.engineer_features(player_stats, prop_request)
        
        # Role-specific performance should be calculated
        self.assertTrue(np.isfinite(features[30]), "Role-specific performance should be finite")
    
    def test_extreme_value_handling(self):
        """Test handling of extreme values"""
        player_stats = {
            'avg_kills': 5.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        # Test with extreme prop value
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 999,  # Extreme value
            'map_range': [1, 2]
        }
        
        features = self.feature_engineer.engineer_features(player_stats, prop_request)
        
        # Should still produce valid features
        self.assertEqual(len(features), 31)
        self.assertTrue(np.all(np.isfinite(features)))

class TestPredictor(unittest.TestCase):
    """Test predictor functionality"""
    
    def setUp(self):
        self.predictor = PropPredictor()
    
    def test_predictor_initialization(self):
        """Test that predictor initializes correctly"""
        self.assertIsNotNone(self.predictor.model, "Model should be loaded")
        self.assertIsNotNone(self.predictor.feature_engineer, "Feature engineer should be initialized")
    
    def test_feature_count_consistency(self):
        """Test that predictor and feature engineer have consistent feature counts"""
        predictor_features = len(self.predictor.feature_engineer.get_feature_names())
        self.assertEqual(predictor_features, 31, f"Predictor should use 31 features, got {predictor_features}")
    
    def test_model_inference(self):
        """Test model inference with dummy features"""
        features = np.random.randn(31)  # 31 features
        
        prediction, confidence = self.predictor._run_model_inference(features)
        
        self.assertIn(prediction, ['MORE', 'LESS'], "Prediction should be MORE or LESS")
        self.assertGreaterEqual(confidence, 0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 100, "Confidence should be <= 100")
    
    def test_deterministic_predictions(self):
        """Test that predictions are deterministic"""
        features = np.random.randn(31)
        
        # Make two identical predictions
        pred1, conf1 = self.predictor._run_model_inference(features)
        pred2, conf2 = self.predictor._run_model_inference(features)
        
        self.assertEqual(pred1, pred2, "Predictions should be identical")
        self.assertAlmostEqual(conf1, conf2, places=1, msg="Confidence should be identical")

class TestFeaturePipeline(unittest.TestCase):
    """Test feature pipeline functionality"""
    
    def setUp(self):
        self.pipeline = FeaturePipeline()
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline.feature_engineer)
        self.assertIsNotNone(self.pipeline.scaler)
        self.assertFalse(self.pipeline.is_fitted)
    
    def test_feature_transformation(self):
        """Test feature transformation"""
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 4.5,
            'map_range': [1, 2]
        }
        
        features = self.pipeline.transform(player_stats, prop_request)
        
        self.assertEqual(len(features), 31, "Should have 31 features")
        self.assertTrue(np.all(np.isfinite(features)), "All features should be finite")

def run_tests():
    """Run all unit tests"""
    print("🧪 Running Outscaled.gg Unit Tests")
    print("=" * 40)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineering))
    test_suite.addTest(unittest.makeSuite(TestPredictor))
    test_suite.addTest(unittest.makeSuite(TestFeaturePipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("=" * 40)
    print(f"📊 Test Results: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} tests passed")
    
    if result.wasSuccessful():
        print("🎉 All unit tests passed!")
        return 0
    else:
        print("❌ Some unit tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 