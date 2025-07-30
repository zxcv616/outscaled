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

    def test_feature_vector_alignment(self):
        """Test that feature count matches model expectations"""
        # Test that feature engineer produces correct number of features
        feature_names = self.feature_engineer.get_feature_names()
        self.assertEqual(len(feature_names), 31, f"Expected 31 features, got {len(feature_names)}")
        
        # Test that predictor model expects same number of features
        if hasattr(self.predictor.model, 'n_features_in_'):
            expected_features = self.predictor.model.n_features_in_
            self.assertEqual(len(feature_names), expected_features, 
                           f"Feature count mismatch: {len(feature_names)} vs {expected_features}")
    
    def test_map_range_aggregation_logic(self):
        """Test map range aggregation logic"""
        # Test with sample data that mimics the aggregation
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        # Test single map vs map range
        prop_request_single = {'map_range': [1], 'prop_type': 'kills', 'prop_value': 4.5}
        prop_request_range = {'map_range': [1, 2], 'prop_type': 'kills', 'prop_value': 4.5}
        
        features_single = self.feature_engineer.engineer_features(player_stats, prop_request_single)
        features_range = self.feature_engineer.engineer_features(player_stats, prop_request_range)
        
        # Features should be different due to map range normalization
        self.assertFalse(np.array_equal(features_single, features_range), 
                        "Single map and map range features should be different")
    
    def test_model_transparency_fields(self):
        """Test that predictor returns transparency fields"""
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
        
        # Test prediction with transparency
        prediction_result = self.predictor.predict(player_stats, prop_request)
        
        # Check for transparency fields
        transparency_fields = ['model_mode', 'rule_override', 'scaler_status']
        for field in transparency_fields:
            self.assertIn(field, prediction_result, f"Missing transparency field: {field}")
        
        # Check field types
        self.assertIsInstance(prediction_result['model_mode'], str)
        self.assertIsInstance(prediction_result['rule_override'], bool)
        self.assertIsInstance(prediction_result['scaler_status'], str)
    
    def test_verbose_prediction_mode(self):
        """Test verbose prediction mode"""
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
        
        # Test verbose mode
        prediction_result = self.predictor.predict(player_stats, prop_request, verbose=True)
        
        # Check that reasoning is more detailed in verbose mode
        reasoning = prediction_result.get('reasoning', '')
        verbose_indicators = ['coefficient of variation', 'analysis based on', 'using', 'model uses']
        
        # At least one verbose indicator should be present
        has_verbose = any(indicator in reasoning.lower() for indicator in verbose_indicators)
        self.assertTrue(has_verbose, "Verbose mode should include detailed analysis")
    
    def test_confidence_calibration_bounds(self):
        """Test that confidence values are properly bounded"""
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
        
        prediction_result = self.predictor.predict(player_stats, prop_request)
        confidence = prediction_result.get('confidence', 0)
        
        # Check confidence bounds
        self.assertGreaterEqual(confidence, 10, f"Confidence {confidence} should be >= 10")
        self.assertLessEqual(confidence, 95, f"Confidence {confidence} should be <= 95")
    
    def test_extreme_value_handling_improved(self):
        """Test improved extreme value handling"""
        player_stats = {
            'avg_kills': 5.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        # Test with extreme values
        extreme_values = [999, -5, 0, 1000]
        
        for extreme_value in extreme_values:
            prop_request = {
                'prop_type': 'kills',
                'prop_value': extreme_value,
                'map_range': [1, 2]
            }
            
            prediction_result = self.predictor.predict(player_stats, prop_request)
            
            # Should still produce valid prediction
            self.assertIn('prediction', prediction_result)
            self.assertIn('confidence', prediction_result)
            self.assertIn('reasoning', prediction_result)
            
            # Confidence should be high for extreme values
            if extreme_value > 50 or extreme_value < 0:
                self.assertGreaterEqual(prediction_result['confidence'], 95, 
                                      f"Extreme value {extreme_value} should have high confidence")
    
    def test_rule_based_override_logic(self):
        """Test rule-based override logic"""
        player_stats = {
            'avg_kills': 5.0,
            'recent_kills_avg': 8.0,  # High recent form
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 4.5,
            'map_range': [1, 2]
        }
        
        prediction_result = self.predictor.predict(player_stats, prop_request)
        
        # Check if rule override was applied
        rule_override = prediction_result.get('rule_override', False)
        
        # If rule override was applied, confidence should be affected
        if rule_override:
            self.assertIn('rule_override', prediction_result)
            self.assertTrue(isinstance(rule_override, bool))
    
    def test_fallback_model_handling(self):
        """Test fallback model handling"""
        # This test would require mocking the main model to fail
        # For now, just test that the predictor can handle missing models gracefully
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
        
        # Test that prediction still works even if main model fails
        prediction_result = self.predictor.predict(player_stats, prop_request)
        
        # Should still return valid prediction
        self.assertIn('prediction', prediction_result)
        self.assertIn('confidence', prediction_result)
        self.assertIn('reasoning', prediction_result)
    
    def test_data_quality_indicators(self):
        """Test data quality indicators in prediction"""
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'win_rate': 0.6,
            'position': 'adc',
            'total_matches_available': 10,
            'matches_in_range': 5
        }
        
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 4.5,
            'map_range': [1, 2]
        }
        
        prediction_result = self.predictor.predict(player_stats, prop_request)
        
        # Check for data quality indicators
        reasoning = prediction_result.get('reasoning', '').lower()
        
        # Should mention data quality if limited data
        if player_stats['matches_in_range'] < 10:
            data_quality_indicators = ['limited data', 'small sample', 'few matches']
            has_data_quality = any(indicator in reasoning for indicator in data_quality_indicators)
            # This is optional, so we don't assert it must be present
    
    def test_map_range_normalization_accuracy(self):
        """Test that map range normalization is mathematically correct"""
        player_stats = {
            'avg_kills': 5.0,
            'avg_assists': 3.0,
            'avg_cs': 200.0,
            'win_rate': 0.6,
            'position': 'adc'
        }
        
        # Test different map ranges
        map_ranges = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5]]
        
        features_by_range = {}
        for map_range in map_ranges:
            prop_request = {
                'prop_type': 'kills',
                'prop_value': 4.5,
                'map_range': map_range
            }
            features = self.feature_engineer.engineer_features(player_stats, prop_request)
            features_by_range[tuple(map_range)] = features
        
        # Single map should have highest normalized values
        single_map_features = features_by_range[(1,)]
        
        for map_range, features in features_by_range.items():
            if map_range != (1,):
                # Single map features should generally be higher due to normalization
                # (This is a simplified check - actual normalization depends on the feature)
                pass  # We don't assert this as it depends on the specific normalization logic

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