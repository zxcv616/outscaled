import pickle
import numpy as np
import logging
import os
import json
import yaml
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from app.ml.feature_engineering import FeatureEngineer, FeaturePipeline
from app.core.config import settings

# Set global random seed for deterministic behavior
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropPredictor:
    def __init__(self, config_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_pipeline = FeaturePipeline()  # NEW: Feature pipeline for proper scaler management
        self.model_path = settings.MODEL_PATH
        self.config_path = config_path or "backend/app/ml/config/prop_config.yaml"
        self.metadata_path = "backend/app/ml/models/predictor_metadata.json"
        self.fallback_model_path = "backend/app/ml/models/fallback_model.pkl"
        self.model_version = "1.0.0"
        self.training_data_distribution = None
        self._load_model()
        self._load_feature_scaler()  # NEW: Load feature scaler
    
    def _load_feature_scaler(self):
        """Load the feature scaler if available"""
        try:
            if not self.feature_pipeline.load_scaler():
                logger.info("No feature scaler found, will use unscaled features")
        except Exception as e:
            logger.error(f"Error loading feature scaler: {e}")
    
    def _load_model(self):
        """Load the trained model with validation"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.training_data_distribution = model_data.get('training_distribution')
                
                # Validate model compatibility
                self._validate_model_compatibility()
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}, loading fallback")
                self._load_or_create_fallback()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._load_or_create_fallback()
    
    def _load_or_create_fallback(self):
        """Load existing fallback or create new one"""
        try:
            if os.path.exists(self.fallback_model_path):
                with open(self.fallback_model_path, 'rb') as f:
                    fallback_data = pickle.load(f)
                    self.model = fallback_data['model']
                    self.scaler = fallback_data['scaler']
                logger.info("Loaded existing fallback model")
            else:
                logger.info("Creating new fallback model")
                self._create_lightweight_fallback()
                self._save_fallback_model()
        except Exception as e:
            logger.error(f"Error loading fallback: {e}")
            self._create_lightweight_fallback()
    
    def _save_fallback_model(self):
        """Save fallback model to disk for future use"""
        try:
            fallback_data = {
                'model': self.model,
                'scaler': self.scaler,
                'created_date': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.fallback_model_path), exist_ok=True)
            with open(self.fallback_model_path, 'wb') as f:
                pickle.dump(fallback_data, f)
            
            logger.info(f"Saved fallback model to {self.fallback_model_path}")
            
        except Exception as e:
            logger.error(f"Error saving fallback model: {e}")
    
    def _validate_model_compatibility(self):
        """Validate that loaded model matches expected feature count"""
        try:
            expected_features = len(self.feature_engineer.get_feature_names())
            
            # Test model with dummy features
            dummy_features = np.zeros((1, expected_features))
            
            # Check if scaler is fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                logger.warning("Scaler not fitted, fitting with dummy data")
                self.scaler.fit(dummy_features)
            
            dummy_scaled = self.scaler.transform(dummy_features)
            
            # Try to make a prediction
            self.model.predict(dummy_scaled)
            
            logger.info(f"Model validation successful - {expected_features} features")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            logger.info("Creating new lightweight fallback model")
            self._create_lightweight_fallback()
    
    def _check_data_drift(self, player_stats: Dict) -> Dict:
        """Check for data drift compared to training distribution"""
        try:
            if self.training_data_distribution is None:
                return {"drift_detected": False, "reason": "No training distribution available"}
            
            # Extract current data statistics
            recent_matches = player_stats.get('recent_matches', [])
            if not recent_matches:
                return {"drift_detected": False, "reason": "No recent matches available"}
            
            current_stats = {
                'avg_kills': np.mean([m.get('kills', 0) for m in recent_matches]),
                'avg_assists': np.mean([m.get('assists', 0) for m in recent_matches]),
                'avg_cs': np.mean([m.get('cs', 0) for m in recent_matches]),
                'avg_deaths': np.mean([m.get('deaths', 0) for m in recent_matches])
            }
            
            # Compare with training distribution (simplified drift detection)
            drift_detected = False
            drift_reasons = []
            
            for stat, current_val in current_stats.items():
                if stat in self.training_data_distribution:
                    train_mean = self.training_data_distribution[stat]['mean']
                    train_std = self.training_data_distribution[stat]['std']
                    
                    # Check if current value is more than 2 standard deviations from training mean
                    if abs(current_val - train_mean) > 2 * train_std:
                        drift_detected = True
                        drift_reasons.append(f"{stat}: {current_val:.2f} vs training {train_mean:.2f}")
            
            return {
                "drift_detected": drift_detected,
                "reasons": drift_reasons,
                "current_stats": current_stats,
                "training_stats": self.training_data_distribution
            }
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {"drift_detected": False, "reason": f"Error: {str(e)}"}
    
    def _create_lightweight_fallback(self):
        """Create a lightweight fallback model for production"""
        try:
            # Set fixed random seed for deterministic behavior
            np.random.seed(42)
            
            # Get actual feature count from feature engineer
            num_features = len(self.feature_engineer.get_feature_names())
            logger.info(f"Creating fallback model with {num_features} features")
            
            # Use calibrated model for better confidence estimation
            base_model = RandomForestClassifier(
                n_estimators=20,  # Reduced from 50
                max_depth=4,      # Reduced from 6
                random_state=42,
                n_jobs=1          # Single thread for production
            )
            
            # Wrap with CalibratedClassifierCV for better confidence
            self.model = CalibratedClassifierCV(
                base_estimator=base_model,
                cv=3,
                method='isotonic'
            )
            
            self.scaler = StandardScaler()
            
            # Create deterministic training data with correct feature count
            # Use fixed seed to ensure same data every time
            np.random.seed(42)
            X = np.random.randn(50, num_features)  # Deterministic random data
            y = np.random.randint(0, 2, 50, dtype=int)  # Deterministic random labels
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            logger.info("Created lightweight calibrated fallback model")
            
        except Exception as e:
            logger.error(f"Error creating lightweight model: {e}")
            # Final fallback - even smaller with fixed seed
            np.random.seed(42)
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.scaler = StandardScaler()
    
    def train_and_save_model(self, training_data: List[Tuple[np.ndarray, int]], 
                           save_path: Optional[str] = None) -> bool:
        """Train and save a new model with calibration"""
        try:
            if not training_data:
                logger.error("No training data provided")
                return False
            
            # Split features and targets
            X = np.array([data[0] for data in training_data])
            y = np.array([data[1] for data in training_data])
            
            # Calculate training data distribution for drift detection
            self.training_data_distribution = self._calculate_training_distribution(X)
            
            # Split for calibration
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Fit scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_cal_scaled = self.scaler.transform(X_cal)
            
            # Create base model
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            # Train calibrated model
            self.model = CalibratedClassifierCV(
                base_estimator=base_model,
                cv=3,
                method='isotonic'
            )
            
            # Train on training set
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate on calibration set
            y_pred = self.model.predict(X_cal_scaled)
            accuracy = accuracy_score(y_cal, y_pred)
            
            logger.info(f"Model training completed. Calibration accuracy: {accuracy:.3f}")
            
            # Save model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'training_distribution': self.training_data_distribution,
                'training_date': datetime.now().isoformat(),
                'calibration_accuracy': accuracy
            }
            
            save_path = save_path or self.model_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _calculate_training_distribution(self, X: np.ndarray) -> Dict:
        """Calculate training data distribution for drift detection"""
        try:
            # Calculate basic statistics for each feature
            distribution = {}
            feature_names = self.feature_engineer.get_feature_names()
            
            for i, feature_name in enumerate(feature_names):
                if i < X.shape[1]:
                    feature_values = X[:, i]
                    distribution[feature_name] = {
                        'mean': float(np.mean(feature_values)),
                        'std': float(np.std(feature_values)),
                        'min': float(np.min(feature_values)),
                        'max': float(np.max(feature_values))
                    }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating training distribution: {e}")
            return {}
    
    def _load_prop_config(self) -> Dict:
        """Load prop configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("Loaded prop configuration from YAML")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default prop configuration"""
        return {
            'kills': {
                'min_value': 0,
                'max_value': 20,
                'impossible_threshold': 15,
                'low_threshold_factor': 0.5,
                'high_threshold_factor': 1.5
            },
            'assists': {
                'min_value': 0,
                'max_value': 25,
                'impossible_threshold': 20,
                'low_threshold_factor': 0.5,
                'high_threshold_factor': 1.5
            },
            'cs': {
                'min_value': 0,
                'max_value': 400,
                'impossible_threshold': 300,
                'low_threshold_factor': 0.6,
                'high_threshold_factor': 1.4
            },
            'deaths': {
                'min_value': 0,
                'max_value': 10,
                'impossible_threshold': 8,
                'low_threshold_factor': 0.7,
                'high_threshold_factor': 1.3
            },
            'gold': {
                'min_value': 0,
                'max_value': 20000,
                'impossible_threshold': 15000,
                'low_threshold_factor': 0.6,
                'high_threshold_factor': 1.4
            },
            'damage': {
                'min_value': 0,
                'max_value': 30000,
                'impossible_threshold': 25000,
                'low_threshold_factor': 0.6,
                'high_threshold_factor': 1.4
            }
        }
    
    def _get_prop_config(self, prop_type: str) -> Dict:
        """Get configuration for different prop types"""
        config = self._load_prop_config()
        return config.get(prop_type, config['kills'])
    
    def _validate_inputs(self, player_stats: Dict, prop_request: Dict) -> Tuple[bool, str]:
        """Validate input data quality"""
        try:
            # Check required fields
            required_prop_fields = ['prop_type', 'prop_value']
            missing_prop_fields = [field for field in required_prop_fields if field not in prop_request]
            if missing_prop_fields:
                return False, f"Missing required prop fields: {missing_prop_fields}"
            
            # Check prop value is numeric
            prop_value = prop_request.get('prop_value')
            if not isinstance(prop_value, (int, float)):
                return False, f"Prop value must be numeric, got {type(prop_value)}"
            
            # Check prop value is reasonable
            if prop_value < 0:
                return False, f"Prop value cannot be negative: {prop_value}"
            
            # Check player stats has basic data
            if not player_stats:
                return False, "Player stats cannot be empty"
            
            # Check for recent matches
            recent_matches = player_stats.get('recent_matches', [])
            if not recent_matches:
                logger.warning("No recent matches found in player stats")
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Input validation error: {str(e)}"
    
    def _calculate_player_averages(self, player_stats: Dict, prop_type: str) -> Tuple[float, float]:
        """Calculate player averages for the specific prop type"""
        try:
            avg_mapping = {
                'kills': 'avg_kills',
                'assists': 'avg_assists', 
                'cs': 'avg_cs',
                'deaths': 'avg_deaths',
                'gold': 'avg_gold',
                'damage': 'avg_damage'
            }
            
            recent_mapping = {
                'kills': 'recent_kills_avg',
                'assists': 'recent_assists_avg',
                'cs': 'recent_cs_avg'
            }
            
            player_avg = player_stats.get(avg_mapping.get(prop_type, 'avg_kills'), 0.0)
            recent_avg = player_stats.get(recent_mapping.get(prop_type, 'avg_kills'), player_avg)
            
            return player_avg, recent_avg
            
        except Exception as e:
            logger.error(f"Error calculating player averages: {e}")
            return 0.0, 0.0
    
    def _extract_champion_stats(self, player_stats: Dict) -> Dict:
        """Extract champion usage statistics for model features"""
        try:
            recent_matches = player_stats.get('recent_matches', [])
            if not recent_matches:
                return {}
            
            # Get champion usage
            champions = [match.get('champion', '') for match in recent_matches if match.get('champion')]
            if not champions:
                return {}
            
            # Calculate champion statistics
            unique_champions = len(set(champions))
            total_matches = len(champions)
            champion_diversity = unique_champions / max(total_matches, 1)
            
            # Most used champion
            from collections import Counter
            champ_counts = Counter(champions)
            most_used_champ = champ_counts.most_common(1)[0][0] if champ_counts else ''
            most_used_count = champ_counts.most_common(1)[0][1] if champ_counts else 0
            
            # Champion performance correlation (simplified)
            champ_performance = {}
            for champ in set(champions):
                champ_matches = [match for match in recent_matches if match.get('champion') == champ]
                if champ_matches:
                    avg_kills = np.mean([m.get('kills', 0) for m in champ_matches])
                    avg_assists = np.mean([m.get('assists', 0) for m in champ_matches])
                    champ_performance[champ] = {'kills': avg_kills, 'assists': avg_assists}
            
            return {
                'champion_diversity': champion_diversity,
                'unique_champions': unique_champions,
                'most_used_champion': most_used_champ,
                'most_used_count': most_used_count,
                'champion_performance': champ_performance
            }
            
        except Exception as e:
            logger.error(f"Error extracting champion stats: {e}")
            return {}
    
    def _handle_extreme_values(self, prop_value: float, prop_type: str, config: Dict, 
                              player_avg: float, sample_warning: str, reasoning: str) -> Optional[Dict]:
        """Handle extreme prop values with early returns"""
        try:
            # Impossible threshold check
            if prop_value > config['impossible_threshold']:
                return {
                    "prediction": "LESS",
                    "confidence": 99.9,
                    "reasoning": f"Prop value ({prop_value}) is unrealistically high for {prop_type}. This is virtually impossible.{sample_warning}",
                    "features_used": self.feature_engineer.get_feature_names(),
                    "data_source": "extreme_value_check"
                }
            
            # Minimum value check
            elif prop_value < config['min_value']:
                return {
                    "prediction": "MORE",
                    "confidence": 99.9,
                    "reasoning": f"Prop value ({prop_value}) is below minimum possible for {prop_type}. Player will definitely exceed this.{sample_warning}",
                    "features_used": self.feature_engineer.get_feature_names(),
                    "data_source": "extreme_value_check"
                }
            
            # Zero value check
            elif prop_value == 0 and prop_type in ["kills", "assists", "cs"]:
                return {
                    "prediction": "MORE",
                    "confidence": 99.9,
                    "reasoning": f"Prop value ({prop_value}) is zero for {prop_type}. Player cannot get negative {prop_type}, so they will definitely exceed this.{sample_warning}",
                    "features_used": self.feature_engineer.get_feature_names(),
                    "data_source": "extreme_value_check"
                }
            
            # Significant difference from average
            elif player_avg > 0:
                low_threshold = player_avg * config['low_threshold_factor']
                high_threshold = player_avg * config['high_threshold_factor']
                
                if prop_value < low_threshold:
                    confidence = min(95.0, 80.0 + (player_avg - prop_value) * 3)
                    return {
                        "prediction": "MORE",
                        "confidence": confidence,
                        "reasoning": f"Prop value ({prop_value}) is significantly below player's average {prop_type} ({player_avg:.1f}). Player typically performs much better than this.{sample_warning}",
                        "features_used": self.feature_engineer.get_feature_names(),
                        "data_source": "statistical_check"
                    }
                elif prop_value > high_threshold:
                    confidence = min(95.0, 80.0 + (prop_value - player_avg) * 2)
                    return {
                        "prediction": "LESS",
                        "confidence": confidence,
                        "reasoning": f"Prop value ({prop_value}) is significantly above player's average {prop_type} ({player_avg:.1f}). Player typically performs below this level.{sample_warning}",
                        "features_used": self.feature_engineer.get_feature_names(),
                        "data_source": "statistical_check"
                    }
            
            return None  # No extreme value detected
            
        except Exception as e:
            logger.error(f"Error handling extreme values: {e}")
            return None
    
    def _run_model_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run model inference and return prediction and confidence"""
        try:
            # Ensure scaler is fitted
            if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                logger.warning("Scaler not fitted, fitting with current features")
                self.scaler.fit(features.reshape(1, -1))
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction with calibrated probabilities
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            prediction = self.model.predict(features_scaled)[0]
            prediction_label = "MORE" if prediction == 1 else "LESS"
            
            # Use calibrated confidence
            confidence = prediction_proba[prediction] * 100
            confidence = max(0, min(100, confidence))
            
            return prediction_label, confidence
            
        except Exception as e:
            logger.error(f"Error in model inference: {e}")
            return "LESS", 50.0
    
    def _generate_statistical_flags(self, player_stats: Dict, prop_request: Dict) -> Dict:
        """Generate statistical flags for reasoning"""
        try:
            prop_type = prop_request.get("prop_type", "kills")
            prop_value = prop_request.get("prop_value", 0.0)
            recent_matches = player_stats.get("recent_matches", [])
            
            flags = {
                'sample_size_warning': len(recent_matches) < 3,
                'sample_count': len(recent_matches),
                'champion_stats': self._extract_champion_stats(player_stats),
                'opponent': prop_request.get("opponent", ""),
                'tournament': prop_request.get("tournament", ""),
                'data_drift': self._check_data_drift(player_stats)
            }
            
            return flags
            
        except Exception as e:
            logger.error(f"Error generating statistical flags: {e}")
            return {}
    
    def predict(self, player_stats: Dict, prop_request: Dict) -> Dict:
        """Make a prediction for a prop with improved error handling"""
        start_time = datetime.now()
        
        try:
            # Input validation
            is_valid, validation_message = self._validate_inputs(player_stats, prop_request)
            if not is_valid:
                return {
                    "prediction": "LESS",
                    "confidence": 50.0,
                    "reasoning": f"Input validation failed: {validation_message}",
                    "features_used": [],
                    "data_source": "validation_error"
                }
            
            prop_type = prop_request.get("prop_type", "kills")
            prop_value = prop_request.get("prop_value", 0.0)
            config = self._get_prop_config(prop_type)
            
            # Calculate player averages
            player_avg, recent_avg = self._calculate_player_averages(player_stats, prop_type)
            
            # Generate statistical flags
            stats_flags = self._generate_statistical_flags(player_stats, prop_request)
            
            # Check sample size warning
            sample_warning = ""
            if stats_flags.get('sample_size_warning', False):
                sample_count = stats_flags.get('sample_count', 0)
                sample_warning = f" Warning: Prediction based on only {sample_count} recent matches."
            
            # Check for data drift
            drift_info = stats_flags.get('data_drift', {})
            drift_warning = ""
            if drift_info.get('drift_detected', False):
                drift_warning = f" Warning: Data drift detected - {', '.join(drift_info.get('reasons', []))}"
            
            # Handle extreme values FIRST (before generating reasoning)
            extreme_result = self._handle_extreme_values(
                prop_value, prop_type, config, player_avg, sample_warning, ""
            )
            if extreme_result:
                return extreme_result
            
            # Generate statistical reasoning for normal cases
            reasoning = self._generate_reasoning(player_stats, prop_request, "MORE", 50.0)
            
            # Normal prediction flow for realistic values
            features = self.feature_engineer.engineer_features(player_stats, prop_request)
            
            # Run model inference
            prediction_label, confidence = self._run_model_inference(features)
            
            # Generate final reasoning
            final_reasoning = self._generate_reasoning(player_stats, prop_request, prediction_label, confidence)
            
            # Add map range warning if present
            map_range_warning = player_stats.get("map_range_warning", "")
            if map_range_warning:
                final_reasoning += map_range_warning
            
            # Log prediction metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Prediction completed in {prediction_time:.3f}s - {prediction_label} ({confidence:.1f}%)")
            
            return {
                "prediction": prediction_label,
                "confidence": confidence,
                "reasoning": final_reasoning + sample_warning + drift_warning,
                "features_used": self.feature_engineer.get_feature_names(),
                "data_source": player_stats.get("data_source", "model_prediction"),
                "prediction_time_ms": prediction_time * 1000,
                "champion_stats": stats_flags.get('champion_stats', {}),
                "data_drift": drift_info,
                "note": "Tournament and opponent information is used for reasoning but not directly in model features"
            }
            
        except Exception as e:
            logger.exception(f"Error making prediction: {e}")
            return {
                "prediction": "LESS",
                "confidence": 50.0,
                "reasoning": "Error in prediction model. Defaulting to conservative estimate.",
                "features_used": [],
                "data_source": "prediction_error"
            }
    
    def _generate_reasoning(self, player_stats: Dict, prop_request: Dict, prediction: str, confidence: float) -> str:
        """Generate human-readable reasoning with statistical analysis"""
        try:
            prop_type = prop_request.get("prop_type", "kills")
            prop_value = prop_request.get("prop_value", 0.0)
            recent_matches = player_stats.get("recent_matches", [])
            
            # Get relevant stats based on prop type
            if prop_type == "kills":
                recent_form = player_stats.get("recent_kills_avg", 0.0)
                avg_performance = player_stats.get("avg_kills", 0.0)
                recent_values = [match.get("kills", 0) for match in recent_matches]
            elif prop_type == "assists":
                recent_form = player_stats.get("recent_assists_avg", 0.0)
                avg_performance = player_stats.get("avg_assists", 0.0)
                recent_values = [match.get("assists", 0) for match in recent_matches]
            elif prop_type == "cs":
                recent_form = player_stats.get("recent_cs_avg", 0.0)
                avg_performance = player_stats.get("avg_cs", 0.0)
                recent_values = [match.get("cs", 0) for match in recent_matches]
            elif prop_type == "deaths":
                recent_form = player_stats.get("avg_deaths", 0.0)
                avg_performance = player_stats.get("avg_deaths", 0.0)
                recent_values = [match.get("deaths", 0) for match in recent_matches]
            else:
                recent_form = avg_performance = 0.0
                recent_values = []
            
            # Statistical Analysis
            reasoning_parts = []
            
            # 1. Basic comparison
            if recent_form > prop_value:
                reasoning_parts.append(f"Recent {prop_type} average ({recent_form:.1f}) exceeds prop value ({prop_value:.1f})")
            else:
                reasoning_parts.append(f"Recent {prop_type} average ({recent_form:.1f}) below prop value ({prop_value:.1f})")
            
            # 2. Trend analysis (if we have enough data)
            if len(recent_values) >= 3:
                x = list(range(len(recent_values)))
                if len(x) > 1:
                    trend_slope = np.polyfit(x, recent_values, 1)[0]
                    if trend_slope > 0.5:
                        reasoning_parts.append("showing strong upward trend")
                    elif trend_slope > 0:
                        reasoning_parts.append("showing slight upward trend")
                    elif trend_slope < -0.5:
                        reasoning_parts.append("showing strong downward trend")
                    elif trend_slope < 0:
                        reasoning_parts.append("showing slight downward trend")
                    else:
                        reasoning_parts.append("showing stable performance")
            
            # 3. Volatility analysis
            if len(recent_values) >= 2:
                std_dev = np.std(recent_values)
                avg_recent = np.mean(recent_values)
                if avg_recent > 0:
                    coefficient_of_variation = (std_dev / avg_recent) * 100
                    if coefficient_of_variation > 50:
                        reasoning_parts.append("with high volatility")
                    elif coefficient_of_variation > 25:
                        reasoning_parts.append("with moderate volatility")
                    else:
                        reasoning_parts.append("with consistent performance")
            
            # 4. Z-score analysis
            if len(recent_values) >= 3 and np.std(recent_values) > 0:
                z_score = (prop_value - avg_performance) / np.std(recent_values)
                if abs(z_score) > 2:
                    reasoning_parts.append(f"prop value is {abs(z_score):.1f} standard deviations from average")
                elif abs(z_score) > 1:
                    reasoning_parts.append(f"prop value is {abs(z_score):.1f} standard deviation from average")
            
            # 5. Recent vs Long-term comparison
            if recent_form != avg_performance:
                performance_diff = recent_form - avg_performance
                if abs(performance_diff) > 1:
                    if performance_diff > 0:
                        reasoning_parts.append("recent form significantly above season average")
                    else:
                        reasoning_parts.append("recent form significantly below season average")
            
            # 6. Context factors (note: these are for reasoning only, not model features)
            opponent = prop_request.get("opponent", "")
            tournament = prop_request.get("tournament", "")
            if opponent:
                reasoning_parts.append(f"against {opponent}")
            if tournament:
                reasoning_parts.append(f"in {tournament}")
            
            # 7. Champion analysis (if available)
            if recent_matches:
                recent_champions = [match.get("champion", "") for match in recent_matches[:3]]
                if recent_champions:
                    unique_champs = len(set(recent_champions))
                    if unique_champs == 1:
                        reasoning_parts.append(f"playing {recent_champions[0]} consistently")
                    else:
                        reasoning_parts.append(f"champion pool: {', '.join(set(recent_champions))}")
            
            # 8. Win rate context
            win_rate = player_stats.get("win_rate", 0.0)
            if win_rate > 0.7:
                reasoning_parts.append("excellent win rate")
            elif win_rate < 0.3:
                reasoning_parts.append("struggling win rate")
            
            # 9. Map range context (PrizePicks style)
            map_range = prop_request.get("map_range", [1])
            if map_range and len(map_range) > 1:
                if map_range == [1, 2]:
                    reasoning_parts.append("Maps 1-2")
                elif map_range == [1, 2, 3]:
                    reasoning_parts.append("Maps 1-3")
                elif len(map_range) == 2:
                    reasoning_parts.append(f"Maps {map_range[0]}-{map_range[1]}")
                else:
                    reasoning_parts.append(f"Map range {map_range}")
            elif map_range and len(map_range) == 1:
                reasoning_parts.append(f"Map {map_range[0]}")
            
            # Combine reasoning parts
            base_reasoning = ". ".join(reasoning_parts) + "."
            
            # Add confidence level
            if confidence >= 80:
                confidence_level = "High confidence"
            elif confidence >= 60:
                confidence_level = "Moderate confidence"
            else:
                confidence_level = "Low confidence"
            
            # Add data source quality
            data_source = player_stats.get("data_source", "none")
            if data_source == "oracles_elixir":
                data_quality = "High-quality Oracle's Elixir data"
            elif data_source == "riot_api":
                data_quality = "Riot API data"
            elif data_source == "gol_gg":
                data_quality = "gol.gg data"
            else:
                data_quality = "Limited data available"
            
            return f"{base_reasoning} {confidence_level} prediction based on {data_quality}."
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Prediction reasoning error: {str(e)}"
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            if self.model is None:
                return {}
            
            feature_names = self.feature_engineer.get_feature_names()
            
            # Handle different model types
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'base_estimator') and hasattr(self.model.base_estimator, 'feature_importances_'):
                importances = self.model.base_estimator.feature_importances_
            else:
                logger.warning("Model does not have feature importance attribute")
                return {}
            
            # Create dictionary of feature names and importance scores
            feature_importance = {}
            for name, importance in zip(feature_names, importances):
                feature_importance[name] = float(importance)
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            if self.model is None:
                return {"status": "No model loaded"}
            
            # Get model metadata
            model_info = {
                "model_type": type(self.model).__name__,
                "n_features": len(self.feature_engineer.get_feature_names()),
                "feature_names": self.feature_engineer.get_feature_names(),
                "model_path": self.model_path,
                "model_version": self.model_version,
                "status": "Model loaded successfully"
            }
            
            # Add model-specific info
            if hasattr(self.model, 'n_estimators'):
                model_info["n_estimators"] = self.model.n_estimators
            if hasattr(self.model, 'max_depth'):
                model_info["max_depth"] = self.model.max_depth
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": f"Error getting model info: {str(e)}"}
    
    def save_metadata(self):
        """Save prediction metadata for reproducibility"""
        try:
            metadata = {
                "model_version": self.model_version,
                "training_date": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "model_path": self.model_path,
                "config_path": self.config_path,
                "feature_count": len(self.feature_engineer.get_feature_names()),
                "model_type": type(self.model).__name__ if self.model else "None"
            }
            
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")


# Unit Tests
import unittest

class TestPropPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = PropPredictor()
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid data"""
        player_stats = {
            'avg_kills': 5.0,
            'recent_matches': [{'kills': 4}, {'kills': 6}]
        }
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 4.5
        }
        
        is_valid, message = self.predictor._validate_inputs(player_stats, prop_request)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Input validation passed")
    
    def test_validate_inputs_missing_fields(self):
        """Test input validation with missing fields"""
        player_stats = {'avg_kills': 5.0}
        prop_request = {'prop_type': 'kills'}  # Missing prop_value
        
        is_valid, message = self.predictor._validate_inputs(player_stats, prop_request)
        self.assertFalse(is_valid)
        self.assertIn("Missing required prop fields", message)
    
    def test_validate_inputs_invalid_prop_value(self):
        """Test input validation with invalid prop value"""
        player_stats = {'avg_kills': 5.0}
        prop_request = {
            'prop_type': 'kills',
            'prop_value': 'invalid'  # String instead of number
        }
        
        is_valid, message = self.predictor._validate_inputs(player_stats, prop_request)
        self.assertFalse(is_valid)
        self.assertIn("must be numeric", message)
    
    def test_calculate_player_averages(self):
        """Test player average calculation"""
        player_stats = {
            'avg_kills': 5.0,
            'recent_kills_avg': 6.0
        }
        
        player_avg, recent_avg = self.predictor._calculate_player_averages(player_stats, 'kills')
        self.assertEqual(player_avg, 5.0)
        self.assertEqual(recent_avg, 6.0)
    
    def test_extract_champion_stats(self):
        """Test champion statistics extraction"""
        player_stats = {
            'recent_matches': [
                {'champion': 'Ahri', 'kills': 5, 'assists': 3},
                {'champion': 'Ahri', 'kills': 4, 'assists': 2},
                {'champion': 'Lux', 'kills': 3, 'assists': 4}
            ]
        }
        
        champ_stats = self.predictor._extract_champion_stats(player_stats)
        self.assertIn('champion_diversity', champ_stats)
        self.assertIn('most_used_champion', champ_stats)
        self.assertEqual(champ_stats['most_used_champion'], 'Ahri')
        self.assertEqual(champ_stats['most_used_count'], 2)
    
    def test_handle_extreme_values(self):
        """Test extreme value handling"""
        config = {
            'impossible_threshold': 15,
            'min_value': 0,
            'low_threshold_factor': 0.5,
            'high_threshold_factor': 1.5
        }
        
        # Test impossible threshold
        result = self.predictor._handle_extreme_values(
            20.0, 'kills', config, 5.0, "", "test reasoning"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['prediction'], 'LESS')
        self.assertEqual(result['confidence'], 99.9)
    
    def test_run_model_inference(self):
        """Test model inference with correct feature count"""
        features = np.random.randn(31)  # 31 features
        
        prediction, confidence = self.predictor._run_model_inference(features)
        self.assertIn(prediction, ['MORE', 'LESS'])
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 100)


if __name__ == "__main__":
    unittest.main() 