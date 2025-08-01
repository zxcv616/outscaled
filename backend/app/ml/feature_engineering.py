import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import os
import pickle

# Set global random seed for deterministic behavior
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturePipeline:
    """Pipeline class to enforce proper scaler usage and provide save/load functionality"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.scaler_path = "backend/app/ml/models/feature_scaler.pkl"
    
    def fit(self, training_data: List[Dict]) -> None:
        """Fit the scaler on training data"""
        try:
            # Extract features from training data
            feature_vectors = []
            for data_point in training_data:
                player_stats = data_point.get('player_stats', {})
                prop_request = data_point.get('prop_request', {})
                features = self.feature_engineer.engineer_features(player_stats, prop_request)
                feature_vectors.append(features)
            
            if feature_vectors:
                X = np.array(feature_vectors)
                self.scaler.fit(X)
                self.is_fitted = True
                logger.info(f"Feature scaler fitted on {len(feature_vectors)} samples")
            else:
                logger.warning("No training data provided for scaler fitting")
                
        except Exception as e:
            logger.error(f"Error fitting feature scaler: {e}")
    
    def transform(self, player_stats: Dict, prop_request: Dict) -> np.ndarray:
        """Transform features using fitted scaler"""
        try:
            features = self.feature_engineer.engineer_features(player_stats, prop_request)
            
            if self.is_fitted:
                return self.scaler.transform(features.reshape(1, -1)).flatten()
            else:
                logger.debug("Scaler not fitted, returning unscaled features")
                return features
                
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            return self.feature_engineer._create_minimal_features(prop_request)
    
    def save_scaler(self) -> None:
        """Save the fitted scaler to disk"""
        try:
            if self.is_fitted:
                os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
                logger.info(f"Feature scaler saved to {self.scaler_path}")
            else:
                logger.warning("Cannot save unfitted scaler")
        except Exception as e:
            logger.error(f"Error saving scaler: {e}")
    
    def load_scaler(self) -> bool:
        """Load the fitted scaler from disk"""
        try:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_fitted = True
                logger.info(f"Feature scaler loaded from {self.scaler_path}")
                return True
            else:
                logger.warning(f"Scaler file not found at {self.scaler_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            return False
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the feature engineer"""
        return self.feature_engineer.get_feature_names()


class FeatureEngineer:
    def __init__(self):
        # Centralized constants for easy tuning
        self.STRONG_TEAMS = {
            'T1', 'Gen.G', 'JD Gaming', 'Bilibili Gaming', 'G2 Esports', 
            'Fnatic', 'Team Liquid', 'Cloud9', '100 Thieves', 'TSM',
            'DRX', 'DWG KIA', 'KT Rolster', 'Hanwha Life Esports'
        }
        
        self.TOURNAMENT_TIERS = {
            'tier1': {'Worlds', 'MSI'},
            'tier2': {'LCS', 'LEC', 'LCK', 'LPL'},
            'tier3': {'LCS Academy', 'LEC Academy', 'LCK Academy', 'LPL Academy'},
            'tier4': {'Regional Leagues', 'Amateur Leagues'}
        }
        
        self.POSITION_FACTORS = {
            'top': 1.0,
            'jungle': 0.9,
            'mid': 1.1,
            'adc': 1.0,
            'support': 0.8
        }
        
        # Centralized pressure weights for tournament tiers
        self.PRESSURE_WEIGHTS = {
            'worlds': 1.0,      # Highest pressure
            'msi': 0.9,         # Very high pressure
            'lcs': 0.7,         # High pressure
            'lec': 0.7,         # High pressure
            'lck': 0.8,         # Very high pressure
            'lpl': 0.8,         # Very high pressure
            'playoffs': 0.9,    # High pressure
            'finals': 1.0,      # Highest pressure
            'semifinals': 0.8,  # High pressure
            'quarterfinals': 0.7 # Medium-high pressure
        }
        
        # Feature names list (41 features total: 14 base + 6 long-term + 17 derived + 4 deviation)
        self.feature_names = [
            # Base features (14) - Keep existing recent stats
            'avg_kills', 'avg_assists', 'avg_cs', 'avg_deaths', 'avg_gold', 
            'avg_damage', 'avg_vision', 'recent_kills_avg', 'recent_assists_avg', 
            'recent_cs_avg', 'win_rate', 'avg_kda', 'avg_gpm', 'avg_kp_percent',
            
            # Long-term averages (6) - NEW: Full dataset averages
            'longterm_kills_avg', 'longterm_assists_avg', 'longterm_cs_avg',
            'longterm_kda', 'longterm_gpm', 'longterm_kp_percent',
            
            # Derived features (17)
            'consistency_score', 'recent_form_trend', 'data_source_quality', 
            'maps_played', 'opponent_strength', 'tournament_tier', 'position_factor',
            'champion_pool_size', 'team_synergy', 'meta_adaptation', 'pressure_handling',
            'late_game_performance', 'early_game_impact', 'mid_game_transition',
            'objective_control', 'champion_performance_variance', 'role_specific_performance',
            
            # Recent form features (3)
            'recent_vs_season_ratio', 'recent_win_rate', 'recent_volatility',
            
            # Deviation & trend features (4) - NEW: Form analysis
            'form_deviation_ratio', 'form_z_score', 'form_trend', 'form_confidence'
        ]
        
        # Feature families for grouped analysis and ablation tests
        self.feature_families = {
            'base': ['avg_kills', 'avg_assists', 'avg_cs', 'avg_deaths', 'avg_gold', 
                    'avg_damage', 'avg_vision', 'recent_kills_avg', 'recent_assists_avg', 
                    'recent_cs_avg', 'win_rate', 'avg_kda', 'avg_gpm', 'avg_kp_percent'],
            'longterm': ['longterm_kills_avg', 'longterm_assists_avg', 'longterm_cs_avg',
                        'longterm_kda', 'longterm_gpm', 'longterm_kp_percent'],
            'derived': ['consistency_score', 'recent_form_trend', 'data_source_quality', 
                       'maps_played', 'opponent_strength', 'tournament_tier', 'position_factor',
                       'champion_pool_size', 'team_synergy', 'meta_adaptation', 'pressure_handling',
                       'late_game_performance', 'early_game_impact', 'mid_game_transition',
                       'objective_control', 'champion_performance_variance', 'role_specific_performance'],
            'recent_form': ['recent_vs_season_ratio', 'recent_win_rate', 'recent_volatility'],
            'deviation': ['form_deviation_ratio', 'form_z_score', 'form_trend', 'form_confidence']
        }
        
        # Feature version for reproducibility
        self.feature_version = "v2.0"
    
    def engineer_features(self, player_stats: Dict, prop_request: Dict, 
                         include_prop_value: bool = False, debug: bool = False,
                         ablation_flags: Optional[Dict[str, bool]] = None) -> np.ndarray:
        """Engineer features from player statistics and prop request with improved map-range support"""
        try:
            player_name = player_stats.get('player_name', 'unknown')
            
            # IMPROVED: Input validation for core stats
            core_stats = ['avg_kills', 'avg_assists', 'avg_cs', 'win_rate']
            missing_core_stats = [stat for stat in core_stats if player_stats.get(stat) is None]
            
            if len(missing_core_stats) > 2:  # If more than 2 core stats are missing
                logger.warning(f"Player {player_name}: Too many missing core stats: {missing_core_stats}. Using fallback features.")
                return self._create_minimal_features(prop_request)
            
            # Extract base features (14 features)
            base_features = self._extract_base_features(player_stats, prop_request)
            
            # Extract derived features (17 features)
            derived_features = self._extract_derived_features(player_stats, prop_request)
            
            # Combine features
            features = base_features + derived_features
            
            # IMPROVED: Dynamic feature validation using feature_names length
            expected_features = len(self.feature_names)
            if len(features) != expected_features:
                logger.debug(f"Player {player_name}: Expected {expected_features} features, got {len(features)}. Padding or truncating.")
                if len(features) < expected_features:
                    features.extend([0.0] * (expected_features - len(features)))
                else:
                    features = features[:expected_features]
            
            # Apply ablation flags if provided
            if ablation_flags:
                features = self._apply_ablation_flags(features, ablation_flags)
            
            # Optional: Add prop_value for inference-only use cases
            if include_prop_value:
                prop_value = prop_request.get('prop_value', 0.0)
                features.append(prop_value)
            
            feature_array = np.array(features, dtype=np.float32)
            logger.debug(f"Player {player_name}: Engineered {len(feature_array)} features")
            
            # Return feature dictionary for debugging if requested
            if debug:
                feature_dict = dict(zip(self.feature_names, feature_array.tolist()))
                return feature_array, feature_dict
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Player {player_stats.get('player_name', 'unknown')}: Error engineering features: {e}")
            return self._create_minimal_features(prop_request)
    
    def _apply_ablation_flags(self, features: List[float], ablation_flags: Dict[str, bool]) -> List[float]:
        """Apply ablation flags to disable specific feature families"""
        try:
            feature_dict = dict(zip(self.feature_names, features))
            
            # Disable base features if flagged
            if not ablation_flags.get('base_features', True):
                for feature in self.feature_families['base']:
                    if feature in feature_dict:
                        feature_dict[feature] = 0.0
            
            # Disable derived features if flagged
            if not ablation_flags.get('derived_features', True):
                for feature in self.feature_families['derived']:
                    if feature in feature_dict:
                        feature_dict[feature] = 0.0
            
            # Disable recent form features if flagged
            if not ablation_flags.get('recent_form_features', True):
                for feature in self.feature_families['recent_form']:
                    if feature in feature_dict:
                        feature_dict[feature] = 0.0
            
            # Disable deviation features if flagged
            if not ablation_flags.get('deviation_features', True):
                for feature in self.feature_families['deviation']:
                    if feature in feature_dict:
                        feature_dict[feature] = 0.0
            
            return list(feature_dict.values())
        except Exception as e:
            logger.error(f"Error applying ablation flags: {e}")
            return features
    
    def _extract_base_features(self, player_stats: Dict, prop_request: Dict) -> List[float]:
        """Extract base statistical features including long-term averages"""
        try:
            player_name = player_stats.get('player_name', 'unknown')
            
            # Base statistics (14 features) - Keep existing recent stats
            base_features = [
                player_stats.get("avg_kills", 0.0),
                player_stats.get("avg_assists", 0.0),
                player_stats.get("avg_cs", 0.0),
                player_stats.get("avg_deaths", 0.0),
                player_stats.get("avg_gold", 0.0),
                player_stats.get("avg_damage", 0.0),
                player_stats.get("avg_vision", 0.0),
                player_stats.get("recent_kills_avg", 0.0),
                player_stats.get("recent_assists_avg", 0.0),
                player_stats.get("recent_cs_avg", 0.0),
                player_stats.get("win_rate", 0.0),
                player_stats.get("avg_kda", 0.0),
                player_stats.get("avg_gpm", 0.0),
                player_stats.get("avg_kp_percent", 0.0)
            ]
            
            # Long-term averages (6 features) - NEW: Full dataset averages
            longterm_features = [
                player_stats.get("longterm_kills_avg", player_stats.get("avg_kills", 0.0)),
                player_stats.get("longterm_assists_avg", player_stats.get("avg_assists", 0.0)),
                player_stats.get("longterm_cs_avg", player_stats.get("avg_cs", 0.0)),
                player_stats.get("longterm_kda", player_stats.get("avg_kda", 0.0)),
                player_stats.get("longterm_gpm", player_stats.get("avg_gpm", 0.0)),
                player_stats.get("longterm_kp_percent", player_stats.get("avg_kp_percent", 0.0))
            ]
            
            # FIX: Calculate recent averages if they're null
            recent_matches = player_stats.get("recent_matches", [])
            if recent_matches:
                # Calculate recent averages from actual match data
                recent_kills = [m.get("kills", 0) for m in recent_matches[:5]]
                recent_assists = [m.get("assists", 0) for m in recent_matches[:5]]
                recent_cs = [m.get("cs", 0) for m in recent_matches[:5]]
                
                if recent_kills and base_features[7] == 0.0:  # recent_kills_avg is null/0
                    base_features[7] = sum(recent_kills) / len(recent_kills)
                if recent_assists and base_features[8] == 0.0:  # recent_assists_avg is null/0
                    base_features[8] = sum(recent_assists) / len(recent_assists)
                if recent_cs and base_features[9] == 0.0:  # recent_cs_avg is null/0
                    base_features[9] = sum(recent_cs) / len(recent_cs)
            
            # Combine base and long-term features
            features = base_features + longterm_features
            
            return features
            
        except Exception as e:
            logger.error(f"Player {player_stats.get('player_name', 'unknown')}: Error extracting base features: {e}")
            return [0.0] * 20  # 14 base + 6 long-term features
    
    def _extract_derived_features(self, player_stats: Dict, prop_request: Dict) -> List[float]:
        """Extract derived/computed features including deviation analysis"""
        try:
            player_name = player_stats.get('player_name', 'unknown')
            
            # Get map range from prop request
            map_range = prop_request.get('map_range', [1])
            maps_played = len(map_range)
            
            # Calculate recent form features
            recent_matches = player_stats.get("recent_matches", [])
            recent_form_features = self._calculate_recent_form_features(recent_matches, player_stats)
            
            # Calculate deviation features - NEW: Form analysis
            deviation_features = self._calculate_deviation_features(player_stats, prop_request)
            
            # Derived features (17 features)
            derived_features = [
                self._calculate_consistency_score(player_stats),
                self._calculate_recent_form_trend(player_stats),
                self._calculate_dynamic_data_quality(player_stats),
                maps_played,  # Number of maps in prediction range
                self._calculate_opponent_strength(prop_request),
                self._calculate_tournament_tier(prop_request),
                self._calculate_position_factor(player_stats),
                self._calculate_champion_pool_size(player_stats),
                self._calculate_team_synergy(player_stats),
                self._calculate_meta_adaptation(player_stats),
                self._calculate_pressure_handling(player_stats, prop_request),
                self._calculate_late_game_performance(player_stats),
                self._calculate_early_game_impact(player_stats),
                self._calculate_mid_game_transition(player_stats),
                self._calculate_objective_control(player_stats),
                self._calculate_champion_performance_variance(player_stats),
                self._calculate_role_specific_performance(player_stats)
            ]
            
            # Add recent form features
            derived_features.extend(recent_form_features)
            
            # Add deviation features - NEW
            derived_features.extend(deviation_features)
            
            return derived_features
            
        except Exception as e:
            logger.error(f"Player {player_stats.get('player_name', 'unknown')}: Error extracting derived features: {e}")
            return [0.0] * 24  # 17 base + 3 recent form + 4 deviation features
    
    def _calculate_recent_form_features(self, recent_matches: List[Dict], player_stats: Dict) -> List[float]:
        """Calculate recent form features to better capture recent performance"""
        try:
            player_name = player_stats.get('player_name', 'unknown')
            
            if not recent_matches:
                return [0.5, 0.5, 0.5]  # Neutral values if no recent matches
            
            # Feature 1: Recent vs Season Performance Ratio
            recent_kills = [m.get("kills", 0) for m in recent_matches[:5]]
            recent_avg = sum(recent_kills) / len(recent_kills) if recent_kills else 0
            season_avg = player_stats.get("avg_kills", 1.0)  # Avoid division by zero
            
            if season_avg > 0:
                performance_ratio = recent_avg / season_avg
                performance_ratio = min(max(performance_ratio, 0.1), 3.0)  # Cap between 0.1 and 3.0
            else:
                performance_ratio = 1.0
            
            # Feature 2: Recent Win Rate (last 5 matches)
            recent_wins = sum(1 for m in recent_matches[:5] if m.get("win", False))
            recent_win_rate = recent_wins / min(5, len(recent_matches))
            
            # Feature 3: Recent Performance Volatility
            if len(recent_kills) >= 2:
                volatility = np.std(recent_kills) / (np.mean(recent_kills) + 0.1)  # Add small constant
                volatility = min(max(volatility, 0.0), 2.0)  # Cap between 0 and 2
            else:
                volatility = 0.5  # Neutral if not enough data
            
            return [performance_ratio, recent_win_rate, volatility]
            
        except Exception as e:
            logger.error(f"Player {player_stats.get('player_name', 'unknown')}: Error calculating recent form features: {e}")
            return [0.5, 0.5, 0.5]
    
    def _normalize_score(self, value: float, expected_max: float) -> float:
        """Helper function to normalize scores to [0, 1] range"""
        return min(max(value / expected_max, 0.0), 1.0)
    
    def _calculate_consistency_score(self, player_stats: Dict) -> float:
        """Calculate consistency score based on standard deviation of recent performance"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches or len(recent_matches) < 3:
            return 0.5  # Default moderate consistency
        
        try:
            kills = [m.get('kills', 0) for m in recent_matches if m is not None]
            if not kills:
                return 0.5
            
            kills_std = np.std(kills)
            kills_mean = np.mean(kills)
            
            # Lower coefficient of variation = higher consistency
            consistency = 1.0 / (1.0 + (kills_std / max(kills_mean, 1.0)))
            return min(max(consistency, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating consistency score: {e}")
            return 0.5
    
    def _calculate_recent_form_trend(self, player_stats: Dict) -> float:
        """Calculate recent form trend using linear regression (IMPROVED)"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches or len(recent_matches) < 5:
            return 0.0
        
        try:
            # Extract kills data for trend analysis
            kills_data = [m.get('kills', 0) for m in recent_matches if m is not None]
            if len(kills_data) < 3:
                return 0.0
            
            # Use linear regression for more robust trend calculation
            y = np.array(kills_data).reshape(-1, 1)
            x = np.arange(len(y)).reshape(-1, 1)
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(x, y)
            slope = reg.coef_[0][0]
            
            # Normalize slope to [-1, 1] range based on data variance
            if len(kills_data) > 1:
                data_std = np.std(kills_data)
                if data_std > 0:
                    normalized_slope = slope / data_std
                    return np.clip(normalized_slope, -1.0, 1.0)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating form trend: {e}")
            return 0.0
    
    def _calculate_data_source_quality(self, player_stats: Dict) -> float:
        """Calculate data source quality score"""
        data_source = player_stats.get('data_source', 'unknown')
        
        quality_scores = {
            'oracles_elixir': 1.0,
            'riot_api': 0.8,
            'gol_gg': 0.7,
            'unknown': 0.5
        }
        
        return quality_scores.get(data_source, 0.5)
    
    def _calculate_dynamic_data_quality(self, player_stats: Dict) -> float:
        """Calculate dynamic data quality score based on recent matches"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches:
            return 0.5
        
        try:
            # Check for common issues like missing kills, assists, or vision
            missing_stats = sum(1 for m in recent_matches if m is not None and (
                m.get('kills') is None or m.get('assists') is None or m.get('vision') is None
            ))
            
            # Score based on missing data
            if missing_stats > 0:
                return 0.3 + (0.7 * (1.0 - (missing_stats / len(recent_matches))))
            return 1.0
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating dynamic data quality: {e}")
            return 0.5
    
    def _calculate_opponent_strength(self, prop_request: Dict) -> float:
        """Calculate opponent strength based on team name"""
        opponent = prop_request.get('opponent', '').strip()
        if not opponent:
            return 0.5
        
        # Check if opponent is in strong teams list
        if opponent in self.STRONG_TEAMS:
            return 0.8
        elif any(strong_team in opponent for strong_team in self.STRONG_TEAMS):
            return 0.7
        else:
            return 0.5
    
    def _calculate_tournament_tier(self, prop_request: Dict) -> float:
        """Calculate tournament tier importance"""
        tournament = prop_request.get('tournament', '').strip()
        if not tournament:
            return 0.5
        
        for tier, tournaments in self.TOURNAMENT_TIERS.items():
            if tournament in tournaments:
                tier_scores = {'tier1': 1.0, 'tier2': 0.8, 'tier3': 0.6, 'tier4': 0.4}
                return tier_scores.get(tier, 0.5)
        
        return 0.5
    
    def _calculate_position_factor(self, player_stats: Dict) -> float:
        """Calculate position-specific factor with role-appropriate features"""
        # IMPROVED: Safe position extraction with fallback
        position = str(player_stats.get('position', 'mid')).lower()
        
        return self.POSITION_FACTORS.get(position, 1.0)
    
    def _calculate_role_specific_performance(self, player_stats: Dict) -> float:
        """Calculate role-specific performance metrics"""
        position = str(player_stats.get('position', 'mid')).lower()
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches:
            return 0.5
        
        try:
            if position == 'adc':
                # ADC: Focus on CS, damage, and KDA
                cs_scores = [m.get('cs', 0) for m in recent_matches if m is not None]
                damage_scores = [m.get('damage', 0) for m in recent_matches if m is not None]
                
                avg_cs = np.mean(cs_scores) if cs_scores else 0
                avg_damage = np.mean(damage_scores) if damage_scores else 0
                
                # ADC performance: CS and damage focused
                cs_score = self._normalize_score(avg_cs, 300.0)  # Normalize to 300 CS
                damage_score = self._normalize_score(avg_damage, 20000.0)  # Normalize to 20k damage
                
                return (cs_score * 0.6) + (damage_score * 0.4)
                
            elif position == 'support':
                # Support: Focus on assists, vision, and low deaths
                assist_scores = [m.get('assists', 0) for m in recent_matches if m is not None]
                vision_scores = [m.get('vision', 0) for m in recent_matches if m is not None]
                death_scores = [m.get('deaths', 0) for m in recent_matches if m is not None]
                
                avg_assists = np.mean(assist_scores) if assist_scores else 0
                avg_vision = np.mean(vision_scores) if vision_scores else 0
                avg_deaths = np.mean(death_scores) if death_scores else 0
                
                # Support performance: Assists and vision focused, low deaths
                assist_score = self._normalize_score(avg_assists, 15.0)  # Normalize to 15 assists
                vision_score = self._normalize_score(avg_vision, 50.0)   # Normalize to 50 vision
                death_score = max(0, 1.0 - (avg_deaths / 5.0))  # Lower deaths = better
                
                return (assist_score * 0.5) + (vision_score * 0.3) + (death_score * 0.2)
                
            elif position == 'mid':
                # Mid: Focus on KDA and damage
                kda_scores = [m.get('kda', 0) for m in recent_matches if m is not None]
                damage_scores = [m.get('damage', 0) for m in recent_matches if m is not None]
                
                avg_kda = np.mean(kda_scores) if kda_scores else 0
                avg_damage = np.mean(damage_scores) if damage_scores else 0
                
                # Mid performance: KDA and damage focused
                kda_score = self._normalize_score(avg_kda, 5.0)  # Normalize to 5.0 KDA
                damage_score = self._normalize_score(avg_damage, 25000.0)  # Normalize to 25k damage
                
                return (kda_score * 0.6) + (damage_score * 0.4)
                
            elif position == 'jungle':
                # Jungle: Focus on assists and vision
                assist_scores = [m.get('assists', 0) for m in recent_matches if m is not None]
                vision_scores = [m.get('vision', 0) for m in recent_matches if m is not None]
                
                avg_assists = np.mean(assist_scores) if assist_scores else 0
                avg_vision = np.mean(vision_scores) if vision_scores else 0
                
                # Jungle performance: Assists and vision focused
                assist_score = self._normalize_score(avg_assists, 12.0)  # Normalize to 12 assists
                vision_score = self._normalize_score(avg_vision, 40.0)   # Normalize to 40 vision
                
                return (assist_score * 0.7) + (vision_score * 0.3)
                
            else:  # top or default
                # Top: Focus on CS and KDA
                cs_scores = [m.get('cs', 0) for m in recent_matches if m is not None]
                kda_scores = [m.get('kda', 0) for m in recent_matches if m is not None]
                
                avg_cs = np.mean(cs_scores) if cs_scores else 0
                avg_kda = np.mean(kda_scores) if kda_scores else 0
                
                # Top performance: CS and KDA focused
                cs_score = self._normalize_score(avg_cs, 250.0)  # Normalize to 250 CS
                kda_score = self._normalize_score(avg_kda, 4.0)  # Normalize to 4.0 KDA
                
                return (cs_score * 0.6) + (kda_score * 0.4)
                
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating role-specific performance: {e}")
            return 0.5
    
    def _calculate_champion_pool_size(self, player_stats: Dict) -> float:
        """Calculate champion pool diversity"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches:
            return 0.5
        
        try:
            champions = [m.get('champion', '') for m in recent_matches if m is not None]
            champions = [c for c in champions if c]  # Remove empty strings
            
            if not champions:
                return 0.5
            
            unique_champions = len(set(champions))
            total_matches = len(champions)
            
            # Normalize to [0, 1] range (0 = same champ every game, 1 = different champ every game)
            diversity = unique_champions / max(total_matches, 1)
            return min(diversity, 1.0)
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating champion pool size: {e}")
            return 0.5
    
    def _calculate_team_synergy(self, player_stats: Dict) -> float:
        """Calculate team synergy based on win rate and recent performance"""
        win_rate = player_stats.get('win_rate', 0.5)
        recent_matches = player_stats.get('recent_matches', [])
        
        if not recent_matches:
            return win_rate
        
        try:
            # Calculate recent win rate
            recent_wins = sum(1 for m in recent_matches if m is not None and m.get('win', False))
            recent_win_rate = recent_wins / len(recent_matches) if recent_matches else 0.5
            
            # Combine overall and recent win rates
            synergy = (win_rate * 0.6) + (recent_win_rate * 0.4)
            return min(max(synergy, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"Player {player_stats.get('player_name', 'unknown')}: Error calculating team synergy: {e}")
            return win_rate
    
    def _calculate_meta_adaptation(self, player_stats: Dict) -> float:
        """Calculate meta adaptation score based on champion diversity and performance"""
        champion_diversity = self._calculate_champion_pool_size(player_stats)
        consistency = self._calculate_consistency_score(player_stats)
        
        # Meta adaptation is higher for players with good consistency and champion diversity
        adaptation = (champion_diversity * 0.6) + (consistency * 0.4)
        return min(max(adaptation, 0.0), 1.0)
    
    def _calculate_pressure_handling(self, player_stats: Dict, prop_request: Dict) -> float:
        """Calculate pressure handling based on performance in high-stakes situations with context"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches:
            return 0.5
        
        try:
            # Get tournament context for pressure assessment
            tournament = prop_request.get('tournament', '').lower()
            opponent = prop_request.get('opponent', '').lower()
            
            # Calculate base pressure from tournament using centralized weights
            base_pressure = 0.5  # Default medium pressure
            for pressure_key, weight in self.PRESSURE_WEIGHTS.items():
                if pressure_key in tournament:
                    base_pressure = weight
                    break
            
            # Adjust pressure based on opponent strength
            if opponent:
                if any(team in opponent for team in self.STRONG_TEAMS):
                    base_pressure = min(1.0, base_pressure + 0.2)  # Increase pressure vs strong teams
            
            # Calculate performance variance under pressure
            performances = []
            for m in recent_matches:
                if m is not None:
                    # Weight performance by pressure level
                    performance = (m.get('kills', 0) + m.get('assists', 0)) * base_pressure
                    performances.append(performance)
            
            if not performances:
                return 0.5
            
            # Lower variance = better pressure handling
            variance = np.var(performances)
            mean_performance = np.mean(performances)
            
            if mean_performance > 0:
                pressure_score = 1.0 / (1.0 + (variance / mean_performance))
                # Adjust score based on pressure level (higher pressure = more weight)
                adjusted_score = pressure_score * (0.5 + 0.5 * base_pressure)
                return min(max(adjusted_score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating pressure handling: {e}")
            return 0.5
    
    def _calculate_late_game_performance(self, player_stats: Dict) -> float:
        """Calculate late game performance (simplified proxy)"""
        # Use KDA as a proxy for late game performance
        avg_kda = player_stats.get('avg_kda', 2.0)
        return self._normalize_score(avg_kda, 5.0)  # Normalize to [0, 1]
    
    def _calculate_early_game_impact(self, player_stats: Dict) -> float:
        """Calculate early game impact (simplified proxy)"""
        # Use CS as a proxy for early game impact
        avg_cs = player_stats.get('avg_cs', 200.0)
        return self._normalize_score(avg_cs, 300.0)  # Normalize to [0, 1]
    
    def _calculate_mid_game_transition(self, player_stats: Dict) -> float:
        """Calculate mid game transition effectiveness"""
        # Use assists as a proxy for mid game transition
        avg_assists = player_stats.get('avg_assists', 5.0)
        return self._normalize_score(avg_assists, 10.0)  # Normalize to [0, 1]
    
    def _calculate_objective_control(self, player_stats: Dict) -> float:
        """Calculate objective control (simplified proxy)"""
        # Use vision score as a proxy for objective control
        avg_vision = player_stats.get('avg_vision', 25.0)
        return self._normalize_score(avg_vision, 50.0)  # Normalize to [0, 1]
    
    def _calculate_champion_performance_variance(self, player_stats: Dict) -> float:
        """Calculate champion performance variance with optional weighting"""
        recent_matches = player_stats.get('recent_matches', [])
        player_name = player_stats.get('player_name', 'unknown')
        
        if not recent_matches:
            return 0.5
        
        try:
            # Extract champion names and performance metrics
            champion_data = []
            for m in recent_matches:
                if m is not None:
                    champion_name = m.get('champion', '')
                    kills = m.get('kills', 0)
                    assists = m.get('assists', 0)
                    vision = m.get('vision', 0)
                    champion_data.append({
                        'champion': champion_name,
                        'kills': kills,
                        'assists': assists,
                        'vision': vision
                    })
            
            if not champion_data:
                return 0.5
            
            # Group by champion and calculate variance for each
            champion_variances = {}
            for item in champion_data:
                champion_name = item['champion']
                if champion_name not in champion_variances:
                    champion_variances[champion_name] = []
                champion_variances[champion_name].append(item['kills'] + item['assists'] + item['vision'])
            
            # Calculate variance for each champion
            champion_variance_scores = {}
            for champion, performances in champion_variances.items():
                if performances:
                    variance = np.var(performances)
                    mean_performance = np.mean(performances)
                    
                    if mean_performance > 0:
                        # Lower variance = better performance consistency
                        variance_score = 1.0 / (1.0 + (variance / mean_performance))
                        champion_variance_scores[champion] = min(max(variance_score, 0.0), 1.0)
                    else:
                        champion_variance_scores[champion] = 0.5 # Neutral if no performance data
                else:
                    champion_variance_scores[champion] = 0.5 # Neutral if no performance data
            
            # Combine champion variances into a single score
            # This is a simplified approach; a more sophisticated method might involve
            # weighting by champion popularity or recent play rate.
            # For now, we'll just take the average of the champion variances.
            if champion_variance_scores:
                average_variance_score = np.mean(list(champion_variance_scores.values()))
                return min(max(average_variance_score, 0.0), 1.0)
            return 0.5
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating champion performance variance: {e}")
            return 0.5
    
    def _calculate_deviation_features(self, player_stats: Dict, prop_request: Dict) -> List[float]:
        """Calculate deviation features for form analysis"""
        try:
            player_name = player_stats.get('player_name', 'unknown')
            prop_type = prop_request.get('prop_type', 'kills')
            prop_value = prop_request.get('prop_value', 0.0)
            recent_matches = player_stats.get('recent_matches', [])
            
            if not recent_matches:
                return [0.5, 0.5, 0.5, 0.5]  # Neutral values if no recent matches
            
            # Get recent values for the prop type
            if prop_type == "kills":
                recent_values = [m.get("kills", 0) for m in recent_matches[:10]]
                recent_avg = player_stats.get("recent_kills_avg", 0.0)
                longterm_avg = player_stats.get("longterm_kills_avg", player_stats.get("avg_kills", 0.0))
            elif prop_type == "assists":
                recent_values = [m.get("assists", 0) for m in recent_matches[:10]]
                recent_avg = player_stats.get("recent_assists_avg", 0.0)
                longterm_avg = player_stats.get("longterm_assists_avg", player_stats.get("avg_assists", 0.0))
            elif prop_type == "cs":
                recent_values = [m.get("cs", 0) for m in recent_matches[:10]]
                recent_avg = player_stats.get("recent_cs_avg", 0.0)
                longterm_avg = player_stats.get("longterm_cs_avg", player_stats.get("avg_cs", 0.0))
            else:
                recent_values = [m.get("kills", 0) for m in recent_matches[:10]]
                recent_avg = player_stats.get("recent_kills_avg", 0.0)
                longterm_avg = player_stats.get("longterm_kills_avg", player_stats.get("avg_kills", 0.0))
            
            # Calculate recent average if not available
            if recent_avg == 0.0 and recent_values:
                recent_avg = sum(recent_values) / len(recent_values)
            
            deviation_features = []
            
            # 1. Form Deviation Ratio: recent_avg / longterm_avg
            if longterm_avg > 0:
                deviation_ratio = recent_avg / longterm_avg
                # Cap between 0.1 and 3.0 to prevent extreme values
                deviation_ratio = min(max(deviation_ratio, 0.1), 3.0)
            else:
                deviation_ratio = 1.0  # Neutral if no long-term data
            deviation_features.append(deviation_ratio)
            
            # 2. Form Z-Score: (prop_value - recent_avg) / recent_std
            if len(recent_values) >= 2:
                recent_std = np.std(recent_values)
                if recent_std > 0:
                    z_score = (prop_value - recent_avg) / recent_std
                    # Cap z-score to prevent extreme values
                    z_score = min(max(z_score, -3.0), 3.0)
                else:
                    z_score = 0.0  # No variance
            else:
                z_score = 0.0  # Not enough data
            deviation_features.append(z_score)
            
            # 3. Form Trend: Linear regression slope of recent performance
            if len(recent_values) >= 3:
                # Use last 5 matches for trend calculation
                trend_values = recent_values[:5]
                x = np.arange(len(trend_values)).reshape(-1, 1)
                y = np.array(trend_values).reshape(-1, 1)
                
                try:
                    reg = LinearRegression()
                    reg.fit(x, y)
                    slope = reg.coef_[0][0]
                    
                    # Normalize slope based on data variance
                    if len(trend_values) > 1:
                        data_std = np.std(trend_values)
                        if data_std > 0:
                            normalized_slope = slope / data_std
                            # Cap between -1 and 1
                            normalized_slope = min(max(normalized_slope, -1.0), 1.0)
                        else:
                            normalized_slope = 0.0
                    else:
                        normalized_slope = 0.0
                except:
                    normalized_slope = 0.0
            else:
                normalized_slope = 0.0  # Not enough data
            deviation_features.append(normalized_slope)
            
            # 4. Form Confidence: Based on sample size and data quality
            if len(recent_values) >= 5:
                confidence = 1.0  # High confidence with 5+ matches
            elif len(recent_values) >= 3:
                confidence = 0.7  # Moderate confidence with 3-4 matches
            else:
                confidence = 0.3  # Low confidence with <3 matches
            
            # Adjust confidence based on data quality (variance)
            if len(recent_values) >= 2:
                data_cv = np.std(recent_values) / np.mean(recent_values) if np.mean(recent_values) > 0 else 0
                if data_cv > 0.5:  # High variance
                    confidence *= 0.8  # Reduce confidence for high variance
                elif data_cv < 0.2:  # Low variance
                    confidence *= 1.1  # Increase confidence for low variance
            
            deviation_features.append(confidence)
            
            return deviation_features
            
        except Exception as e:
            logger.warning(f"Player {player_name}: Error calculating deviation features: {e}")
            return [0.5, 0.5, 0.5, 0.5]  # Neutral values on error
    
    def _create_minimal_features(self, prop_request: Dict) -> np.ndarray:
        """Create minimal features for fallback cases"""
        try:
            # IMPROVED: Use feature_names length instead of hardcoded 34
            features = [0.0] * len(self.feature_names)
            
            # Set basic values
            features[16] = len(prop_request.get('map_range', [1]))  # maps_played
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating minimal features: {e}")
            return np.zeros(len(self.feature_names), dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_families(self) -> Dict[str, List[str]]:
        """Get feature families for grouped analysis"""
        return self.feature_families.copy()
    
    def get_feature_version(self) -> str:
        """Get feature version for reproducibility"""
        return self.feature_version
    
    def get_feature_description(self, feature_name: str) -> str:
        """Get description of a feature"""
        descriptions = {
            'avg_kills': 'Average kills per game',
            'avg_assists': 'Average assists per game',
            'avg_cs': 'Average CS per game',
            'avg_deaths': 'Average deaths per game',
            'avg_gold': 'Average gold per game',
            'avg_damage': 'Average damage per game',
            'avg_vision': 'Average vision score per game',
            'recent_kills_avg': 'Recent kills average (last 5 games)',
            'recent_assists_avg': 'Recent assists average (last 5 games)',
            'recent_cs_avg': 'Recent CS average (last 5 games)',
            'win_rate': 'Overall win rate',
            'avg_kda': 'Average KDA ratio',
            'avg_gpm': 'Average gold per minute',
            'avg_kp_percent': 'Average kill participation percentage',
            'longterm_kills_avg': 'Long-term kills average (full dataset)',
            'longterm_assists_avg': 'Long-term assists average (full dataset)',
            'longterm_cs_avg': 'Long-term CS average (full dataset)',
            'longterm_kda': 'Long-term KDA ratio (full dataset)',
            'longterm_gpm': 'Long-term gold per minute (full dataset)',
            'longterm_kp_percent': 'Long-term kill participation percentage (full dataset)',
            'consistency_score': 'Performance consistency score',
            'recent_form_trend': 'Recent form trend (improving/declining)',
            'data_source_quality': 'Quality score of data source',
            'maps_played': 'Number of maps in the prediction range',
            'opponent_strength': 'Opponent team strength',
            'tournament_tier': 'Tournament tier importance',
            'position_factor': 'Position-specific factor',
            'champion_pool_size': 'Champion pool diversity',
            'team_synergy': 'Team synergy score',
            'meta_adaptation': 'Meta adaptation score',
            'pressure_handling': 'Pressure handling score',
            'late_game_performance': 'Late game performance score',
            'early_game_impact': 'Early game impact score',
            'mid_game_transition': 'Mid game transition score',
            'objective_control': 'Objective control score',
            'champion_performance_variance': 'Champion performance variance score',
            'role_specific_performance': 'Role-specific performance score',
            'recent_vs_season_ratio': 'Recent vs season performance ratio',
            'recent_win_rate': 'Recent win rate (last 5 matches)',
            'recent_volatility': 'Recent performance volatility',
            'form_deviation_ratio': 'Form deviation ratio (recent vs long-term average)',
            'form_z_score': 'Form Z-score (deviation from recent performance mean)',
            'form_trend': 'Form trend (linear regression slope of recent performance)',
            'form_confidence': 'Form confidence score (based on sample size and data quality)'
        }
        return descriptions.get(feature_name, 'Unknown feature') 