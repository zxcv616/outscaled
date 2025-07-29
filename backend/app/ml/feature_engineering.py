import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler
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
                logger.warning("Scaler not fitted, returning unscaled features")
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


# Legacy FeatureEngineer class for backward compatibility
class FeatureEngineer:
    def __init__(self):
        # Extract constants to class variables
        self.STRONG_TEAMS = {
            'T1', 'Gen.G', 'JDG', 'BLG', 'TES', 'WBG', 'LNG', 'EDG', 'RNG', 'OMG',
            'G2', 'FNC', 'MAD', 'VIT', 'KOI', 'TH', 'SK', 'BDS', 'AST', 'GX',
            'C9', 'TL', '100T', 'EG', 'FLY', 'GG', 'IMT', 'DIG', 'CLG', 'TSM'
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
        
        self.feature_names = [
            'avg_kills', 'avg_assists', 'avg_cs', 'avg_deaths', 'avg_gold', 
            'avg_damage', 'avg_vision', 'recent_kills_avg', 'recent_assists_avg', 
            'recent_cs_avg', 'win_rate', 'avg_kda', 'avg_gpm', 'avg_kp_percent',
            'consistency_score', 'recent_form_trend', 'data_source_quality',
            'maps_played', 'opponent_strength', 'tournament_tier',
            'position_factor', 'champion_pool_size', 'team_synergy',
            'meta_adaptation', 'pressure_handling', 'late_game_performance',
            'early_game_impact', 'mid_game_transition', 'objective_control',
            'champion_performance_variance', 'role_specific_performance'
        ]
        
        # Initialize scaler for optional feature scaling
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False
    
    def engineer_features(self, player_stats: Dict, prop_request: Dict) -> np.ndarray:
        """Engineer features from player statistics and prop request with improved map-range support"""
        try:
            # IMPROVED: Input validation for core stats
            core_stats = ['avg_kills', 'avg_assists', 'avg_cs', 'win_rate']
            missing_core_stats = [stat for stat in core_stats if player_stats.get(stat) is None]
            
            if len(missing_core_stats) > 2:  # If more than 2 core stats are missing
                logger.warning(f"Too many missing core stats: {missing_core_stats}. Using fallback features.")
                return self._create_minimal_features(prop_request)
            
            features = []
            
            # Get map range from prop request
            map_range = prop_request.get('map_range', [1])
            maps_played = len(map_range)
            
            # IMPROVED: Calculate normalized averages instead of simple multiplication
            # This avoids scale drift and provides more realistic feature values
            normalization_factor = max(maps_played, 1)  # Avoid division by zero
            
            # Basic player statistics (14 features) - now properly normalized
            features.extend([
                player_stats.get('avg_kills', 0.0) / normalization_factor,  # Normalized average
                player_stats.get('avg_assists', 0.0) / normalization_factor,
                player_stats.get('avg_cs', 0.0) / normalization_factor,
                player_stats.get('avg_deaths', 0.0) / normalization_factor,
                player_stats.get('avg_gold', 0.0) / normalization_factor,
                player_stats.get('avg_damage', 0.0) / normalization_factor,
                player_stats.get('avg_vision', 0.0) / normalization_factor,
                player_stats.get('recent_kills_avg', 0.0) / normalization_factor,
                player_stats.get('recent_assists_avg', 0.0) / normalization_factor,
                player_stats.get('recent_cs_avg', 0.0) / normalization_factor,
                player_stats.get('win_rate', 0.0),  # Win rate doesn't need normalization
                player_stats.get('avg_kda', 0.0),   # KDA doesn't need normalization
                player_stats.get('avg_gpm', 0.0),   # GPM doesn't need normalization
                player_stats.get('avg_kp_percent', 0.0)  # KP% doesn't need normalization
            ])
            
            # Derived features (17 features) - enhanced with new features (prop_value removed for training)
            features.extend([
                self._calculate_consistency_score(player_stats),
                self._calculate_recent_form_trend(player_stats),
                self._calculate_dynamic_data_quality(player_stats),  # IMPROVED: Dynamic quality
                # prop_value removed from training features to prevent data leakage
                maps_played,  # Use maps_played instead of map_number
                self._calculate_opponent_strength(prop_request),
                self._calculate_tournament_tier(prop_request),
                self._calculate_position_factor(player_stats),
                self._calculate_champion_pool_size(player_stats),
                self._calculate_team_synergy(player_stats),
                self._calculate_meta_adaptation(player_stats),
                self._calculate_pressure_handling(player_stats, prop_request),  # IMPROVED: Context-aware
                self._calculate_late_game_performance(player_stats),
                self._calculate_early_game_impact(player_stats),
                self._calculate_mid_game_transition(player_stats),
                self._calculate_objective_control(player_stats),  # FIXED: Added missing feature
                self._calculate_champion_performance_variance(player_stats),  # NEW: Champion variance
                self._calculate_role_specific_performance(player_stats) # NEW: Role-specific performance
            ])
            
            # Ensure exactly 31 features (prop_value removed from training)
            if len(features) != 31:
                logger.warning(f"Expected 31 features, got {len(features)}. Padding or truncating.")
                if len(features) < 31:
                    features.extend([0.0] * (31 - len(features)))
                else:
                    features = features[:31]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return self._create_minimal_features(prop_request)
    
    def fit_scaler(self, training_data: List[np.ndarray]) -> None:
        """Fit the scaler on training data for consistent scaling"""
        try:
            if training_data:
                X = np.array(training_data)
                self.scaler.fit(X)
                self.is_scaler_fitted = True
                logger.info("Feature scaler fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}")
    
    def scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features if scaler is fitted"""
        if self.is_scaler_fitted:
            return self.scaler.transform(features.reshape(1, -1)).flatten()
        return features
    
    def _calculate_consistency_score(self, player_stats: Dict) -> float:
        """Calculate consistency score based on standard deviation of recent performance"""
        recent_matches = player_stats.get('recent_matches', [])
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
            logger.warning(f"Error calculating consistency score: {e}")
            return 0.5
    
    def _calculate_recent_form_trend(self, player_stats: Dict) -> float:
        """Calculate recent form trend (positive = improving, negative = declining)"""
        recent_matches = player_stats.get('recent_matches', [])
        if not recent_matches or len(recent_matches) < 5:
            return 0.0
        
        try:
            # Split into two halves and compare
            first_half = recent_matches[:len(recent_matches)//2]
            second_half = recent_matches[len(recent_matches)//2:]
            
            first_avg = np.mean([m.get('kills', 0) for m in first_half if m is not None]) if first_half else 0.0
            second_avg = np.mean([m.get('kills', 0) for m in second_half if m is not None]) if second_half else 0.0
            
            # Normalize trend to [-1, 1] range
            if first_avg > 0:
                trend = (second_avg - first_avg) / first_avg
                return np.clip(trend, -1.0, 1.0)
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating form trend: {e}")
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
            logger.warning(f"Error calculating dynamic data quality: {e}")
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
        # Try to infer position from player stats or use default
        position = player_stats.get('position', 'mid').lower()  # Default to mid
        
        # Position-specific factors based on role importance for different stats
        position_factors = {
            'top': 1.0,      # Balanced role
            'jungle': 0.9,   # Slightly less KDA focused
            'mid': 1.1,      # High KDA importance
            'adc': 1.0,      # Balanced, but CS focused
            'support': 0.8    # Lower KDA, higher vision/assists
        }
        
        return position_factors.get(position, 1.0)
    
    def _calculate_role_specific_performance(self, player_stats: Dict) -> float:
        """Calculate role-specific performance metrics"""
        position = player_stats.get('position', 'mid').lower()
        recent_matches = player_stats.get('recent_matches', [])
        
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
                cs_score = min(avg_cs / 300.0, 1.0)  # Normalize to 300 CS
                damage_score = min(avg_damage / 20000.0, 1.0)  # Normalize to 20k damage
                
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
                assist_score = min(avg_assists / 15.0, 1.0)  # Normalize to 15 assists
                vision_score = min(avg_vision / 50.0, 1.0)   # Normalize to 50 vision
                death_score = max(0, 1.0 - (avg_deaths / 5.0))  # Lower deaths = better
                
                return (assist_score * 0.5) + (vision_score * 0.3) + (death_score * 0.2)
                
            elif position == 'mid':
                # Mid: Focus on KDA and damage
                kda_scores = [m.get('kda', 0) for m in recent_matches if m is not None]
                damage_scores = [m.get('damage', 0) for m in recent_matches if m is not None]
                
                avg_kda = np.mean(kda_scores) if kda_scores else 0
                avg_damage = np.mean(damage_scores) if damage_scores else 0
                
                # Mid performance: KDA and damage focused
                kda_score = min(avg_kda / 5.0, 1.0)  # Normalize to 5.0 KDA
                damage_score = min(avg_damage / 25000.0, 1.0)  # Normalize to 25k damage
                
                return (kda_score * 0.6) + (damage_score * 0.4)
                
            elif position == 'jungle':
                # Jungle: Focus on assists and vision
                assist_scores = [m.get('assists', 0) for m in recent_matches if m is not None]
                vision_scores = [m.get('vision', 0) for m in recent_matches if m is not None]
                
                avg_assists = np.mean(assist_scores) if assist_scores else 0
                avg_vision = np.mean(vision_scores) if vision_scores else 0
                
                # Jungle performance: Assists and vision focused
                assist_score = min(avg_assists / 12.0, 1.0)  # Normalize to 12 assists
                vision_score = min(avg_vision / 40.0, 1.0)   # Normalize to 40 vision
                
                return (assist_score * 0.7) + (vision_score * 0.3)
                
            else:  # top or default
                # Top: Focus on CS and KDA
                cs_scores = [m.get('cs', 0) for m in recent_matches if m is not None]
                kda_scores = [m.get('kda', 0) for m in recent_matches if m is not None]
                
                avg_cs = np.mean(cs_scores) if cs_scores else 0
                avg_kda = np.mean(kda_scores) if kda_scores else 0
                
                # Top performance: CS and KDA focused
                cs_score = min(avg_cs / 250.0, 1.0)  # Normalize to 250 CS
                kda_score = min(avg_kda / 4.0, 1.0)  # Normalize to 4.0 KDA
                
                return (cs_score * 0.6) + (kda_score * 0.4)
                
        except Exception as e:
            logger.warning(f"Error calculating role-specific performance: {e}")
            return 0.5
    
    def _calculate_champion_pool_size(self, player_stats: Dict) -> float:
        """Calculate champion pool diversity"""
        recent_matches = player_stats.get('recent_matches', [])
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
            logger.warning(f"Error calculating champion pool size: {e}")
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
            logger.warning(f"Error calculating team synergy: {e}")
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
        if not recent_matches:
            return 0.5
        
        try:
            # Get tournament context for pressure assessment
            tournament = prop_request.get('tournament', '').lower()
            opponent = prop_request.get('opponent', '').lower()
            
            # Define pressure levels based on tournament tier
            pressure_weights = {
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
            
            # Calculate base pressure from tournament
            base_pressure = 0.5  # Default medium pressure
            for pressure_key, weight in pressure_weights.items():
                if pressure_key in tournament:
                    base_pressure = weight
                    break
            
            # Adjust pressure based on opponent strength
            if opponent:
                strong_teams = ['t1', 'gen.g', 'jdg', 'blg', 'tes', 'g2', 'fnc', 'c9', 'tl']
                if any(team in opponent for team in strong_teams):
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
            logger.warning(f"Error calculating pressure handling: {e}")
            return 0.5
    
    def _calculate_late_game_performance(self, player_stats: Dict) -> float:
        """Calculate late game performance (simplified proxy)"""
        # Use KDA as a proxy for late game performance
        avg_kda = player_stats.get('avg_kda', 2.0)
        return min(avg_kda / 5.0, 1.0)  # Normalize to [0, 1]
    
    def _calculate_early_game_impact(self, player_stats: Dict) -> float:
        """Calculate early game impact (simplified proxy)"""
        # Use CS as a proxy for early game impact
        avg_cs = player_stats.get('avg_cs', 200.0)
        return min(avg_cs / 300.0, 1.0)  # Normalize to [0, 1]
    
    def _calculate_mid_game_transition(self, player_stats: Dict) -> float:
        """Calculate mid game transition effectiveness"""
        # Use assists as a proxy for mid game transition
        avg_assists = player_stats.get('avg_assists', 5.0)
        return min(avg_assists / 10.0, 1.0)  # Normalize to [0, 1]
    
    def _calculate_objective_control(self, player_stats: Dict) -> float:
        """Calculate objective control (simplified proxy)"""
        # Use vision score as a proxy for objective control
        avg_vision = player_stats.get('avg_vision', 25.0)
        return min(avg_vision / 50.0, 1.0)  # Normalize to [0, 1]
    
    def _calculate_champion_performance_variance(self, player_stats: Dict) -> float:
        """Calculate champion performance variance"""
        recent_matches = player_stats.get('recent_matches', [])
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
            logger.warning(f"Error calculating champion performance variance: {e}")
            return 0.5
    
    def _create_minimal_features(self, prop_request: Dict) -> np.ndarray:
        """Create minimal features for fallback cases"""
        try:
            features = [0.0] * 31  # Initialize with zeros for 31 features (prop_value removed)
            
            # Set basic values (prop_value removed from training features)
            features[16] = len(prop_request.get('map_range', [1]))  # maps_played
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating minimal features: {e}")
            return np.zeros(31, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
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
            'consistency_score': 'Performance consistency score',
            'recent_form_trend': 'Recent form trend (improving/declining)',
            'data_source_quality': 'Quality score of data source',
            'prop_value': 'Target prop value',
            'map_number': 'Map number in series',
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
            'role_specific_performance': 'Role-specific performance score'
        }
        return descriptions.get(feature_name, 'Unknown feature') 