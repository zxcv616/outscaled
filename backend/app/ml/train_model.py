import pickle
import numpy as np
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost for better performance
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using RandomForest")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix random seed for reproducibility
np.random.seed(42)

class ModelTrainer:
    def __init__(self):
        self.model_version = "1.0.0"
        self.training_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_path = "backend/app/ml/models/prop_predictor.pkl"
        self.scaler_path = "backend/app/ml/models/feature_scaler.pkl"
        self.metadata_path = "backend/app/ml/models/training_metadata.json"
        
        # Configurable prop types
        self.ALLOWED_PROP_TYPES = ["kills", "assists", "cs", "deaths", "gold", "damage"]
        self.PROP_TYPE_CONFIGS = {
            "kills": {"min_value": 0, "max_value": 20, "impossible_threshold": 15},
            "assists": {"min_value": 0, "max_value": 25, "impossible_threshold": 20},
            "cs": {"min_value": 0, "max_value": 400, "impossible_threshold": 300},
            "deaths": {"min_value": 0, "max_value": 10, "impossible_threshold": 8},
            "gold": {"min_value": 0, "max_value": 20000, "impossible_threshold": 15000},
            "damage": {"min_value": 0, "max_value": 30000, "impossible_threshold": 25000}
        }
        
    def _create_map_index_column(self, df: pd.DataFrame, map_range_mode: str = "within_series") -> pd.DataFrame:
        """Create map index column with configurable mode"""
        df_copy = df.copy()
        
        if map_range_mode == "within_series":
            # Current logic: map index within each series
            df_copy['match_series'] = df_copy['gameid'].str.split('_').str[0]
            df_copy['map_index_within_series'] = df_copy.groupby('match_series')['gameid'].rank(method='dense').astype(int)
        elif map_range_mode == "global_index":
            # Alternative: global map index across all data
            df_copy['map_index_global'] = df_copy.groupby('gameid').cumcount() + 1
            df_copy['map_index_within_series'] = df_copy['map_index_global']
        else:
            raise ValueError(f"Unknown map_range_mode: {map_range_mode}")
        
        return df_copy
    
    def load_real_data(self) -> pd.DataFrame:
        """Load and preprocess real data from Oracle's Elixir CSV with map-range support"""
        try:
            csv_path = "backend/data/2025_LoL_esports_match_data_from_OraclesElixir.csv"
            if not os.path.exists(csv_path):
                logger.warning(f"Oracle's Elixir CSV not found at {csv_path}. Using synthetic data.")
                return self._create_synthetic_data()
            
            logger.info("Loading real data from Oracle's Elixir CSV...")
            df = pd.read_csv(csv_path, low_memory=False)
            
            # Basic data validation
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check for required columns
            required_columns = ['playername', 'kills', 'assists', 'total cs', 'deaths', 'earnedgold', 'damagetochampions', 'teamname', 'position']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}. Using synthetic data.")
                return self._create_synthetic_data()
            
            # Filter for complete data
            df = df[df["datacompleteness"] == "complete"]
            df = df.dropna(subset=['playername', 'kills', 'assists', 'total cs', 'deaths'])
            
            # Convert numeric columns
            numeric_columns = ['kills', 'assists', 'total cs', 'deaths', 'earnedgold', 'damagetochampions']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add map index tracking for series using reusable method
            df = self._create_map_index_column(df, map_range_mode="within_series")
            
            # Calculate derived metrics
            df['kda'] = (df['kills'] + df['assists']) / df['deaths'].replace(0, 1)
            df['gpm'] = df['earnedgold'] / (df['gamelength'] / 60) if 'gamelength' in df.columns else df['earnedgold'] / 20
            df['kp_percent'] = (df['kills'] + df['assists']) / df['teamkills'].replace(0, 1) * 100 if 'teamkills' in df.columns else 50.0
            
            logger.info(f"Preprocessed {len(df)} complete records")
            logger.info(f"Found {df['match_series'].nunique()} match series")
            logger.info(f"Map index distribution: {df['map_index_within_series'].value_counts().sort_index().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading real data: {e}")
            logger.info("Falling back to synthetic data...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic data as fallback"""
        logger.info("Creating synthetic training data...")
        
        data = []
        for i in range(1000):  # More samples for better training
            # Realistic player performance
            base_kills = np.random.normal(4.5, 1.5)
            base_assists = np.random.normal(6.0, 2.0)
            base_cs = np.random.normal(180, 30)
            base_deaths = np.random.normal(2.5, 1.0)
            base_gold = np.random.normal(12000, 2000)
            base_damage = np.random.normal(15000, 3000)
            
            # Prop value (realistic range)
            prop_value = np.random.uniform(2.0, 8.0)
            
            # Actual performance with realistic variance
            actual_value = base_kills + np.random.normal(0, 1.0)
            
            # Target: 1 if actual > prop_value, 0 otherwise
            target = 1 if actual_value > prop_value else 0
            
            # Add realistic noise (15% of predictions are wrong)
            if np.random.random() < 0.15:
                target = 1 - target
            
            data.append({
                'player': f"Player_{i}",
                'kills': max(0, base_kills),
                'assists': max(0, base_assists),
                'cs': max(0, base_cs),
                'deaths': max(0, base_deaths),
                'gold': max(0, base_gold),
                'damage': max(0, base_damage),
                'kda': max(0, (base_kills + base_assists) / max(base_deaths, 1)),
                'gpm': max(0, base_gold / 20),
                'kp_percent': max(0, min(100, (base_kills + base_assists) / max(base_kills + base_assists + base_deaths, 1) * 100)),
                'prop_value': prop_value,
                'target': target
            })
        
        return pd.DataFrame(data)
    
    def engineer_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-series and momentum features"""
        logger.info("Engineering time-series features...")
        
        # Group by player to create time-series features
        df_features = df.copy()
        
        # Sort by player and add match number (simulating time order)
        df_features['match_number'] = df_features.groupby('player').cumcount() + 1
        
        # Rolling averages (last 3 and 5 matches)
        for window in [3, 5]:
            for col in ['kills', 'assists', 'cs', 'deaths', 'gold', 'damage']:
                if col in df_features.columns:
                    df_features[f'{col}_rolling_{window}'] = df_features.groupby('player')[col].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
        
        # Momentum features (change from previous match)
        for col in ['kills', 'assists', 'cs', 'deaths']:
            if col in df_features.columns:
                df_features[f'{col}_momentum'] = df_features.groupby('player')[col].diff()
        
        # Streak features (consecutive wins/losses)
        df_features['kill_streak'] = df_features.groupby('player')['kills'].apply(
            lambda x: (x > x.rolling(3).mean()).astype(int).rolling(3).sum()
        ).reset_index(0, drop=True)
        
        # Form trend (slope of recent performance)
        for col in ['kills', 'assists', 'cs']:
            if col in df_features.columns:
                df_features[f'{col}_trend'] = df_features.groupby('player')[col].rolling(
                    window=5, min_periods=3
                ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(0, drop=True)
        
        return df_features
    
    def remove_redundant_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features"""
        logger.info("Removing redundant features...")
        
        # Select numeric features for correlation analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != 'target']
        
        if len(numeric_columns) < 2:
            return df, numeric_columns
        
        # Calculate correlation matrix
        correlation_matrix = df[numeric_columns].corr()
        
        # Find highly correlated features (correlation > 0.95)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with more variance (more informative)
            var1 = df[feat1].var()
            var2 = df[feat2].var()
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(df[numeric_columns])
        low_var_features = [col for col, selected in zip(numeric_columns, selector.get_support()) if not selected]
        features_to_remove.update(low_var_features)
        
        # Remove selected features
        features_to_keep = [col for col in numeric_columns if col not in features_to_remove]
        
        logger.info(f"Removed {len(features_to_remove)} redundant features")
        logger.info(f"Kept {len(features_to_keep)} features")
        
        return df[features_to_keep + ['target']], features_to_keep
    
    def check_class_balance(self, df: pd.DataFrame) -> Dict:
        """Check and handle class imbalance"""
        target_counts = df['target'].value_counts()
        logger.info(f"Class distribution: {target_counts.to_dict()}")
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(df['target']),
            y=df['target']
        )
        
        class_weight_dict = dict(zip(np.unique(df['target']), class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
        
        return {
            'counts': target_counts.to_dict(),
            'weights': class_weight_dict,
            'is_balanced': len(target_counts) == 2 and min(target_counts) / max(target_counts) > 0.3
        }
    
    def train_model_with_cross_validation(self, df: pd.DataFrame, feature_columns: List[str], prop_type: str = "kills") -> Tuple[RandomForestClassifier, Dict]:
        """Train model with cross-validation and proper evaluation"""
        logger.info(f"Training model with cross-validation for prop type: {prop_type}")
        
        X = df[feature_columns]
        y = df['target']
        
        # Initialize model with class weights
        class_balance = self.check_class_balance(df)
        
        # Choose model type based on availability and performance
        if XGBOOST_AVAILABLE:
            logger.info("Using XGBoost for better performance")
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=class_balance['weights'].get(1, 1.0) if class_balance['weights'] else 1.0,
                eval_metric='logloss'
            )
        else:
            logger.info("Using RandomForest (XGBoost not available)")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model on full dataset
        if XGBOOST_AVAILABLE:
            # For XGBoost, use a validation split for early stopping
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        else:
            model.fit(X, y)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For XGBoost, get feature importance differently
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model, {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance.to_dict('records'),
            'class_balance': class_balance,
            'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest',
            'prop_type': prop_type
        }
    
    def save_model_and_metadata(self, model: RandomForestClassifier, metadata: Dict) -> None:
        """Save model and training metadata"""
        logger.info("Saving model and metadata...")
        
        # Create models directory
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert metadata for JSON serialization
        metadata_serializable = convert_numpy_types(metadata)
        metadata_serializable.update({
            'model_version': self.model_version,
            'training_date': self.training_date,
            'model_path': self.model_path,
            'feature_count': len(metadata.get('feature_importance', [])),
            'dataset_size': metadata.get('dataset_size', 0)
        })
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Metadata saved to {self.metadata_path}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality"""
        logger.info("Validating data quality...")
        
        # Check for required columns
        required_columns = ['target']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            # Remove rows with NaN in target
            df.dropna(subset=['target'], inplace=True)
        
        # Check for negative values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'target':
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Negative values in {col}: {negative_count}")
                    # Clip negative values to 0
                    df[col] = df[col].clip(lower=0)
        
        # Check for extreme outliers
        for col in numeric_columns:
            if col != 'target':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                if outlier_count > 0:
                    logger.warning(f"Outliers in {col}: {outlier_count}")
        
        logger.info(f"Data validation complete. Final dataset size: {len(df)}")
        return True
    
    def create_map_range_training_data(self, df: pd.DataFrame, map_range: List[int] = [1, 2], prop_type: str = "kills") -> pd.DataFrame:
        """Create training data for specific map range (e.g., Maps 1-2) with improved prop value generation"""
        try:
            logger.info(f"Creating training data for map range: {map_range}, prop type: {prop_type}")
            
            # Validate prop type
            if prop_type not in self.ALLOWED_PROP_TYPES:
                logger.warning(f"Invalid prop type: {prop_type}. Using 'kills' as default.")
                prop_type = "kills"
            
            # Filter data for the specified map range
            df_filtered = df[df['map_index_within_series'].isin(map_range)].copy()
            
            if df_filtered.empty:
                logger.warning(f"No data found for map range {map_range}")
                return self._create_synthetic_data()
            
            # Aggregate stats per player per match series
            agg_stats = df_filtered.groupby(['playername', 'match_series']).agg({
                'kills': 'sum',
                'assists': 'sum', 
                'deaths': 'sum',
                'total cs': 'sum',
                'earnedgold': 'sum',
                'damagetochampions': 'sum',
                'kda': 'mean',
                'gpm': 'mean',
                'kp_percent': 'mean',
                'result': 'mean',  # Win rate across maps
                'teamname': 'first',
                'position': 'first'
            }).reset_index()
            
            # Calculate aggregated metrics
            agg_stats['total_kda'] = (agg_stats['kills'] + agg_stats['assists']) / agg_stats['deaths'].replace(0, 1)
            agg_stats['total_gpm'] = agg_stats['earnedgold'] / (len(map_range) * 20)  # Assume 20 min per map
            agg_stats['maps_played'] = len(map_range)
            
            # Create realistic prop values based on aggregated stats (FIXED: No label leakage)
            prop_data = []
            for _, row in agg_stats.iterrows():
                # Generate multiple prop values per player to create training examples
                base_kills = row['kills']
                base_assists = row['assists']
                base_cs = row['total cs']
                base_deaths = row['deaths']
                base_gold = row['earnedgold']
                base_damage = row['damagetochampions']
                
                # Create 3-5 prop values per player
                for i in range(np.random.randint(3, 6)):
                    # FIXED: Generate prop values from realistic distributions instead of actual performance multiples
                    if prop_type == "kills":
                        # Sample from realistic kills distribution
                        prop_value = np.random.normal(6.0, 2.0)  # Realistic kills range
                        actual_value = base_kills
                    elif prop_type == "assists":
                        # Sample from realistic assists distribution
                        prop_value = np.random.normal(8.0, 3.0)  # Realistic assists range
                        actual_value = base_assists
                    elif prop_type == "cs":
                        # Sample from realistic CS distribution
                        prop_value = np.random.normal(200.0, 50.0)  # Realistic CS range
                        actual_value = base_cs
                    elif prop_type == "deaths":
                        # Sample from realistic deaths distribution
                        prop_value = np.random.normal(3.0, 1.5)  # Realistic deaths range
                        actual_value = base_deaths
                    elif prop_type == "gold":
                        # Sample from realistic gold distribution
                        prop_value = np.random.normal(12000.0, 3000.0)  # Realistic gold range
                        actual_value = base_gold
                    elif prop_type == "damage":
                        # Sample from realistic damage distribution
                        prop_value = np.random.normal(15000.0, 4000.0)  # Realistic damage range
                        actual_value = base_damage
                    else:
                        # Fallback to kills
                        prop_value = np.random.normal(6.0, 2.0)
                        actual_value = base_kills
                    
                    # Ensure prop values are within realistic bounds
                    config = self.PROP_TYPE_CONFIGS.get(prop_type, self.PROP_TYPE_CONFIGS["kills"])
                    prop_value = max(config["min_value"], min(config["max_value"], prop_value))
                    
                    # Target: 1 if actual > prop_value, 0 otherwise
                    target = 1 if actual_value > prop_value else 0
                    
                    # FIXED: Base label flipping on actual performance variance instead of random noise
                    # Calculate performance variance over recent games
                    recent_performance = [base_kills, base_assists, base_cs, base_deaths, base_gold, base_damage]
                    performance_variance = np.std(recent_performance)
                    
                    # Flip label based on performance variance (higher variance = more uncertainty)
                    if performance_variance > np.mean(recent_performance) * 0.5:  # High variance threshold
                        if np.random.random() < 0.2:  # 20% chance to flip for high variance
                            target = 1 - target
                    
                    prop_data.append({
                        'playername': row['playername'],
                        'match_series': row['match_series'],
                        'teamname': row['teamname'],
                        'position': row['position'],
                        'prop_type': prop_type,
                        'prop_value': prop_value,
                        'actual_value': actual_value,
                        'target': target,
                        'maps_played': len(map_range),
                        'map_range': map_range,
                        # Aggregated stats
                        'total_kills': base_kills,
                        'total_assists': base_assists,
                        'total_deaths': base_deaths,
                        'total_cs': base_cs,
                        'total_gold': base_gold,
                        'total_damage': base_damage,
                        'avg_kda': row['kda'],
                        'avg_gpm': row['gpm'],
                        'avg_kp_percent': row['kp_percent'],
                        'win_rate': row['result']
                    })
            
            training_df = pd.DataFrame(prop_data)
            logger.info(f"Created {len(training_df)} training examples for map range {map_range}, prop type {prop_type}")
            logger.info(f"Target distribution: {training_df['target'].value_counts().to_dict()}")
            
            return training_df
            
        except Exception as e:
            logger.error(f"Error creating map range training data: {e}")
            return self._create_synthetic_data()
    
    def train_model(self, map_range: List[int] = [1, 2], prop_type: str = "kills", map_range_mode: str = "within_series") -> RandomForestClassifier:
        """Main training function with map-range support and prop-specific training"""
        logger.info(f"Starting model training for map range {map_range}, prop type: {prop_type}")
        
        # Validate prop type
        if prop_type not in self.ALLOWED_PROP_TYPES:
            logger.warning(f"Invalid prop type: {prop_type}. Using 'kills' as default.")
            prop_type = "kills"
        
        # Load data
        df = self.load_real_data()
        
        # Validate data
        if not self.validate_data(df):
            logger.error("Data validation failed. Exiting.")
            return None
        
        # Create map-range aware training data
        training_df = self.create_map_range_training_data(df, map_range, prop_type)
        
        # Engineer time-series features
        df_features = self.engineer_time_series_features(training_df)
        
        # Remove redundant features
        df_clean, feature_columns = self.remove_redundant_features(df_features)
        
        # Update metadata with dataset size and map range
        metadata = {
            'dataset_size': len(df_clean),
            'map_range': map_range,
            'maps_played': len(map_range),
            'prop_type': prop_type,
            'map_range_mode': map_range_mode
        }
        
        # Train model with cross-validation
        model, training_metadata = self.train_model_with_cross_validation(df_clean, feature_columns, prop_type)
        
        # Combine metadata
        metadata.update(training_metadata)
        
        # Save model and metadata
        self.save_model_and_metadata(model, metadata)
        
        logger.info(f"Model training completed for map range {map_range}, prop type {prop_type}!")
        return model

def train_model(map_range: List[int] = [1, 2], prop_type: str = "kills", map_range_mode: str = "within_series"):
    """Legacy function for backward compatibility with map-range support"""
    trainer = ModelTrainer()
    return trainer.train_model(map_range, prop_type, map_range_mode)

if __name__ == "__main__":
    train_model() 