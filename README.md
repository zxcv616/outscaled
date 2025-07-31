# Outscaled.GG

Machine learning platform that predicts League of Legends player props using professional match data and statistical analysis. 

## Features

- **Smart Player Search**: Autocomplete search with 3,944+ professional players from Oracle's Elixir dataset
- **ML Predictions**: AI-powered OVER/UNDER predictions for kills, assists, CS, deaths, gold, and damage
- **Advanced Analytics**: Statistical reasoning with trend analysis, volatility, z-scores, and performance differentials
- **Dual-Layered Architecture**: Balances recent vs long-term performance with 44 engineered features
- **Multi-Year Data**: Powered by Oracle's Elixir professional match dataset (162,833 matches, 3,944 players from 2024-2025)
- **Data Year Tracking**: API responses include data year distribution (e.g., "2024 (108 matches), 2025 (67 matches)")
- **Beautiful UI**: Glass-morphism design with blurred background and professional interface
- **Extreme Value Handling**: Smart detection of unrealistic prop values with logical responses
- **Comprehensive Testing**: Full test suite covering all API endpoints and edge cases
- **Map-Range Support**: Handles PrizePicks-style props across multiple maps (Maps 1-2, Maps 1-3, etc.)
- **Role-Specific Analysis**: Position-aware feature engineering (ADC, Support, Mid, Jungle, Top)
- **Tournament Context**: Pressure-aware predictions based on tournament tier and opponent strength

## Tech Stack

- **Backend**: FastAPI, Python 3.8+, PostgreSQL, SQLAlchemy
- **ML**: XGBoost (primary), RandomForest (fallback), Scikit-learn, 31 engineered features
- **Frontend**: React, TypeScript, Tailwind CSS, Glass-morphism design
- **Data**: Oracle's Elixir professional match dataset with map-range aggregation
- **Deployment**: Docker, Docker Compose
- **Testing**: Pytest, comprehensive API test suite

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for development)
- Node.js 16+ (for development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/outscaled.git
   cd outscaled
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Demo: http://localhost:3000/demo.html

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /api/v1/players/search` - Player search with autocomplete
- `GET /api/v1/players/validate/{player_name}` - Player validation
- `POST /api/v1/predict` - Make ML predictions
- `GET /api/v1/playerstats/{player_name}` - Get player statistics
- `GET /api/v1/teams` - Get available teams for opponent selection
- `GET /api/v1/model/info` - Model information
- `GET /api/v1/model/features` - Feature importance

### Example Prediction Request (Map-Range Support)

```json
{
  "player_name": "PatkicaA",
  "prop_type": "kills",
  "prop_value": 6.5,
  "opponent": "Gen.G",
  "tournament": "LCS",
  "map_range": [1, 2],
  "start_map": 1,
  "end_map": 2
}
```

### Example Response

```json
{
  "prediction": "MORE",
  "confidence": 85.2,
  "reasoning": "Recent kills average (8.0) exceeds prop value (6.5). showing strong upward trend. with low volatility. prop value is 0.8 standard deviation from average. recent form significantly above season average. against Gen.G. in LCS. champion pool: K'Sante, Rumble, Jayce. High confidence prediction based on High-quality Oracle's Elixir data. Maps 1-2",
  "player_stats": {
    "player_name": "PatkicaA",
    "avg_kills": 4.0,
    "avg_assists": 5.5,
    "avg_cs": 256,
    "win_rate": 0.636,
    "data_years": "2024 (108 matches), 2025 (67 matches)",
    "maps_played": 2,
    "map_range_warning": "",
    "recent_matches": [...]
  }
}
```

## Testing

### **Current Test Status**
- **✅ All Required Tests Passing** (4/4)
- **✅ All Optional Tests Passing** (7/7 total)
- **✅ Frontend Data Display Fixed** - Z-score, percentile, and probability analysis now working
- **✅ Confidence Calculation Improved** - More natural confidence levels

### **Test Coverage**

#### **Core API Tests (13 tests)**
- Health check endpoint
- Player search functionality
- Player validation
- Basic prediction functionality
- Extreme value handling (1000+ kills, negative values)
- Invalid player handling
- Statistical reasoning analysis
- Different prop types (kills, assists, cs, deaths, gold, damage)
- Map-range support (Maps 1-2, Maps 1-3, etc.)
- Model information endpoints
- Feature importance analysis
- Deterministic predictions (same inputs = same outputs)
- Data year tracking in API responses

#### **Advanced Test Suites**
- **Confidence Logic Test** - Validates confidence calculation patterns
- **Betting Logic Test** - Tests PrizePicks-style map range logic
- **Map Ranges Test** - Validates multi-map aggregation
- **Natural Confidence Test** - Ensures confidence levels feel natural
- **Pytest API Tests** - Comprehensive endpoint testing

### **Run API Tests**

```bash
# Install test dependencies
pip install pytest requests

# Run comprehensive tests
python3 tests/run_all_tests.py

# Run individual test suites
python3 tests/run_tests.py
python3 tests/test_confidence_logic.py
python3 tests/test_betting_logic.py
python3 tests/test_map_ranges.py

# Run pytest tests
python3 -m pytest tests/test_api.py -v
```

### **Test Results Summary**
```
🧪 Outscaled.gg Comprehensive Test Suite
============================================================
✅ Main API Test Suite (13 tests) - PASSED
✅ Confidence Logic Test - PASSED  
✅ Betting Logic Test - PASSED
✅ Map Ranges Test - PASSED
✅ Natural Confidence Test - PASSED
✅ Pytest API Tests (23 tests) - PASSED
✅ ML Components Test - PASSED

🎯 Results: 7/7 test suites passed
🎯 Required tests: 4/4 passed
🎉 All required tests passed!
```

### **Test Features Validated**
- Health check and service readiness
- Player search with autocomplete
- Teams endpoint for opponent selection
- Basic predictions with confidence
- Map-range support (PrizePicks style)
- Map range aggregation
- Verbose prediction mode
- Model transparency features
- Extreme value handling
- All prop types (kills, assists, cs, deaths, gold, damage)
- Statistical reasoning analysis
- Model information and feature importance
- Deterministic predictions
- Natural confidence levels
- Betting logic accuracy

## Data Sources

- **Primary**: Oracle's Elixir professional match dataset (2024-2025)
  - **162,833 total matches** (92,543 from 2024, 70,290 from 2025)
  - **3,944 unique players** (1,542 players with data in both years)
  - **636 unique teams** across multiple leagues
  - **58 unique leagues** with comprehensive coverage
  - **Extended temporal coverage**: January 2024 to July 2025
  - **Cross-year player tracking**: Historical performance patterns
  - **Enhanced model training**: More diverse and comprehensive data
  - **Better prediction accuracy**: Multi-year meta evolution insights
  - Complete match statistics with map-range aggregation
  - Map index tracking within match series
  - Multi-year meta evolution tracking
- **Future**: Riot Games API integration (optional add-on)

### Dataset Benefits

**Combined Dataset Advantages**
- **Increased Coverage**: 162,833 matches vs individual year averages
- **Broader Player Base**: 3,944 unique players vs 2,894 (2024) or 2,591 (2025)
- **More Team Diversity**: 636 unique teams vs 456 (2024) or 395 (2025)
- **Extended Timeline**: 18+ months of data vs individual year limitations
- **Cross-Year Analysis**: 1,542 players with data in both years
- **Enhanced ML Training**: Larger, more diverse training dataset
- **Better Predictions**: Historical performance patterns across years

**Data Year Tracking**
- API responses include `data_years` field showing distribution
- Example: `"2024 (108 matches), 2025 (67 matches)"`
- Helps users understand data recency and coverage
- Enables cross-year performance analysis

## ML Model

### Model Architecture

- **Primary Algorithm**: XGBoost Classifier with early stopping
- **Fallback Algorithm**: RandomForest Classifier (when XGBoost unavailable)
- **Calibration**: CalibratedClassifierCV for better probability estimates
- **Features**: **44 engineered features** with **dual-layered signal architecture** (upgraded from 31)
- **Training**: Professional match data with map-range aggregation
- **Validation**: Cross-validation with real prop outcomes
- **Edge Cases**: Smart handling of extreme/unrealistic values

### **Dual-Layered Signal Architecture**

The model now uses a **dual-layered signal architecture** that balances recent vs long-term performance:

#### **Base Features (14) - Recent Performance**
- `avg_kills` (normalized by map count)
- `avg_assists` (normalized by map count)
- `avg_cs` (normalized by map count)
- `avg_deaths` (normalized by map count)
- `avg_gold` (normalized by map count)
- `avg_damage` (normalized by map count)
- `avg_vision` (normalized by map count)
- `recent_kills_avg` (normalized by map count)
- `recent_assists_avg` (normalized by map count)
- `recent_cs_avg` (normalized by map count)
- `win_rate` (percentage)
- `avg_kda` (ratio)
- `avg_gpm` (gold per minute)
- `avg_kp_percent` (kill participation percentage)

#### **Long-term Averages (6) - NEW: Full Dataset Performance**
- `longterm_kills_avg` - Full dataset kills average
- `longterm_assists_avg` - Full dataset assists average
- `longterm_cs_avg` - Full dataset CS average
- `longterm_kda` - Full dataset KDA ratio
- `longterm_gpm` - Full dataset gold per minute
- `longterm_kp_percent` - Full dataset kill participation percentage

#### **Derived Features (17) - Enhanced Analysis**
- `consistency_score` - Performance consistency across matches
- `recent_form_trend` - Linear regression slope of recent performance
- `data_source_quality` - Dynamic quality based on missing fields
- `maps_played` - Number of maps in prop range (e.g., 2 for Maps 1-2)
- `opponent_strength` - Team strength factor (T1, Gen.G, JDG, etc.)
- `tournament_tier` - Tournament importance (Worlds: 1.0, MSI: 0.9, LCS: 0.7)
- `position_factor` - Role-specific importance weights
- `champion_pool_size` - Champion diversity metric
- `team_synergy` - Team performance correlation
- `meta_adaptation` - Champion diversity + consistency
- `pressure_handling` - Context-aware pressure performance
- `late_game_performance` - KDA-based late game proxy
- `early_game_impact` - CS-based early game proxy
- `mid_game_transition` - Assists-based mid game proxy
- `objective_control` - Vision score-based objective control
- `champion_performance_variance` - Performance consistency across champions
- `role_specific_performance` - Position-appropriate metrics

#### **Deviation Features (4) - NEW: Form Analysis**
- `form_deviation_ratio` - Recent vs long-term performance ratio
- `form_z_score` - Statistical deviation from recent performance mean
- `form_trend` - Linear regression slope of recent performance
- `form_confidence` - Confidence score based on sample size and data quality

#### **Additional Features (3) - System Features**
- System-added features for enhanced analysis

### **Enhanced Confidence Calculation**

The model now incorporates **form_z_score** for better balance between recent vs long-term performance:

- **High form_z_score (>1.5)**: Recent form significantly above long-term average
  - Boosts confidence if prediction aligns with recent form
  - Reduces confidence if prediction contradicts recent form
- **Low form_z_score (<-1.5)**: Recent form significantly below long-term average
  - Reduces confidence if prediction contradicts long-term average
  - Slight boost if prediction aligns with long-term average
- **Balanced approach**: Model weighs recency without ignoring long-term skill

### **Benefits of Dual-Layered Architecture**

1. **Reduced Overreactions**: Less overreaction to 1-2 good or bad games
2. **Better Statistical Grounding**: More statistically grounded predictions
3. **Balanced Decision Making**: Model weighs recency without ignoring long-term skill
4. **Improved Robustness**: Better handling of players with large datasets vs small datasets

### Role-Specific Feature Engineering

**ADC (Attack Damage Carry)**
- Focus: CS and damage output
- Metrics: CS/300.0 (60%), Damage/20000.0 (40%)
- Context: Primary damage dealer, farming efficiency critical

**Support**
- Focus: Assists, vision, low deaths
- Metrics: Assists/15.0 (50%), Vision/50.0 (30%), Low deaths (20%)
- Context: Team utility, vision control, survivability

**Mid Lane**
- Focus: KDA and damage output
- Metrics: KDA/5.0 (60%), Damage/25000.0 (40%)
- Context: Primary carry role, high KDA expectations

**Jungle**
- Focus: Assists and vision control
- Metrics: Assists/12.0 (70%), Vision/40.0 (30%)
- Context: Map control, ganking, objective control

**Top Lane**
- Focus: CS and KDA
- Metrics: CS/250.0 (60%), KDA/4.0 (40%)
- Context: Island play, farming, team fighting

### Map-Range Aggregation Logic

**Data Processing**
```python
# Map index tracking within series
df['match_series'] = df['gameid'].str.split('_').str[0]
df['map_index_within_series'] = df.groupby('match_series')['gameid'].rank(method='dense').astype(int)

# Filter for specific map range (e.g., Maps 1-2)
df_filtered = df[df['map_index_within_series'].isin([1, 2])]

# Aggregate stats across map range
agg_stats = df_filtered.groupby(['playername', 'match_series']).agg({
    'kills': 'sum',
    'assists': 'sum',
    'deaths': 'sum',
    'total cs': 'sum'
})
```

**Feature Normalization**
```python
# Instead of multiplication (overinflates values)
avg_kills * maps_played

# Use proper normalization
avg_kills / normalization_factor
```

### Pressure Handling Algorithm

**Tournament Tier Weights**
```python
pressure_weights = {
    'worlds': 1.0,      # Highest pressure
    'msi': 0.9,         # Very high pressure
    'lcs': 0.7,         # High pressure
    'lec': 0.7,         # High pressure
    'lck': 0.8,         # Very high pressure
    'lpl': 0.8,         # Very high pressure
    'playoffs': 0.9,    # High pressure
    'finals': 1.0,      # Highest pressure
}
```

**Strong Team Bonus**
```python
strong_teams = ['t1', 'gen.g', 'jdg', 'blg', 'tes', 'g2', 'fnc', 'c9', 'tl']
if opponent in strong_teams:
    pressure += 0.2
```

### Model Performance

- **Algorithm**: XGBoost Classifier with early stopping
- **Features**: 31 engineered features (no data leakage)
- **Training**: Professional match data with map-range aggregation
- **Validation**: Cross-validation with real prop outcomes
- **Edge Cases**: Smart handling of extreme/unrealistic values
- **Deterministic**: Fixed random seeds ensure reproducible results

### Feature Pipeline Management

**Scaler Management**
```python
# FeaturePipeline class enforces proper scaler usage
pipeline = FeaturePipeline()
pipeline.fit(training_data)
pipeline.save_scaler()
pipeline.load_scaler()
features = pipeline.transform(player_stats, prop_request)
```

## Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm start
```

### Database Setup

```bash
# Create database
docker-compose up db -d

# Run migrations (if needed)
# Currently using SQLAlchemy with auto-creation
```

## Security

- Environment variables for sensitive data
- Comprehensive .gitignore
- No hardcoded credentials
- Input validation and sanitization
- CORS configuration
- Error handling and logging
- Deterministic predictions (no random variance)

## Environment Variables

Create a `.env` file based on `env.example`:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/outscaled

# API Keys (optional)
RIOT_API_KEY=your_riot_api_key_here

# Model Configuration
MODEL_PATH=models/prop_predictor.pkl

# CORS Settings
BACKEND_CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]

# Debug Mode
DEBUG=False
```

## Project Structure

```
outscaled/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Configuration
│   │   ├── ml/            # Machine learning
│   │   │   ├── feature_engineering.py  # 31 features with role-specific logic
│   │   │   ├── predictor.py           # XGBoost + RandomForest
│   │   │   ├── train_model.py        # Map-range training
│   │   │   ├── config/               # Model configuration
│   │   │   └── models/               # Trained models and scalers
│   │   ├── services/      # Data fetching with map aggregation
│   │   ├── schemas/       # Pydantic models
│   │   ├── models/        # Database models
│   │   ├── utils/         # Utility functions
│   │   └── main.py        # FastAPI app
│   ├── tests/             # Test files
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── services/      # API service layer
│   │   ├── types/         # TypeScript type definitions
│   │   └── App.tsx        # Main app with map range UI
│   └── package.json
├── tests/                 # API test suite
├── batch/                 # Batch processing utilities
├── data/                  # Data files (CSV datasets)
├── scripts/               # Analysis and utility scripts
├── demo.html              # Standalone demo
├── docker-compose.yml     # Docker setup
├── .gitignore            # Git ignore rules
└── README.md
```

### **⚠️ Current Issues in Project Structure**

#### **Files Needing Cleanup**
- **Debug/Test Files**: 15+ files in `backend/` root that should be moved to `scripts/` or `tests/`
- **Duplicate Models**: Models exist in both `backend/app/ml/models/` and `backend/backend/app/ml/models/`
- **Large Data Files**: CSV files (72MB + 52MB) should be moved to `data/` directory
- **Analysis Scripts**: `analyze_dataset.py` should be moved to `scripts/`

#### **Recommended Structure Improvements**
```bash
# Create scripts directory for analysis and debug files
mkdir -p scripts/
mv backend/test_*.py scripts/
mv backend/debug_*.py scripts/
mv analyze_dataset.py scripts/

# Move large data files
mkdir -p data/
mv *.csv data/

# Remove duplicate model directory
rm -rf backend/backend/

# Clean up duplicate demo files
# Keep only demo.html, remove prediction-demo.html
```

### **Configuration Files**
- `backend/app/core/config.py` - Application configuration
- `backend/app/ml/config/prop_config.yaml` - Model configuration
- `env.example` - Environment variables template
- `docker-compose.yml` - Docker services configuration

### **Key Directories**
- **`backend/app/ml/`** - Core machine learning logic
- **`backend/app/api/`** - FastAPI route definitions
- **`frontend/src/components/`** - React UI components
- **`tests/`** - Comprehensive test suite
- **`data/`** - Large datasets and model files

## Current Status & Known Issues

### ✅ **Working Features**
- **All core tests passing** (7/7 test suites)
- **Frontend data display fixed** - Z-score, percentile, and probability analysis now show correctly
- **Confidence calculation improved** - More natural confidence levels
- **API endpoints functional** - All endpoints working as expected
- ✅ DUAL-LAYERED ARCHITECTURE IMPLEMENTED - Model now uses 44 features with long-term averages and deviation analysis
- ✅ Model successfully retrained - New architecture is active and tested
- ✅ Enhanced confidence calculation - Incorporates form_z_score for better balance

### ⚠️ **Known Issues & Technical Debt**

#### **Critical Issues**
1. **Configuration Path Inconsistencies**
   - Multiple conflicting model paths in `config.py`
   - `MODEL_PATH` uses relative path while `FALLBACK_MODEL_PATH` uses absolute path
   - `PROP_CONFIG_PATH` uses Docker path format

2. **Hardcoded Database Credentials**
   - Database password "password" hardcoded in `config.py` and `docker-compose.yml`
   - Should use environment variables for production

3. **Duplicate Model Files**
   - Models exist in both `backend/app/ml/models/` and `backend/backend/app/ml/models/`
   - Potential confusion about which model is being used

#### **Code Quality Issues**
1. **Excessive Debug/Test Files**
   - 15+ debug/test files in backend root directory
   - Should be moved to `scripts/` or `tests/` directories

2. **Frontend Error Handling**
   - Using `alert()` for error handling instead of proper React error states
   - Poor user experience

3. **Broad Exception Handling**
   - Many `except Exception as e:` blocks without specific error types
   - Could mask important errors

#### **Unused/Redundant Files**
- `analyze_dataset.py` - One-time analysis script
- `batch/batch_predict.py` - Not imported anywhere
- `demo.html` and `prediction-demo.html` - Duplicate demo files
- Large CSV files (72MB + 52MB) in root directory

### 🔧 **Recommended Cleanup Actions**

#### **Immediate (Critical)**
```bash
# Fix configuration paths
# Standardize all model/config paths in backend/app/core/config.py

# Remove hardcoded credentials
# Use environment variables for all database credentials

# Clean up duplicate model files
# Keep only backend/app/ml/models/ and remove backend/backend/

# Move large data files
mv *.csv data/
```

#### **Short-term (Medium Priority)**
```bash
# Clean up debug files
mkdir scripts/
mv backend/test_*.py scripts/
mv backend/debug_*.py scripts/
mv analyze_dataset.py scripts/

# Improve frontend error handling
# Replace alert() calls with proper React error states

# Add specific exception handling
# Replace broad Exception catches with specific types
```

#### **Long-term (Quality)**
- Standardize error handling across codebase
- Simplify model loading logic
- Add comprehensive logging
- Consolidate documentation files

## Development Guidelines

### **Code Quality Standards**
- Use specific exception types instead of broad `Exception` catches
- Implement proper error states in React components (no `alert()`)
- Add comprehensive type hints to all functions
- Use environment variables for all credentials and sensitive data

### **File Organization**
- Keep debug/analysis scripts in `scripts/` directory
- Store large data files in `data/` directory
- Maintain clean separation between backend, frontend, and utilities

### **Configuration Management**
- All paths should be relative or absolute consistently
- Use environment variables for all credentials
- Document all configuration options

## Security Considerations

### **Current Security Status**
- ✅ Environment variables used for most configuration
- ✅ Comprehensive .gitignore excludes sensitive files
- ✅ No hardcoded API keys in production code
- ⚠️ Hardcoded database password needs to be environment variable
- ⚠️ Some debug files may contain sensitive information

### **Production Readiness Checklist**
- [ ] Remove all hardcoded credentials
- [ ] Clean up debug files and temporary scripts
- [ ] Standardize configuration paths
- [ ] Implement proper error handling
- [ ] Add comprehensive logging
- [ ] Review and update security documentation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Oracle's Elixir** for comprehensive professional match data
- **Riot Games** for League of Legends API
- **FastAPI** for the excellent web framework
- **XGBoost** for superior gradient boosting performance
- **Scikit-learn** for machine learning tools
- **Tailwind CSS** for beautiful styling

---

**Note**: This is a demonstration project showcasing advanced ML techniques for sports analytics. The system is specifically designed for PrizePicks-style map-range props and includes sophisticated role-specific analysis for League of Legends professional play. 

### **Recent Improvements (Latest Update)**
- ✅ **Frontend Data Display Fixed** - Z-score, percentile, and probability analysis now display correctly
- ✅ **Confidence Calculation Improved** - More natural confidence levels with reduced artificial constraints
- ✅ **All Core Tests Passing** - 7/7 test suites passing with comprehensive coverage
- ✅ **API Endpoints Functional** - All endpoints working as expected
- ✅ **DUAL-LAYERED ARCHITECTURE IMPLEMENTED** - Model upgraded from 31 to 44 features with long-term averages and deviation analysis
- ✅ **Model Successfully Retrained** - New architecture is active and tested with all features working
- ✅ **Enhanced Confidence Calculation** - Incorporates form_z_score for better balance between recent vs long-term performance
- ✅ **Reduced Overreactions** - Less overreaction to 1-2 good or bad games through dual-layered signal architecture

### **Production Readiness**
The system is functional and all core features are working correctly. However, before production deployment, consider addressing the technical debt items listed in the "Current Status & Known Issues" section, particularly:
- Standardizing configuration paths
- Removing hardcoded credentials
- Cleaning up debug files
- Improving error handling

For production use, ensure proper data licensing, API rate limits, and compliance with betting regulations. 