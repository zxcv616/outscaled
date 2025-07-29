# Outscaled.GG

Machine learning platform that predicts League of Legends player props using professional match data and statistical analysis. 

## ✨ Features

- **🔍 Smart Player Search**: Autocomplete search with 2,500+ professional players from Oracle's Elixir dataset
- **🤖 ML Predictions**: AI-powered OVER/UNDER predictions for kills, assists, CS, deaths, gold, and damage
- **📊 Advanced Analytics**: Statistical reasoning with trend analysis, volatility, z-scores, and performance differentials
- **🎯 Real Data**: Powered by Oracle's Elixir professional match dataset (177,660 matches, 3,945 players from 2024-2025)
- **🎨 Beautiful UI**: Glass-morphism design with blurred background and professional interface
- **⚡ Extreme Value Handling**: Smart detection of unrealistic prop values with logical responses
- **📈 Comprehensive Testing**: Full test suite covering all API endpoints and edge cases
- **🗺️ Map-Range Support**: Handles PrizePicks-style props across multiple maps (Maps 1-2, Maps 1-3, etc.)
- **🎮 Role-Specific Analysis**: Position-aware feature engineering (ADC, Support, Mid, Jungle, Top)
- **🏆 Tournament Context**: Pressure-aware predictions based on tournament tier and opponent strength

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+, PostgreSQL, SQLAlchemy
- **ML**: XGBoost (primary), RandomForest (fallback), Scikit-learn, 31 engineered features
- **Frontend**: React, TypeScript, Tailwind CSS, Glass-morphism design
- **Data**: Oracle's Elixir professional match dataset with map-range aggregation
- **Deployment**: Docker, Docker Compose
- **Testing**: Pytest, comprehensive API test suite

## 🚀 Quick Start

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

## 📋 API Endpoints

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
    "maps_played": 2,
    "map_range_warning": "",
    "recent_matches": [...]
  }
}
```

## 🧪 Testing

### Run API Tests

```bash
# Install test dependencies
pip install pytest requests

# Run comprehensive tests
python tests/run_tests.py

# Run pytest tests
pytest tests/test_api.py -v
```

### Test Coverage

- ✅ Health check endpoint
- ✅ Player search functionality
- ✅ Player validation
- ✅ Basic prediction functionality
- ✅ Extreme value handling (1000+ kills, negative values)
- ✅ Invalid player handling
- ✅ Statistical reasoning analysis
- ✅ Different prop types (kills, assists, cs, deaths, gold, damage)
- ✅ Map-range support (Maps 1-2, Maps 1-3, etc.)
- ✅ Model information endpoints
- ✅ Feature importance analysis
- ✅ Deterministic predictions (same inputs = same outputs)

## 📊 Data Sources

- **Primary**: Oracle's Elixir professional match dataset (2024-2025)
  - 177,660 matches analyzed (100,956 from 2024, 76,704 from 2025)
  - 3,945 professional players (1,542 players in both years)
  - 636 unique teams across multiple leagues
  - Complete match statistics with map-range aggregation
  - Map index tracking within match series
  - Multi-year meta evolution tracking
- **Future**: Riot Games API integration (optional add-on)

## 🤖 ML Model

### Model Architecture

- **Primary Algorithm**: XGBoost Classifier with early stopping
- **Fallback Algorithm**: RandomForest Classifier (when XGBoost unavailable)
- **Calibration**: CalibratedClassifierCV for better probability estimates
- **Features**: 31 engineered features (prop_value removed to prevent data leakage)
- **Training**: Professional match data with map-range aggregation
- **Validation**: Cross-validation with real prop outcomes
- **Edge Cases**: Smart handling of extreme/unrealistic values

### Features (31 total)

**Player Statistics (14) - Map-Range Normalized**
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

**Derived Features (17)**
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

## 🎨 UI/UX Features

- **Glass-morphism Design**: Modern blurred background with glass effects
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Player Search**: Real-time autocomplete with 2,500+ players
- **Map Range Selection**: Start Map and End Map dropdowns for PrizePicks-style props
- **Visual Feedback**: Loading states, confidence indicators, color-coded results
- **Professional Styling**: Clean, modern interface with proper typography
- **Tournament Context**: Opponent team dropdown populated from dataset

## 🔧 Development

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

## 🔐 Security

- ✅ API keys removed from code
- ✅ Environment variables for sensitive data
- ✅ Comprehensive .gitignore
- ✅ No hardcoded credentials
- ✅ Input validation and sanitization
- ✅ CORS configuration
- ✅ Error handling and logging
- ✅ Deterministic predictions (no random variance)

## 🌍 Environment Variables

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

## 📁 Project Structure

```
outscaled/
├── backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Configuration
│   │   ├── ml/            # Machine learning
│   │   │   ├── feature_engineering.py  # 31 features with role-specific logic
│   │   │   ├── predictor.py           # XGBoost + RandomForest
│   │   │   └── train_model.py        # Map-range training
│   │   ├── services/      # Data fetching with map aggregation
│   │   └── main.py        # FastAPI app
│   ├── tests/             # Test files
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/    # React components
│   │   └── App.tsx        # Main app with map range UI
│   └── package.json
├── tests/                 # API test suite
├── demo.html              # Standalone demo
├── docker-compose.yml     # Docker setup
├── .gitignore            # Git ignore rules
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Oracle's Elixir** for comprehensive professional match data
- **Riot Games** for League of Legends API
- **FastAPI** for the excellent web framework
- **XGBoost** for superior gradient boosting performance
- **Scikit-learn** for machine learning tools
- **Tailwind CSS** for beautiful styling

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This is a demonstration project showcasing advanced ML techniques for sports analytics. For production use, ensure proper data licensing, API rate limits, and compliance with betting regulations. The system is specifically designed for PrizePicks-style map-range props and includes sophisticated role-specific analysis for League of Legends professional play. 