# 🧠 Outscaled.GG – Internal Technical Plan and Dataset Logic

## 📆 Data Source: Oracle's Elixir (2025 Dataset)

**Primary File**: `2025_LoL_esports_match_data_from_OraclesElixir.csv`

This CSV file includes match-level and player-level statistics from professional League of Legends matches. It serves as the **core dataset** for our machine learning model pipeline.

### ✅ What It Contains

* Match metadata: `league`, `gameId`, `date`, `patch`, `side`, `result`
* Player stats: `kills`, `deaths`, `assists`, `dpm`, `gold`, `cs`, `xp`, `vision`, etc.
* Champion info: `champion`, `role`, `position`
* Team context: `teamname`, `opponent`, `earnedgoldshare`, `datacompleteness`

---

## ✅ Why This Dataset

* **Labeled, structured**: Contains clean numerical data ideal for modeling.
* **Pro-level only**: Avoids noisy solo queue data.
* **No API keys required**: No rate limits, timeouts, or authentication steps.
* **Local + fast**: Great for repeatable, offline experimentation.

---

## 🛠️ Project Pipeline Overview

### 1. 📊 Data Cleaning & Preprocessing

* Convert column types: floats, categories
* Drop junk rows (e.g., `league` is NaN)
* Rename ambiguous fields (e.g., `position` vs. `role`)
* Normalize continuous variables (e.g., CS, gold)
* Cap or clip extreme outliers (e.g., 1000 kills)

### 2. 🔧 Feature Engineering

* **Aggregate**: stats by `gameId`, `playername`, `match_series`, `map_index_within_series`
* **Create**:

  * `kda = (kills + assists) / max(1, deaths)`
  * `gpm = gold / duration`
  * `kp% = (kills + assists) / team_kills`
  * `cs@15`, `xpd@15`, `earnedgoldshare`
* **Contextual Features**:

  * `position_factor` (based on role: ADC, Top, etc.)
  * `tournament_tier` weight
  * `opponent_strength` adjustment
  * `map_range` aggregation (e.g., Maps 1-2 only)
  * `pressure_handling` proxy from tier/opponent

### 3. 🎯 Labeling Targets

#### Classification:

* Predict **OVER/UNDER** (binary) on props like `kills`, `assists`, `cs`, `gold`
* Based on provided prop value (e.g., 6.5 kills)

#### Regression:

* Predict raw `kills`, `dpm`, `earnedgoldshare`, etc.
* Supports continuous prediction for trend modeling or calibration

### 4. 🤖 Model Training

* Supported models:

  * `RandomForestClassifier` (fallback)
  * `XGBoostClassifier` (primary)
  * `LightGBM` (alternative if memory-constrained)
  * `MLP` (for experiments)

* Includes:

  * CalibratedClassifierCV (for calibrated confidence)
  * Stratified map-based splitting
  * Deterministic seeds (`np.random.seed(42)`, model `random_state=42`)

### 5. 🔍 Evaluation Strategy

* Split by `match_series` or `gameid`, not random
* Metrics:

  * **Classification**: accuracy, precision, recall, AUC
  * **Regression**: MAE, RMSE, R^2
* Visuals:

  * Error distribution by player, team, role
  * Confidence calibration curve

---

## 🌐 Riot API (Optional)

Not required, but useful for:

* **Live matches** via Spectator API
* **Recent solo queue history** for form approximation
* **Account metadata**: summoner name → PUUID → rank

---

## 🔧 Example Code Snippet

```python
import pandas as pd

df = pd.read_csv("2025_LoL_esports_match_data_from_OraclesElixir.csv")
df = df[df["league"].notna()]  # drop invalid rows

# Add KDA
df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)

# See Sneaky's games
sneaky = df[df["playername"] == "Sneaky"]
print(sneaky[["gameid", "kills", "assists", "deaths", "kda"]].head())
```

---

## 📌 Notes for Internal Agents

* File: `2025_LoL_esports_match_data_from_OraclesElixir.csv`
* Always use: `playername`, `position`, `teamname`, `gameid` for filtering
* Inputs: `kills`, `assists`, `deaths`, `dpm`, `gold`, `cs`, `earnedgoldshare`
* Filter by: `datacompleteness == 'complete'`
* For training: only include rows where game was played (exclude DNPs or fillers)
* Match aggregation: Use `gameid` + `match_series` + map index to slice by `Maps 1-2`, `Maps 1-3`, etc.

---

## 🔄 Internal Integration Summary

* Model inputs are derived from this dataset via `FeatureEngineer`
* Prediction is served via `PropPredictor` and FastAPI endpoint `/api/v1/predict`
* Features are scaled using `MinMaxScaler` and tracked by `FeaturePipeline`
* Scaler + Model are versioned and persisted to disk for reproducibility

---

## 🚀 Future Improvements

* Add patch version awareness
* Track champion matchups and counters
* Add lane assignment consistency metric
* Support dynamic adjustment of features via ablation testing
* Surface feature importance via SHAP or gain-based ranking
