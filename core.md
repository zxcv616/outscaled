# Outscaled.GG – Model Core Logic (Internal Reference)

This document outlines the **core logic** of the Outscaled.GG prediction system. It defines the foundational rules and architecture that **must not be altered** unless explicitly approved.

---

## 🎯 Purpose

Predict whether a League of Legends player will go **OVER** or **UNDER** a given prop (kills, assists, CS, etc.) using professional match data, engineered features, and a calibrated machine learning model with **dual-layered signal architecture**.

---

## 🧠 Core Model Architecture

### 1. **Base Algorithm**

* **Model**: Random Forest Classifier
* **Trees**: 100
* **Max Depth**: 8
* **Calibration**: Isotonic Calibration using `CalibratedClassifierCV`
* **Output**: `predict_proba()` → confidence score

### 2. **Feature Vector (DUAL-LAYERED ARCHITECTURE)**

* Total of **44 engineered features** (upgraded from 31)
* **Dual-layered signal architecture** balances recent vs long-term performance:

#### **Base Features (14) - Recent Performance**
- Season averages (kills, assists, CS, etc.)
- Recent form (last 5 games)
- Win rate, KDA, GPM, KP percentage

#### **Long-term Averages (6) - NEW: Full Dataset Performance**
- `longterm_kills_avg` - Full dataset kills average
- `longterm_assists_avg` - Full dataset assists average  
- `longterm_cs_avg` - Full dataset CS average
- `longterm_kda` - Full dataset KDA ratio
- `longterm_gpm` - Full dataset gold per minute
- `longterm_kp_percent` - Full dataset kill participation

#### **Derived Features (17) - Enhanced Analysis**
- Volatility metrics (std dev, coefficient of variation)
- Role-specific adjustments
- Map-range normalized stats
- Opponent & tournament context (pressure handling)
- Champion pool analysis
- Team synergy metrics

#### **Deviation Features (4) - NEW: Form Analysis**
- `form_deviation_ratio` - Recent vs long-term performance ratio
- `form_z_score` - Statistical deviation from recent performance mean
- `form_trend` - Linear regression slope of recent performance
- `form_confidence` - Confidence score based on sample size and data quality

#### **Additional Features (3) - System Features**
- System-added features for enhanced analysis

---

## 🔄 Prediction Flow

1. **Input Validation**

   * Ensure all required fields are present
   * Autocomplete corrects player and prop type

2. **Feature Engineering**

   * Use `FeatureEngineer` class to compute all 44 features
   * **NEW**: Calculate long-term averages from full dataset
   * **NEW**: Compute deviation features for form analysis
   * Normalize by map count where applicable

3. **Scaling**

   * Features passed through **StandardScaler**
   * Must match training-time scaler

4. **Model Inference**

   * Predict using Random Forest
   * Output = probability of **MORE**

5. **Enhanced Confidence Calculation**

   * **NEW**: Incorporate `form_z_score` for better balance
   * If z-score (prop vs. recent avg) is available:

     * `z = (prop - recent_avg) / std`
     * If `abs(z) > 3`:

       * `confidence = min(99.9, abs(z) * 10)`
   * **NEW**: Form Z-score adjustments:
     * High form_z_score (>1.5): Boost confidence if prediction aligns with recent form
     * Low form_z_score (<-1.5): Reduce confidence if prediction contradicts long-term average
   * Otherwise: `confidence = calibrated_probability * 100`

6. **Prediction Decision**

   * If probability > 0.5 → **MORE**
   * Else → **LESS**

---

## 🚫 Rule-Based Overrides (When Allowed)

Overrides are allowed **only in edge cases**, such as:

* Missing data or NaN in critical features
* Z-score > 6 (statistically impossible)
* Coefficient of variation (CV) > 140%

If override is triggered:

* Force conservative prediction (usually **LESS**)
* Set `rule_override = True`
* Use clear language in reasoning (e.g., "Unrealistically high prop value")

---

## ✅ Output JSON Structure (Required)

```json
{
  "prediction": "MORE" | "LESS",
  "confidence": 70.0,
  "reasoning": "...",
  "model_mode": "primary" | "rule_based",
  "rule_override": true | false,
  "scaler_status": "loaded" | "missing",
  "player_stats": {...}
}
```

---

## 🔧 Core Logic Specification: `predictor.py` (Prediction Engine)

### **Purpose**

Handles League of Legends prop predictions (e.g., kills, assists) by:

* Loading a trained ML model and scaler
* Engineering features from player stats using **dual-layered architecture**
* Applying calibrated predictions with **enhanced confidence calculation**
* Returning confidence, reasoning, and override metadata

### **1. Class: `PropPredictor`**

#### Initialization

* Loads:

  * ML model (`RandomForestClassifier` or fallback)
  * Scaler (`StandardScaler`)
  * `FeatureEngineer` for custom feature generation (44 features)
  * `FeaturePipeline` for model-scaler version management
  * Config file (default: `prop_config.yaml`)
* Paths:

  * Model: `settings.MODEL_PATH`
  * Config: `"backend/app/ml/config/prop_config.yaml"`
  * Metadata: `"backend/app/ml/models/model_metadata.json"`

### **2. Key Methods**

#### `load_model()`

* Loads model and scaler from disk
* Falls back to uncalibrated model if primary is missing
* Tracks model status (`primary` or `fallback`) and scaler status

#### `predict(player_stats, prop_request)`

* **Step-by-step:**

  1. Use `FeaturePipeline.engineer_features()` to create 44-feature vector
  2. **NEW**: Calculate long-term averages and deviation features
  3. Apply scaler transformation
  4. Predict class (`MORE` or `LESS`) via calibrated model
  5. **NEW**: Enhanced confidence calculation incorporating `form_z_score`
  6. Check for rule overrides (e.g., extreme prop values, volatility)
  7. Return:

     * `prediction`, `confidence`
     * `reasoning`
     * `model_mode`, `rule_override`, `scaler_status`

---

## 🌐 Core Logic Specification: `main.py` (FastAPI Entry Point)

### **Purpose**

Initializes and configures the FastAPI web server for the Outscaled.GG prediction API.

### **Key Components**

* Creates a FastAPI instance with:

  * Title: `Outscaled.GG API`
  * Docs: `/docs`, Redoc: `/redoc`

* Configures **CORS Middleware**:

  * Allows frontend connections from:

    * `http://localhost:3000`
    * `https://outscaled.gg`
  * Accepts all headers, methods, and credentials

* Mounts all versioned API routes via:

  ```python
  app.include_router(router, prefix="/api/v1")
  ```

* Defines root (`/`) and health check (`/health`) endpoints:

  * `/`: returns welcome message and API version
  * `/health`: returns `{"status": "healthy"}`

* Runs `uvicorn` server if launched directly

---

## 📡 Core Logic Specification: `routes.py` (API Layer)

### **Purpose**

Defines all REST API endpoints used to:

* Search players, teams, and validate input
* Request predictions and statistical breakdowns
* Serve ML model metadata

### **Key Endpoints**

* `GET /health` → Simple server check
* `GET /players/search` → Search player list
* `GET /players/popular` → Return top players (hardcoded logic for now)
* `GET /players/validate/{player_name}` → Validates name + provides suggestions
* `GET /playerstats/{player_name}` → Returns recent stats and match metrics

### **Prediction Endpoint**

* `POST /predict`

  * Validates player
  * Retrieves player stats
  * Passes request to `PropPredictor.predict()` with **dual-layered features**
  * Includes `verbose` option
  * Adds no-cache headers

### **Statistical Analysis Endpoints**

* `POST /statistics/probability-distribution`

  * Returns probability curve around prop using z-distribution
  * Uses mock DB to reduce dependency

* `POST /statistics/insights`

  * Returns z-score, volatility, trends, and confidence intervals

* `POST /statistics/comprehensive`

  * Combines prediction, probability, and insights into one

### **Implementation Details**

* Uses lazy singletons for `DataFetcher` and `PropPredictor`
* Uses `Depends(get_db)` for SQLAlchemy integration
* API responses are JSON and no-cache
* Includes suggestion mechanism if player not found

---

## 🔒 Change Policy

**DO NOT modify the model's prediction, confidence, or reasoning logic without updating this document and notifying the team.**

**NEW**: The dual-layered signal architecture is now **core to the system** and must not be altered without approval.

Use this doc as a source of truth for all prediction-related decisions.
