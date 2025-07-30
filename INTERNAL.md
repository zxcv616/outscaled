# INTERNAL.md

## Purpose

This internal document defines the domain knowledge, decision logic, and structural assumptions used by the Outscaled.GG platform. It is intended for internal LLM agents (such as Cursor AI) and engineers working on the system.

---

## 🧠 Domain Overview

Outscaled.GG predicts OVER/UNDER outcomes for League of Legends player performance stats ("props") using machine learning. The system mimics platforms like **PrizePicks**, which offer esports props like "More or Less than 6.5 kills" across a range of maps (Maps 1-2, Maps 1-3).

### Key Betting Concepts

* **Prop Value**: A line set for a stat (e.g. 6.5 kills). We predict if the player's actual performance will be MORE or LESS.
* **Confidence**: A model-calibrated probability (0-100%) of the prediction.
* **Extreme Value Handling**: Props with unrealistic values (e.g. 1000 kills) are filtered early and assigned logical defaults.
* **Rule-Based Overrides**: In clear cases (e.g., recent performance far above/below prop value), the system can override the ML model.

### Map Range Concept

PrizePicks props are defined across map ranges (e.g., Maps 1-2 of a best-of-3 series). Our model aggregates data accordingly:

```python
match_series = df['gameid'].str.split('_').str[0]
df['map_index_within_series'] = df.groupby('match_series')['gameid'].rank(method='dense')
```

This ensures stats reflect the exact subset of maps the prop applies to.

---

## 🎮 League of Legends Domain Logic

### Roles

Each player is assigned a **role** (ADC, Support, Mid, Jungle, Top). Role-specific behavior affects stats:

* **ADC**: Prioritized CS and damage output
* **Support**: Vision, assists, and survivability
* **Jungle**: Assists, map control
* **Top**: Mixed KDA and CS focus
* **Mid**: High KDA and burst damage

### Champion Pool Analysis

* Diversity in champion usage is tracked (e.g. 3 unique champs in last 5 games)
* Role-aware champion context is planned for future (e.g., different interpretation of pool variety for Support vs Top)
* **Champion pool listed includes up to 3 most frequently used champions in the map range or recent matches**
* Champion selection prioritizes recent matches within the specified map range

### Tournament Context

* Each match includes a tournament tag (e.g., "Worlds", "LCS"). We assign pressure weights:

```python
pressure_weights = { 'worlds': 1.0, 'lcs': 0.7, 'msi': 0.9 }
```

* Opponent team strength is also tracked (e.g., playing vs Gen.G adds pressure)

---

## 🧠 Model Internals

### Primary Model

* `XGBoostClassifier` trained on 31 engineered features
* Calibrated with `CalibratedClassifierCV(method='isotonic')`
* Fallback: RandomForest with `sigmoid` calibration

### Features

Features are grouped into:

* **Raw Player Stats**: Kills, assists, CS, gold, damage, win rate
* **Recent Form**: Averages and trends over the last \~5–10 matches
* **Contextual Metrics**: Opponent, tournament tier, map range, champion pool size
* **Derived Features**: Z-score, volatility, consistency, pressure response, etc.

### Confidence Adjustments

Confidence is modulated by:

* Volatility (coefficient of variation)
* Win streak / loss streak
* Sample size (small n = lower confidence)
* Recent performance vs season average
* **If `matches_in_range` < 3, prediction confidence is reduced by 10–20%, depending on volatility**

### Prediction Labels

* `MORE`: Prediction that player exceeds the prop value
* `LESS`: Prediction that player fails to reach the prop value

---

## 🧪 Reasoning Logic

The explanation system generates natural language reasons like:

> "Recent kills average (8.0) exceeds prop value (6.5), showing strong upward trend, with low volatility. Against Gen.G in LCS. Champion pool: Rumble, Jayce, K'Sante. High confidence prediction."

Explanation components include:

* Recent vs average comparison
* Trend (up/down/stable)
* Volatility category
* Z-score relative to season stats
* Tournament and opponent context
* **If map range is defined (e.g., Maps 1-2), it is explicitly included at the end of the reasoning for clarity**
* **Match count in range (e.g., "Based on 2 maps in range") may be added to explanations when relevant**

### Missing Scaler Handling

* **If scaler is missing, confidence may be less reliable. A warning is included in the reasoning string when applicable**
* **Internal Behavior**: Add fallback logic for scaler or explicitly state that prediction was made using raw (unscaled) features
* **Scaler Status**: `scaler_status` field indicates "loaded", "missing", or "fallback" - affects prediction reliability

### Rule-Based Overrides

* **Overrides occur if confidence is < 60% AND recent form shows large deviation from prop (Z > 2.0)**
* **Override still uses model but adjusts prediction toward conservative outcome**
* Rule-based overrides only occur when confidence is below 60%
* Recent form override logic is good but limited to low-confidence predictions

### Confidence Source Transparency

* **When `verbose` is enabled, include which features influenced confidence (e.g., volatility + sample size + model type)**
* Confidence adjustments are logged and can be included in verbose reasoning
* Model mode (primary/fallback) affects confidence reliability

---

## 🔍 API Behavior

### Response Fields

The API returns structured prediction data:

```json
{
  "prediction": "MORE",
  "confidence": 83.5,
  "reasoning": "...",
  "model_mode": "primary",  // or "fallback", "rule_based"
  "rule_override": false,
  "scaler_status": "loaded",  // "loaded", "missing", or "fallback"
  "features_used": [...],
  "data_source": "model_prediction"
}
```

### Validation Logic

* `prop_value` must be numeric and non-negative
* Extreme values are caught before ML model
* Rule-based overrides only occur when confidence is < 60%
* Final confidence is always capped between 10–95%

---

## 🔁 Planned Improvements

1. **Add `opponent_strength` and `tournament_tier` as model features**
2. **Track patch version or `meta_shift_flag`** for better trend analysis
3. **Dynamic thresholds**: Replace hardcoded impossible values (e.g., 15 kills) with statistical upper bounds
4. **Role-specific models**: Use different classifiers per role
5. **Add `model_version`, `feature_version` to API output** for traceability

---

## ✅ Summary for Internal AI Agents

When helping write or debug this project:

* Always reason using League + sports betting logic
* PrizePicks-style props are OVER/UNDER bets across map ranges
* Don't treat 6.5 kills as universally high or low — context matters (ADC vs Support, tournament, opponent)
* Never modify `prop_value` — it's the target threshold
* Do not remove or bypass extreme value filters unless you're updating their logic
* Use 31 features; `prop_value` is **not** a model input (it would cause leakage)
* Always return predictions with reasoning and capped confidence
* Keep predictions deterministic (fixed seed)
* **Missing scalers reduce confidence reliability - include warnings in reasoning**
* **Map range affects data aggregation but recent matches should be consistent**
* **Champion pool includes up to 3 most frequent recent champions**
* **Rule overrides only apply when confidence < 60% AND Z-score > 2.0**
