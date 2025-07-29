🧠 Outscaled.gg – Updated Plan Using Oracle’s Elixir Dataset
📦 Data Source
We're now using a single CSV file from Oracle’s Elixir which contains historical League of Legends esports match data:

File: 2025_LoL_esports_match_data_from_OraclesElixir.csv

Includes:

Match metadata (league, gameId, date)

Player stats (kills, deaths, assists, gold, etc.)

Champion picks and positions

Team context and outcome flags

✅ Why This Approach
📚 Rich, labeled historical data (good for ML)

🏆 Covers pro matches (not solo queue)

📁 No rate limits or key expiration like the Riot API

🚀 Fast local iteration for modeling

🧱 Updated Project Plan
1. 📊 Data Cleaning & Preprocessing
Convert column types (e.g., numbers, categories)

Drop or fill missing values

Rename ambiguous columns

Normalize numerical stats

2. 🔧 Feature Engineering
Aggregate stats by team, role, etc.

Create new features: KDA, GPM, KP%, XPD@15, CS@15

Include match context like patch, side, and opponent

3. 🎯 Labeling Targets
Predict:

Over/Under (binary) props for stats

Regression: e.g. predicted Kills, DPM

4. 🤖 Model Training
Models to try:

Random Forest

LightGBM / XGBoost

Simple MLP / FFNN

5. 🔍 Evaluation
Split by date or matchId (not random)

Metrics: MAE, RMSE, classification accuracy

Visualize errors by player/team/role

🌐 Riot API (Optional Add-On)
Still useful for:

Live match detection (Spectator API)

Recent solo queue match history

Linking summoner → PUUID → current rank

But not required for prediction pipeline.

🛠 Example Code
python
Copy
Edit
import pandas as pd

df = pd.read_csv("2025_LoL_esports_match_data_from_OraclesElixir.csv")
df = df[df["league"].notna()]  # filter out junk rows

# Add a KDA column
df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].replace(0, 1)

# See Sneaky’s games
sneaky = df[df["playername"] == "Sneaky"]
print(sneaky[["gameid", "kills", "assists", "deaths", "kda"]].head())
📌 Notes for the Agent
File: 2025_LoL_esports_match_data_from_OraclesElixir.csv

Use playername, position, teamname, gameid for filters

Use kills, assists, deaths, dpm, earnedgoldshare, etc. as input or targets

Always check datacompleteness == 'complete'