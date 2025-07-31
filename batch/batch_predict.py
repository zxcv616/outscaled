import requests

API_URL = "http://localhost:8000/api/v1/predict"

prediction_requests = [
    {
        "player_name": "Delight",
        "prop_type": "assists",
        "prop_value": 23.5,
        "opponent": "Nongshim RedForce",
        "tournament": "LCK",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Zeka + Viper",
        "prop_type": "kills",
        "prop_value": 18.5,
        "opponent": "Nongshim RedForce",
        "tournament": "LCK",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "shad0w",
        "prop_type": "kills",
        "prop_value": 7.5,
        "opponent": "Weibo Gaming",
        "tournament": "LPL",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Hang",
        "prop_type": "assists",
        "prop_value": 24.5,
        "opponent": "Team WE",
        "tournament": "LPL",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Paduck",
        "prop_type": "kills",
        "prop_value": 11.0,
        "opponent": "T1 Esports Academy",
        "tournament": "LCK Challengers",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Harpoon",
        "prop_type": "kills",
        "prop_value": 11.0,
        "opponent": "BDS Academy",
        "tournament": "LFL",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Clozer",
        "prop_type": "assists",
        "prop_value": 13.0,
        "opponent": "FearX",
        "tournament": "LCK",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    },
    {
        "player_name": "Viper",
        "prop_type": "kills",
        "prop_value": 10.5,
        "opponent": "Nongshim RedForce",
        "tournament": "LCK",
        "map_range": [1, 2],
        "start_map": 1,
        "end_map": 2
    }
]

def make_prediction(payload):
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"\n🧠 Prediction for {payload['player_name']} ({payload['prop_type']} {payload['prop_value']})")
            print(f"Result: {result['prediction']} | Confidence: {result['confidence']}%")
            print(f"Reason: {result['reasoning'][:200]}...")  # Trimmed reasoning
        else:
            print(f"\n❌ Error for {payload['player_name']}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"\n⚠️ Exception for {payload['player_name']}: {str(e)}")

def main():
    for request in prediction_requests:
        make_prediction(request)

if __name__ == "__main__":
    main()
