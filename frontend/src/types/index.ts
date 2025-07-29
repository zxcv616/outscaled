export interface PropRequest {
  player_name: string;
  prop_type: string;
  prop_value: number;
  opponent?: string;
  tournament?: string;
  map_number?: number;
}

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  player_name: string;
  prop_type: string;
  prop_value: number;
  reasoning: string;
  recent_stats: PlayerStats;
}

export interface PlayerStats {
  player_name: string;
  team: string;
  role: string;
  recent_matches: PlayerMatch[];
  avg_kills: number;
  avg_assists: number;
  avg_cs: number;
  avg_deaths: number;
  win_rate: number;
  last_updated?: string;
}

export interface PlayerMatch {
  match_id: string;
  tournament: string;
  opponent: string;
  champion: string;
  kills: number;
  deaths: number;
  assists: number;
  cs: number;
  gold: number;
  damage_dealt: number;
  vision_score: number;
  map_number: number;
  side: string;
  result: string;
  match_date: string;
}

export interface ApiError {
  error: string;
  message: string;
} 