import axios from 'axios';
import { PropRequest, PredictionResponse, PlayerStats } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Predict a prop
  async predictProp(propRequest: PropRequest): Promise<PredictionResponse> {
    try {
      const response = await api.post('/api/v1/predict', propRequest);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to get prediction');
      }
      throw error;
    }
  },

  // Get player statistics
  async getPlayerStats(playerName: string): Promise<PlayerStats> {
    try {
      const response = await api.get(`/api/v1/playerstats/${playerName}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to get player stats');
      }
      throw error;
    }
  },

  // Get list of players
  async getPlayers(): Promise<string[]> {
    try {
      const response = await api.get('/api/v1/players');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to get players');
      }
      throw error;
    }
  },

  // Get recent matches
  async getRecentMatches(): Promise<any[]> {
    try {
      const response = await api.get('/api/v1/matches');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to get matches');
      }
      throw error;
    }
  },

  // Health check
  async healthCheck(): Promise<{ status: string; model_loaded: boolean }> {
    try {
      const response = await api.get('/api/v1/health');
      return response.data;
    } catch (error) {
      throw new Error('API is not available');
    }
  },
}; 