import React, { useState } from 'react';
import { PropRequest, PredictionResponse } from '../types';
import { apiService } from '../services/api';
import { TrendingUp, TrendingDown, Loader2, Target, Users, Trophy } from 'lucide-react';

const PropPredictor: React.FC = () => {
  const [formData, setFormData] = useState<PropRequest>({
    player_name: '',
    prop_type: 'kills',
    prop_value: 0,
    opponent: '',
    tournament: '',
    map_number: 1,
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const propTypes = [
    { value: 'kills', label: 'Kills' },
    { value: 'assists', label: 'Assists' },
    { value: 'cs', label: 'CS' },
    { value: 'deaths', label: 'Deaths' },
    { value: 'gold', label: 'Gold' },
    { value: 'damage', label: 'Damage' },
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await apiService.predictProp(formData);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: keyof PropRequest, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-success-600';
    if (confidence >= 0.6) return 'text-warning-600';
    return 'text-gray-600';
  };

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence';
    if (confidence >= 0.6) return 'Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Outscaled.GG
        </h1>
        <p className="text-xl text-gray-600">
          League of Legends Prop Prediction
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Prediction Form */}
        <div className="card">
          <h2 className="text-2xl font-semibold mb-6 flex items-center">
            <Target className="mr-2" />
            Prop Prediction
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Player Name
              </label>
              <input
                type="text"
                value={formData.player_name}
                onChange={(e) => handleInputChange('player_name', e.target.value)}
                className="input-field"
                placeholder="e.g., Faker, Chovy, Knight"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prop Type
              </label>
              <select
                value={formData.prop_type}
                onChange={(e) => handleInputChange('prop_type', e.target.value)}
                className="input-field"
              >
                {propTypes.map(type => (
                  <option key={type.value} value={type.value}>
                    {type.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Prop Value
              </label>
              <input
                type="number"
                step="0.5"
                value={formData.prop_value}
                onChange={(e) => handleInputChange('prop_value', parseFloat(e.target.value))}
                className="input-field"
                placeholder="e.g., 3.5"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Opponent (Optional)
              </label>
              <input
                type="text"
                value={formData.opponent}
                onChange={(e) => handleInputChange('opponent', e.target.value)}
                className="input-field"
                placeholder="e.g., T1, Gen.G"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tournament (Optional)
              </label>
              <input
                type="text"
                value={formData.tournament}
                onChange={(e) => handleInputChange('tournament', e.target.value)}
                className="input-field"
                placeholder="e.g., LCK Spring 2024"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Map Number
              </label>
              <input
                type="number"
                min="1"
                max="5"
                value={formData.map_number}
                onChange={(e) => handleInputChange('map_number', parseInt(e.target.value))}
                className="input-field"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin mr-2" />
                  Predicting...
                </>
              ) : (
                'Get Prediction'
              )}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-100 border border-red-300 text-red-700 rounded-md">
              {error}
            </div>
          )}
        </div>

        {/* Prediction Results */}
        <div className="card">
          <h2 className="text-2xl font-semibold mb-6 flex items-center">
            <Trophy className="mr-2" />
            Prediction Results
          </h2>

          {prediction ? (
            <div className="space-y-6">
              {/* Main Prediction */}
              <div className={`p-4 rounded-lg border-2 ${
                prediction.prediction === 'more' 
                  ? 'prediction-more' 
                  : 'prediction-less'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg font-semibold">
                    {prediction.prediction === 'more' ? 'MORE' : 'LESS'}
                  </span>
                  <div className="flex items-center">
                    {prediction.prediction === 'more' ? (
                      <TrendingUp className="w-5 h-5 mr-1" />
                    ) : (
                      <TrendingDown className="w-5 h-5 mr-1" />
                    )}
                    <span className={`font-semibold ${getConfidenceColor(prediction.confidence)}`}>
                      {Math.round(prediction.confidence * 100)}%
                    </span>
                  </div>
                </div>
                <p className="text-sm opacity-75">
                  {getConfidenceText(prediction.confidence)}
                </p>
              </div>

              {/* Prop Details */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="font-semibold mb-2">Prop Details</h3>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-600">Player:</span>
                    <span className="ml-2 font-medium">{prediction.player_name}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Type:</span>
                    <span className="ml-2 font-medium capitalize">{prediction.prop_type}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Value:</span>
                    <span className="ml-2 font-medium">{prediction.prop_value}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Confidence:</span>
                    <span className="ml-2 font-medium">{Math.round(prediction.confidence * 100)}%</span>
                  </div>
                </div>
              </div>

              {/* Reasoning */}
              <div>
                <h3 className="font-semibold mb-2">Analysis</h3>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {prediction.reasoning}
                </p>
              </div>

              {/* Player Stats Summary */}
              {prediction.recent_stats && (
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold mb-2 flex items-center">
                    <Users className="mr-2" />
                    Player Stats
                  </h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-600">Avg Kills:</span>
                      <span className="ml-2 font-medium">{prediction.recent_stats.avg_kills.toFixed(1)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Avg Assists:</span>
                      <span className="ml-2 font-medium">{prediction.recent_stats.avg_assists.toFixed(1)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Avg CS:</span>
                      <span className="ml-2 font-medium">{prediction.recent_stats.avg_cs.toFixed(1)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Win Rate:</span>
                      <span className="ml-2 font-medium">{(prediction.recent_stats.win_rate * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Enter a prop above to get your prediction</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PropPredictor; 