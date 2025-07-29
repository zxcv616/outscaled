import React from 'react';

interface PredictionResultsProps {
  prediction: any;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ prediction }) => {
  if (!prediction) return null;

  const { prediction: result, confidence, reasoning, player_stats, prop_request } = prediction;

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-green-600 bg-green-100';
    if (confidence >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 80) return 'High Confidence';
    if (confidence >= 60) return 'Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">
        📊 Prediction Results
      </h2>
      
      <div className="space-y-6">
        {/* Main Prediction */}
        <div className="text-center">
          <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold ${result === 'MORE' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {result === 'MORE' ? '📈 OVER' : '📉 UNDER'}
          </div>
          <p className="text-sm text-gray-600 mt-2">
            {prop_request.player_name} {prop_request.prop_type} {prop_request.prop_value}
          </p>
        </div>

        {/* Confidence */}
        <div className="text-center">
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(confidence)}`}>
            {getConfidenceLabel(confidence)} ({confidence.toFixed(1)}%)
          </div>
        </div>

        {/* Reasoning */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Analysis</h3>
          <p className="text-gray-700">{reasoning}</p>
        </div>

        {/* Player Stats Summary */}
        {player_stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-600">{player_stats.avg_kills?.toFixed(1) || '0.0'}</div>
              <div className="text-sm text-blue-600">Avg Kills</div>
            </div>
            <div className="bg-green-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-600">{player_stats.avg_assists?.toFixed(1) || '0.0'}</div>
              <div className="text-sm text-green-600">Avg Assists</div>
            </div>
            <div className="bg-purple-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-600">{player_stats.avg_cs?.toFixed(0) || '0'}</div>
              <div className="text-sm text-purple-600">Avg CS</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg text-center">
              <div className="text-2xl font-bold text-orange-600">{(player_stats.win_rate * 100)?.toFixed(1) || '0.0'}%</div>
              <div className="text-sm text-orange-600">Win Rate</div>
            </div>
          </div>
        )}

        {/* Recent Matches */}
        {player_stats?.recent_matches && player_stats.recent_matches.length > 0 && (
          <div>
            <h3 className="font-semibold text-gray-800 mb-3">Recent Matches</h3>
            <div className="space-y-2 max-h-40 overflow-y-auto">
              {player_stats.recent_matches.slice(0, 5).map((match: any, index: number) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                  <div className="flex items-center space-x-2">
                    <span className={`w-2 h-2 rounded-full ${match.win ? 'bg-green-500' : 'bg-red-500'}`}></span>
                    <span className="text-sm font-medium">{match.champion}</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {match.kills}/{match.deaths}/{match.assists} ({match.cs} CS)
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Data Source */}
        <div className="text-center text-sm text-gray-500">
          Data Source: {player_stats?.data_source || 'Unknown'}
        </div>
      </div>
    </div>
  );
};

export default PredictionResults; 