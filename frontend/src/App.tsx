import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [availableTeams, setAvailableTeams] = useState<string[]>([]);

  // Load available teams on component mount
  useEffect(() => {
    fetchAvailableTeams();
  }, []);

  const fetchAvailableTeams = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/teams');
      if (response.ok) {
        const data = await response.json();
        setAvailableTeams(data.teams || []);
      }
    } catch (err) {
      console.error('Error fetching teams:', err);
      // Fallback to common teams if API fails
      setAvailableTeams(['T1', 'Gen.G', 'JDG', 'BLG', 'TES', 'G2', 'FNC', 'C9', 'TL']);
    }
  };

  // Player search functionality
  const searchPlayers = async (query: string) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/players/search?query=${encodeURIComponent(query)}&limit=10`
      );
      
      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.players || []);
      } else {
        setSearchResults([]);
      }
    } catch (err) {
      console.error('Error searching players:', err);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    // Debounce search
    const timeoutId = setTimeout(() => searchPlayers(query), 300);
    return () => clearTimeout(timeoutId);
  };

  const handlePlayerSelect = (playerName: string) => {
    setSelectedPlayer(playerName);
    setSearchQuery(playerName);
    setSearchResults([]);
  };

  // Prediction functionality
  const handlePredictionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedPlayer) {
      alert('Please select a player first');
      return;
    }

    const formData = new FormData(e.target as HTMLFormElement);
    const startMap = parseInt(formData.get('start_map') as string);
    const endMap = parseInt(formData.get('end_map') as string);
    
    // Create map range array
    const mapRange = [];
    for (let i = startMap; i <= endMap; i++) {
      mapRange.push(i);
    }

    const predictionData = {
      player_name: selectedPlayer,
      prop_type: formData.get('prop_type') as string,
      prop_value: parseFloat(formData.get('prop_value') as string),
      opponent: formData.get('opponent') as string,
      map_range: mapRange,
      start_map: startMap,
      end_map: endMap
    };

    setIsSubmitting(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionData)
      });

      if (response.ok) {
        const result = await response.json();
        setPrediction(result);
        // Trigger the slide animation after a brief delay
        setTimeout(() => setShowResults(true), 100);
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail?.message || 'Failed to make prediction'}`);
      }
    } catch (error) {
      alert('Network error. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

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

  const handleNewPrediction = () => {
    setShowResults(false);
    setPrediction(null);
    setSelectedPlayer('');
    setSearchQuery('');
    setTimeout(() => {
      // Reset form
      const form = document.querySelector('form') as HTMLFormElement;
      if (form) form.reset();
    }, 300);
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Image */}
      <div 
        className="fixed inset-0 z-0"
        style={{
          backgroundImage: 'url(./background.jpeg)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          filter: 'blur(8px) brightness(0.7)',
          transform: 'scale(1.1)'
        }}
      />
      
      {/* Content Overlay */}
      <div className="relative z-10 min-h-screen bg-black bg-opacity-30">
        <div className="container mx-auto px-4 py-8">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4 drop-shadow-lg">
              Outscaled.GG
            </h1>
            <p className="text-xl text-gray-200 drop-shadow-md">
              League of Legends Prop Predictions
            </p>
          </div>

          {/* Main Content */}
          <div className={`transition-all duration-700 ease-in-out ${
            showResults ? 'grid grid-cols-1 lg:grid-cols-2 gap-8' : 'flex justify-center'
          }`}>
            
            {/* Prediction Form */}
            <div className={`transition-all duration-700 ease-in-out ${
              showResults ? 'lg:col-span-1' : 'w-full max-w-2xl'
            }`}>
              <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-white/20">
                <div className="text-center mb-8">
                  <h2 className="text-3xl font-bold text-gray-900 mb-2">
                    Make a Prediction
                  </h2>
                  <p className="text-gray-600">
                    Select a player and enter prop details to get predictions
                  </p>
                </div>

                <form onSubmit={handlePredictionSubmit} className="space-y-6">
                  {/* Player Search */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Player Name
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={handleSearchChange}
                        placeholder="Search for a player..."
                        className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg"
                      />
                      {isSearching && (
                        <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                        </div>
                      )}
                    </div>
                    
                    {/* Search Results */}
                    {searchResults.length > 0 && (
                      <div className="mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                        {searchResults.map((player, index) => (
                          <button
                            key={index}
                            type="button"
                            onClick={() => handlePlayerSelect(player)}
                            className="w-full text-left px-4 py-3 hover:bg-gray-50 focus:bg-gray-50 focus:outline-none border-b border-gray-100 last:border-b-0"
                          >
                            {player}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Selected Player Info */}
                    {selectedPlayer && (
                      <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-xl text-sm text-green-800">
                        Selected: {selectedPlayer}
                      </div>
                    )}
                  </div>

                  {/* Prop Type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Prop Type
                    </label>
                    <select name="prop_type" className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg">
                      <option value="kills">Kills</option>
                      <option value="assists">Assists</option>
                      <option value="cs">CS (Creep Score)</option>
                      <option value="deaths">Deaths</option>
                      <option value="gold">Gold</option>
                      <option value="damage">Damage</option>
                    </select>
                  </div>

                  {/* Prop Value */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Prop Value
                    </label>
                    <input
                      type="number"
                      name="prop_value"
                      step="0.5"
                      defaultValue="4.5"
                      className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg"
                      placeholder="4.5"
                    />
                  </div>

                  {/* Opponent Team */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Opponent Team
                    </label>
                    <select name="opponent" className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg">
                      <option value="">Select opponent team...</option>
                      {availableTeams.map((team, index) => (
                        <option key={index} value={team}>{team}</option>
                      ))}
                    </select>
                  </div>

                  {/* Map Range */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Map Range
                    </label>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label htmlFor="start_map" className="block text-xs font-medium text-gray-500">Start Map</label>
                        <select name="start_map" id="start_map" className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg">
                          <option value="1">Map 1</option>
                          <option value="2">Map 2</option>
                          <option value="3">Map 3</option>
                          <option value="4">Map 4</option>
                          <option value="5">Map 5</option>
                        </select>
                      </div>
                      <div>
                        <label htmlFor="end_map" className="block text-xs font-medium text-gray-500">End Map</label>
                        <select name="end_map" id="end_map" className="w-full px-4 py-4 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white/90 backdrop-blur-sm text-lg">
                          <option value="1">Map 1</option>
                          <option value="2">Map 2</option>
                          <option value="3">Map 3</option>
                          <option value="4">Map 4</option>
                          <option value="5">Map 5</option>
                        </select>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      💡 Tip: PrizePicks often uses "Maps 1-2" for total kills across first two maps
                    </p>
                  </div>

                  {/* Submit Button */}
                  <button
                    type="submit"
                    disabled={!selectedPlayer || isSubmitting}
                    className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-4 px-6 rounded-xl hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-105 text-lg font-semibold shadow-lg"
                  >
                    {isSubmitting ? (
                      <div className="flex items-center justify-center">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                        Making Prediction...
                      </div>
                    ) : (
                      'Make Prediction'
                    )}
                  </button>
                </form>
              </div>
            </div>

            {/* Prediction Results */}
            {prediction && (
              <div className={`transition-all duration-700 ease-in-out ${
                showResults ? 'lg:col-span-1 opacity-100 translate-x-0' : 'opacity-0 translate-x-full'
              }`}>
                <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-white/20">
                  <div className="flex justify-between items-center mb-6">
                    <h2 className="text-3xl font-bold text-gray-900">
                      Prediction Results
                    </h2>
                    <button
                      onClick={handleNewPrediction}
                      className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                    >
                      New Prediction
                    </button>
                  </div>
                  
                  <div className="space-y-8">
                    {/* Main Prediction */}
                    <div className="text-center">
                      <div className={`inline-flex items-center px-6 py-3 rounded-full text-2xl font-bold ${prediction.prediction === 'MORE' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        {prediction.prediction === 'MORE' ? 'OVER' : 'UNDER'}
                      </div>
                      <p className="text-lg text-gray-600 mt-3">
                        {prediction.prop_request?.player_name || selectedPlayer} {prediction.prop_request?.prop_type || 'kills'} {prediction.prop_request?.prop_value || '0'}
                      </p>
                      {prediction.prop_request?.map_range && prediction.prop_request.map_range.length > 1 && (
                        <p className="text-sm text-gray-500 mt-1">
                          Maps {prediction.prop_request.map_range[0]}-{prediction.prop_request.map_range[prediction.prop_request.map_range.length - 1]}
                        </p>
                      )}
                    </div>

                    {/* Confidence */}
                    <div className="text-center">
                      <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-medium ${getConfidenceColor(prediction.confidence)}`}>
                        {getConfidenceLabel(prediction.confidence)} ({prediction.confidence.toFixed(1)}%)
                      </div>
                    </div>

                    {/* Reasoning */}
                    <div className="bg-gray-50/80 backdrop-blur-sm p-6 rounded-xl">
                      <h3 className="font-semibold text-gray-800 mb-3 text-lg">Analysis</h3>
                      <p className="text-gray-700 leading-relaxed">{prediction.reasoning}</p>
                    </div>

                    {/* Player Stats Summary */}
                    {prediction.player_stats && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-blue-600">{prediction.player_stats.avg_kills?.toFixed(1) || '0.0'}</div>
                          <div className="text-sm text-blue-600">Avg Kills</div>
                        </div>
                        <div className="bg-green-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-green-600">{prediction.player_stats.avg_assists?.toFixed(1) || '0.0'}</div>
                          <div className="text-sm text-green-600">Avg Assists</div>
                        </div>
                        <div className="bg-purple-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-purple-600">{prediction.player_stats.avg_cs?.toFixed(0) || '0'}</div>
                          <div className="text-sm text-purple-600">Avg CS</div>
                        </div>
                        <div className="bg-orange-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-orange-600">{prediction.player_stats.win_rate?.toFixed(1) || '0.0'}%</div>
                          <div className="text-sm text-orange-600">Win Rate</div>
                        </div>
                      </div>
                    )}

                    {/* Recent Matches */}
                    {prediction.player_stats?.recent_matches && prediction.player_stats.recent_matches.length > 0 && (
                      <div className="bg-gray-50/80 backdrop-blur-sm p-6 rounded-xl">
                        <h3 className="font-semibold text-gray-800 mb-4 text-lg">Recent Performance</h3>
                        <div className="space-y-3 max-h-40 overflow-y-auto">
                          {prediction.player_stats.recent_matches.slice(0, 5).map((match: any, index: number) => (
                            <div key={index} className="flex justify-between items-center p-3 bg-white rounded-lg">
                              <div className="flex items-center space-x-3">
                                <span className="text-sm font-medium text-gray-600">
                                  {match.champion || 'Unknown'}
                                </span>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                  match.result === 1 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                }`}>
                                  {match.result === 1 ? 'W' : 'L'}
                                </span>
                              </div>
                              <div className="text-sm text-gray-600">
                                {match.kills}/{match.deaths}/{match.assists}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Data Source Info */}
                    <div className="text-center text-sm text-gray-500">
                      <p>Prediction based on {prediction.data_source || 'model analysis'}</p>
                      {prediction.prediction_time_ms && (
                        <p>Generated in {(prediction.prediction_time_ms / 1000).toFixed(3)}s</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 