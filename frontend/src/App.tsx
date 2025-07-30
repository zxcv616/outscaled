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
  const [verboseMode, setVerboseMode] = useState(false);
  
  // NEW: Statistical analysis state
  const [statisticalData, setStatisticalData] = useState<any>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [showStatisticalAnalysis, setShowStatisticalAnalysis] = useState(false);
  const [rangeStd, setRangeStd] = useState(5.0);

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
      end_map: endMap,
      verbose: verboseMode // Include verbose mode
    };

    setIsSubmitting(true);
    try {
      const response = await fetch(`http://localhost:8000/api/v1/predict?verbose=${verboseMode}`, {
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
        setTimeout(() => {
          setShowResults(true);
          // Smooth scroll to results
          setTimeout(() => {
            const resultsElement = document.getElementById('prediction-results');
            if (resultsElement) {
              resultsElement.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
              });
            }
          }, 100);
        }, 100);
        
        // NEW: Fetch statistical analysis in background
        fetchStatisticalAnalysis(predictionData);
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

  // NEW: Statistical analysis functions
  const fetchStatisticalAnalysis = async (predictionData: any) => {
    setIsLoadingStats(true);
    try {
      // Get comprehensive statistical analysis
      const response = await fetch(`http://localhost:8000/api/v1/statistics/comprehensive?range_std=${rangeStd}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionData)
      });

      if (response.ok) {
        const result = await response.json();
        setStatisticalData(result);
        setShowStatisticalAnalysis(true);
      } else {
        const error = await response.json();
        console.error('Statistical analysis error:', error);
        // Don't show alert, just log the error
      }
    } catch (error) {
      console.error('Network error in statistical analysis:', error);
    } finally {
      setIsLoadingStats(false);
    }
  };

  const fetchProbabilityDistribution = async (predictionData: any) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/statistics/probability-distribution?range_std=${rangeStd}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionData)
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Error fetching probability distribution:', error);
    }
    return null;
  };

  const fetchStatisticalInsights = async (predictionData: any) => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/statistics/insights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionData)
      });

      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Error fetching statistical insights:', error);
    }
    return null;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return 'text-blue-600 bg-blue-100';
    if (confidence >= 60) return 'text-blue-600 bg-blue-100';
    return 'text-gray-600 bg-gray-100';
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
    // NEW: Reset statistical analysis data
    setStatisticalData(null);
    setShowStatisticalAnalysis(false);
    setTimeout(() => {
      // Reset form
      const form = document.querySelector('form') as HTMLFormElement;
      if (form) form.reset();
    }, 300);
  };

  const handleCopyAsJson = async () => {
    if (!prediction) return;
    
    try {
      // Create a clean JSON object with all prediction data
      const jsonData = {
        prediction: prediction.prediction,
        confidence: prediction.confidence,
        reasoning: prediction.reasoning,
        prop_request: prediction.prop_request,
        player_stats: prediction.player_stats,
        model_mode: prediction.model_mode,
        rule_override: prediction.rule_override,
        scaler_status: prediction.scaler_status,
        data_source: prediction.data_source,
        prediction_time_ms: prediction.prediction_time_ms,
        generated_at: new Date().toISOString()
      };
      
      const jsonString = JSON.stringify(jsonData, null, 2);
      await navigator.clipboard.writeText(jsonString);
      
      // Show success feedback
      const button = document.getElementById('copy-json-btn');
      if (button) {
        const originalText = button.textContent;
        button.textContent = '✅ Copied!';
        button.classList.add('bg-green-500', 'hover:bg-green-600');
        button.classList.remove('bg-blue-500', 'hover:bg-blue-600');
        
        setTimeout(() => {
          button.textContent = originalText;
          button.classList.remove('bg-green-500', 'hover:bg-green-600');
          button.classList.add('bg-blue-500', 'hover:bg-blue-600');
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to copy JSON:', error);
      alert('Failed to copy JSON to clipboard');
    }
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
          <div className="transition-all duration-700 ease-in-out">
            
            {/* Prediction Form */}
            <div className="w-full max-w-2xl mx-auto">
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

                  {/* Verbose Mode Toggle */}
                  <div className="flex items-center justify-center">
                    <input
                      type="checkbox"
                      id="verbose-toggle"
                      checked={verboseMode}
                      onChange={(e) => setVerboseMode(e.target.checked)}
                      className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="verbose-toggle" className="text-sm text-gray-700">
                      Show Verbose Prediction
                    </label>
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

            {/* Prediction Results - Now appears below */}
            {prediction && (
              <div className={`transition-all duration-700 ease-in-out mt-8 ${
                showResults ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
              }`} id="prediction-results">
                <div className="w-full max-w-6xl mx-auto bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-8 border border-white/20">
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
                      <div className={`inline-flex items-center px-6 py-3 rounded-full text-2xl font-bold ${prediction.prediction === 'MORE' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'}`}>
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

                    {/* Prop Request Details */}
                    {verboseMode && prediction.prop_request && (
                      <div className="bg-gray-50/80 backdrop-blur-sm p-6 rounded-xl">
                        <h3 className="font-semibold text-gray-800 mb-3 text-lg">Prop Request Details</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Prop Type</div>
                            <div className="text-gray-700 capitalize">{prediction.prop_request.prop_type}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Prop Value</div>
                            <div className="text-gray-700">{prediction.prop_request.prop_value}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Map Range</div>
                            <div className="text-gray-700">
                              {prediction.prop_request.map_range?.join('-') || '1'}
                            </div>
                          </div>
                          {prediction.prop_request.opponent && (
                            <div className="text-center">
                              <div className="font-medium text-gray-600">Opponent</div>
                              <div className="text-gray-700">{prediction.prop_request.opponent}</div>
                            </div>
                          )}
                          {prediction.prop_request.tournament && (
                            <div className="text-center">
                              <div className="font-medium text-gray-600">Tournament</div>
                              <div className="text-gray-700">{prediction.prop_request.tournament}</div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

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

                    {/* Model Transparency Section */}
                    <div className="bg-blue-50/80 backdrop-blur-sm p-6 rounded-xl">
                      <h3 className="font-semibold text-gray-800 mb-3 text-lg">Model Information</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div className="text-center">
                          <div className="font-medium text-blue-600">Model Mode</div>
                          <div className="text-gray-700 capitalize">{prediction.model_mode || 'primary'}</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium text-blue-600">Rule Override</div>
                          <div className="text-gray-700">{prediction.rule_override ? 'Yes' : 'No'}</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium text-blue-600">Scaler Status</div>
                          <div className="text-gray-700 capitalize">{prediction.scaler_status || 'unknown'}</div>
                        </div>
                        <div className="text-center">
                          <div className="font-medium text-blue-600">Data Source</div>
                          <div className="text-gray-700 capitalize">{prediction.data_source || 'unknown'}</div>
                        </div>
                      </div>
                      {prediction.rule_override && (
                        <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                          <p className="text-sm text-yellow-800">
                            <strong>Rule Override Applied:</strong> Recent form significantly differs from prop value, 
                            triggering conservative prediction adjustment.
                          </p>
                        </div>
                      )}
                      {prediction.scaler_status === 'missing' && (
                        <div className="mt-3 p-3 bg-orange-50 border border-orange-200 rounded-lg">
                          <p className="text-sm text-orange-800">
                            <strong>Scaler Warning:</strong> Using unscaled features may affect prediction reliability.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Confidence Breakdown */}
                    <div className="bg-blue-50/80 backdrop-blur-sm p-6 rounded-xl">
                      <h3 className="font-semibold text-gray-800 mb-3 text-lg">Confidence Analysis</h3>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-medium text-gray-600">Base Confidence</span>
                          <span className="text-sm text-gray-700">
                            {prediction.confidence.toFixed(1)}%
                          </span>
                        </div>
                        {prediction.player_stats?.matches_in_range && prediction.player_stats.matches_in_range < 3 && (
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-blue-600">Limited Data Penalty</span>
                            <span className="text-sm text-blue-700">
                              {prediction.player_stats.matches_in_range} matches in range
                            </span>
                          </div>
                        )}
                        {prediction.player_stats?.recent_matches && (
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-600">Recent Volatility</span>
                            <span className="text-sm text-gray-700">
                              {(() => {
                                const recentKills = prediction.player_stats.recent_matches.slice(0, 5).map((m: any) => m.kills);
                                const mean = recentKills.reduce((a: number, b: number) => a + b, 0) / recentKills.length;
                                const variance = recentKills.reduce((acc: number, val: number) => acc + Math.pow(val - mean, 2), 0) / recentKills.length;
                                const cv = (Math.sqrt(variance) / mean * 100).toFixed(1);
                                return `${cv}% CV`;
                              })()}
                            </span>
                          </div>
                        )}
                        {prediction.player_stats?.win_rate && (
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-600">Recent Win Rate</span>
                            <span className="text-sm text-gray-700">
                              {(prediction.player_stats.win_rate * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Player Stats Summary */}
                    {prediction.player_stats && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-blue-600">{prediction.player_stats.avg_kills?.toFixed(1) || '0.0'}</div>
                          <div className="text-sm text-blue-600">Avg Kills</div>
                        </div>
                        <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-blue-600">{prediction.player_stats.avg_assists?.toFixed(1) || '0.0'}</div>
                          <div className="text-sm text-blue-600">Avg Assists</div>
                        </div>
                        <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-blue-600">{prediction.player_stats.avg_cs?.toFixed(0) || '0'}</div>
                          <div className="text-sm text-blue-600">Avg CS</div>
                        </div>
                        <div className="bg-blue-50/80 backdrop-blur-sm p-4 rounded-xl text-center">
                          <div className="text-3xl font-bold text-blue-600">{(prediction.player_stats.win_rate * 100)?.toFixed(1) || '0.0'}%</div>
                          <div className="text-sm text-blue-600">Win Rate</div>
                        </div>
                      </div>
                    )}

                    {/* Additional Player Stats */}
                    {prediction.player_stats && (
                      <div className="bg-gray-50/80 backdrop-blur-sm p-6 rounded-xl">
                        <h3 className="font-semibold text-gray-800 mb-3 text-lg">Detailed Statistics</h3>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Recent Kills Avg</div>
                            <div className="text-lg font-bold text-gray-800">{prediction.player_stats.recent_kills_avg?.toFixed(1) || '0.0'}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Recent Assists Avg</div>
                            <div className="text-lg font-bold text-gray-800">{prediction.player_stats.recent_assists_avg?.toFixed(1) || '0.0'}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Recent CS Avg</div>
                            <div className="text-lg font-bold text-gray-800">{prediction.player_stats.recent_cs_avg?.toFixed(0) || '0'}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Avg KDA</div>
                            <div className="text-lg font-bold text-gray-800">{prediction.player_stats.avg_kda?.toFixed(2) || '0.00'}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">Avg GPM</div>
                            <div className="text-lg font-bold text-gray-800">{prediction.player_stats.avg_gpm?.toFixed(0) || '0'}</div>
                          </div>
                          <div className="text-center">
                            <div className="font-medium text-gray-600">KP %</div>
                            <div className="text-lg font-bold text-gray-800">{(prediction.player_stats.avg_kp_percent * 100)?.toFixed(1) || '0.0'}%</div>
                          </div>
                        </div>
                        {prediction.player_stats.data_years && (
                          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                            <p className="text-sm text-blue-800">
                              <strong>Data Coverage:</strong> {prediction.player_stats.data_years}
                            </p>
                          </div>
                        )}
                        {prediction.player_stats.map_range_warning && (
                          <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <p className="text-sm text-yellow-800">
                              <strong>Map Range Note:</strong> {prediction.player_stats.map_range_warning}
                            </p>
                          </div>
                        )}
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
                                  match.win === true ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                                }`}>
                                  {match.win === true ? 'W' : 'L'}
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

                    {/* Verbose Mode Details */}
                    {verboseMode && prediction.features_used && (
                      <div className="bg-blue-50/80 backdrop-blur-sm p-6 rounded-xl">
                        <h3 className="font-semibold text-gray-800 mb-3 text-lg">Model Details</h3>
                        <div className="space-y-3">
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-600">Features Used</span>
                            <span className="text-sm text-gray-700">{prediction.features_used.length} engineered features</span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-gray-600">Model Type</span>
                            <span className="text-sm text-gray-700 capitalize">{prediction.model_mode || 'primary'}</span>
                          </div>
                          {prediction.champion_stats && (
                            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                              <p className="text-sm text-blue-800">
                                <strong>Champion Analysis:</strong> {prediction.champion_stats.unique_champions || 0} unique champions, 
                                {prediction.champion_stats.champion_diversity ? ` ${(prediction.champion_stats.champion_diversity * 100).toFixed(1)}%` : ' 0%'} diversity
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Copy as JSON Button */}
                    <div className="text-center">
                      <button
                        id="copy-json-btn"
                        onClick={handleCopyAsJson}
                        className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
                      >
                        Copy as JSON
                      </button>
                    </div>

                    {/* NEW: Statistical Analysis Section */}
                    {statisticalData && (
                      <div className="bg-gradient-to-r from-blue-50 to-gray-50 backdrop-blur-sm p-6 rounded-xl">
                        <h3 className="font-semibold text-gray-800 mb-3 text-lg">Statistical Analysis</h3>
                        
                        {/* Confidence Difference Explanation */}
                        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                          <p className="text-sm text-blue-800">
                            <strong>Note:</strong> Statistical confidence values shown below are calculated using pure statistical methods 
                            and may differ from the main prediction confidence, which includes additional risk factors (volatility, 
                            rule overrides, data quality penalties).
                          </p>
                        </div>

                        {statisticalData?.summary_stats && (
                          <div className="bg-blue-50/80 backdrop-blur-sm p-6 rounded-xl">
                            <h4 className="font-semibold text-gray-800 mb-3">Summary Statistics</h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                              <div className="text-center">
                                <div className="font-medium text-blue-600">Recent Average</div>
                                <div className="text-lg font-bold text-blue-800">{statisticalData.summary_stats.mean_recent || 'N/A'}</div>
                              </div>
                              <div className="text-center">
                                <div className="font-medium text-blue-600">Std Deviation</div>
                                <div className="text-lg font-bold text-blue-800">{statisticalData.summary_stats.std_recent || 'N/A'}</div>
                              </div>
                              <div className="text-center">
                                <div className="font-medium text-blue-600">Z-Score</div>
                                <div className="text-lg font-bold text-blue-800">{statisticalData.summary_stats.input_z_score || 'N/A'}</div>
                              </div>
                              <div className="text-center">
                                <div className="font-medium text-blue-600">Range Analyzed</div>
                                <div className="text-lg font-bold text-blue-800">{statisticalData.summary_stats.range_analyzed || 'N/A'}</div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Probability Distribution */}
                        {statisticalData.probability_distribution && (
                          <div className="bg-gray-50/80 backdrop-blur-sm p-6 rounded-xl">
                            <h4 className="font-semibold text-gray-800 mb-3">Probability Distribution</h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-60 overflow-y-auto">
                              {Object.entries(statisticalData.probability_distribution)
                                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                                .map(([value, data]: [string, any]) => (
                                  <div key={value} className="bg-white p-3 rounded-lg border">
                                    <div className="text-center">
                                      <div className="font-bold text-lg">{value}</div>
                                      <div className={`text-sm font-medium ${
                                        data.prediction === 'MORE' ? 'text-blue-600' : 'text-gray-600'
                                      }`}>
                                        {data.prediction}
                                      </div>
                                      <div className="text-xs text-gray-600">
                                        {data.confidence}% confidence
                                      </div>
                                      <div className="text-xs text-gray-500 mt-1">
                                        MORE: {data.probability_more}% | LESS: {data.probability_less}%
                                      </div>
                                      {data.statistical_significance === 'high' && (
                                        <div className="text-xs text-blue-600 font-medium mt-1">High Significance</div>
                                      )}
                                    </div>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}

                        {/* Statistical Insights */}
                        {statisticalData?.statistical_insights && (
                          <div className="bg-blue-50/80 backdrop-blur-sm p-6 rounded-xl">
                            <h4 className="font-semibold text-gray-800 mb-3">Statistical Insights</h4>
                            <div className="space-y-4">
                              {/* Z-Score and Significance */}
                              <div className="grid grid-cols-2 gap-4">
                                <div className="bg-white p-3 rounded-lg">
                                  <div className="text-sm font-medium text-gray-600">Z-Score</div>
                                  <div className="text-lg font-bold text-gray-800">
                                    {statisticalData.statistical_insights?.statistical_measures?.z_score || 'N/A'}
                                  </div>
                                </div>
                                <div className="bg-white p-3 rounded-lg">
                                  <div className="text-sm font-medium text-gray-600">Percentile</div>
                                  <div className="text-lg font-bold text-gray-800">
                                    {statisticalData.statistical_insights?.statistical_measures?.percentile || 'N/A'}%
                                  </div>
                                </div>
                              </div>

                              {/* Probability Analysis */}
                              {statisticalData.statistical_insights?.probability_analysis && (
                                <div className="bg-white p-4 rounded-lg">
                                  <div className="text-sm font-medium text-gray-600 mb-2">Probability Analysis</div>
                                  <div className="grid grid-cols-2 gap-4">
                                    <div>
                                      <div className="text-sm text-gray-500">MORE Probability</div>
                                      <div className="text-lg font-bold text-blue-600">
                                        {statisticalData.statistical_insights.probability_analysis.probability_more || 'N/A'}%
                                      </div>
                                    </div>
                                    <div>
                                      <div className="text-sm text-gray-500">LESS Probability</div>
                                      <div className="text-lg font-bold text-gray-600">
                                        {statisticalData.statistical_insights.probability_analysis.probability_less || 'N/A'}%
                                      </div>
                                    </div>
                                  </div>
                                  <div className="mt-2 text-sm text-gray-600">
                                    Recommended: <span className="font-medium">
                                      {statisticalData.statistical_insights.probability_analysis.recommended_prediction || 'N/A'}
                                    </span>
                                  </div>
                                </div>
                              )}

                              {/* Confidence Intervals */}
                              {statisticalData.statistical_insights?.confidence_intervals && (
                                <div className="bg-white p-4 rounded-lg">
                                  <div className="text-sm font-medium text-gray-600 mb-2">95% Confidence Interval</div>
                                  <div className="text-lg font-bold text-gray-800">
                                    [{statisticalData.statistical_insights.confidence_intervals['95_percent']?.lower || 'N/A'}, {statisticalData.statistical_insights.confidence_intervals['95_percent']?.upper || 'N/A'}]
                                  </div>
                                  <div className="text-sm text-gray-500">
                                    Width: {statisticalData.statistical_insights.confidence_intervals['95_percent']?.width || 'N/A'}
                                  </div>
                                </div>
                              )}

                              {/* Volatility Metrics */}
                              {statisticalData.statistical_insights?.volatility_metrics && (
                                <div className="bg-white p-4 rounded-lg">
                                  <div className="text-sm font-medium text-gray-600 mb-2">Volatility Analysis</div>
                                  <div className="text-lg font-bold text-gray-800">
                                    {statisticalData.statistical_insights.volatility_metrics.volatility_percentage || 'N/A'}% CV
                                  </div>
                                  <div className="text-sm text-gray-500">
                                    {statisticalData.statistical_insights.volatility_metrics.high_volatility ? 'High' : 
                                     statisticalData.statistical_insights.volatility_metrics.moderate_volatility ? 'Moderate' : 'Low'} Volatility
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Loading State */}
                        {isLoadingStats && (
                          <div className="text-center py-4">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                            <p className="text-sm text-gray-600 mt-2">Loading statistical analysis...</p>
                          </div>
                        )}
                      </div>
                    )}
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