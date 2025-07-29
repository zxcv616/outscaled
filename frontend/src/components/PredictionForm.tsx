import React, { useState } from 'react';
import PlayerSearch from './PlayerSearch';

interface PredictionFormProps {
  onSubmit: (prediction: any) => void;
}

interface PredictionData {
  player_name: string;
  prop_type: string;
  prop_value: number;
  opponent: string;
  tournament: string;
  map_number: number;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onSubmit }) => {
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [propType, setPropType] = useState('kills');
  const [propValue, setPropValue] = useState(4.5);
  const [opponent, setOpponent] = useState('');
  const [tournament, setTournament] = useState('LCS');
  const [mapNumber, setMapNumber] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const propTypes = [
    { value: 'kills', label: 'Kills' },
    { value: 'assists', label: 'Assists' },
    { value: 'cs', label: 'CS (Creep Score)' },
    { value: 'deaths', label: 'Deaths' },
    { value: 'gold', label: 'Gold' },
    { value: 'damage', label: 'Damage' }
  ];

  const tournaments = [
    { value: 'LCS', label: 'LCS' },
    { value: 'LEC', label: 'LEC' },
    { value: 'LCK', label: 'LCK' },
    { value: 'LPL', label: 'LPL' },
    { value: 'MSI', label: 'MSI' },
    { value: 'Worlds', label: 'Worlds' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedPlayer) {
      alert('Please select a player first');
      return;
    }

    setIsSubmitting(true);
    
    const predictionData: PredictionData = {
      player_name: selectedPlayer,
      prop_type: propType,
      prop_value: propValue,
      opponent,
      tournament,
      map_number: mapNumber
    };

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
        onSubmit(result);
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

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">
        🎯 Make a Prediction
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Player Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Player *
          </label>
          <PlayerSearch
            onPlayerSelect={setSelectedPlayer}
            selectedPlayer={selectedPlayer}
            placeholder="Search for a player..."
          />
        </div>

        {/* Prop Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prop Type
          </label>
          <select
            value={propType}
            onChange={(e) => setPropType(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {propTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        {/* Prop Value */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Prop Value
          </label>
          <input
            type="number"
            step="0.5"
            value={propValue}
            onChange={(e) => setPropValue(parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="4.5"
          />
        </div>

        {/* Opponent */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Opponent Team
          </label>
          <input
            type="text"
            value={opponent}
            onChange={(e) => setOpponent(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="Gen.G"
          />
        </div>

        {/* Tournament */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tournament
          </label>
          <select
            value={tournament}
            onChange={(e) => setTournament(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {tournaments.map(tourney => (
              <option key={tourney.value} value={tourney.value}>
                {tourney.label}
              </option>
            ))}
          </select>
        </div>

        {/* Map Number */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Map Number
          </label>
          <select
            value={mapNumber}
            onChange={(e) => setMapNumber(parseInt(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={1}>Map 1</option>
            <option value={2}>Map 2</option>
            <option value={3}>Map 3</option>
          </select>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!selectedPlayer || isSubmitting}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isSubmitting ? 'Making Prediction...' : 'Make Prediction'}
        </button>
      </form>
    </div>
  );
};

export default PredictionForm; 