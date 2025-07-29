import React, { useState, useEffect, useRef } from 'react';
import { Search, ChevronDown, X } from 'lucide-react';

interface PlayerSearchProps {
  onPlayerSelect: (playerName: string) => void;
  selectedPlayer?: string;
  placeholder?: string;
}

interface Player {
  name: string;
  team?: string;
  league?: string;
}

const PlayerSearch: React.FC<PlayerSearchProps> = ({ 
  onPlayerSelect, 
  selectedPlayer = '', 
  placeholder = "Search for a player..." 
}) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Search for players when query changes
  useEffect(() => {
    const searchPlayers = async () => {
      if (query.length < 2) {
        setSuggestions([]);
        setIsOpen(false);
        return;
      }

      setIsLoading(true);
      setError('');

      try {
        const response = await fetch(
          `${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/v1/players/search?query=${encodeURIComponent(query)}&limit=10`
        );
        
        if (response.ok) {
          const data = await response.json();
          setSuggestions(data.players || []);
          setIsOpen(data.players && data.players.length > 0);
        } else {
          setError('Failed to search players');
          setSuggestions([]);
          setIsOpen(false);
        }
      } catch (err) {
        setError('Error searching players');
        setSuggestions([]);
        setIsOpen(false);
      } finally {
        setIsLoading(false);
      }
    };

    const debounceTimer = setTimeout(searchPlayers, 300);
    return () => clearTimeout(debounceTimer);
  }, [query]);

  const handlePlayerSelect = (playerName: string) => {
    setQuery(playerName);
    setIsOpen(false);
    onPlayerSelect(playerName);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
    if (e.target.value === '') {
      onPlayerSelect('');
    }
  };

  const handleClear = () => {
    setQuery('');
    setIsOpen(false);
    onPlayerSelect('');
    inputRef.current?.focus();
  };

  return (
    <div className="relative w-full" ref={dropdownRef}>
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => query.length >= 2 && suggestions.length > 0 && setIsOpen(true)}
          placeholder={placeholder}
          className="block w-full pl-10 pr-10 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
        />
        <div className="absolute inset-y-0 right-0 flex items-center">
          {query && (
            <button
              onClick={handleClear}
              className="p-1 mr-2 text-gray-400 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
          )}
          <ChevronDown className="h-5 w-5 text-gray-400 mr-3" />
        </div>
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
          {isLoading ? (
            <div className="px-4 py-2 text-sm text-gray-500">Searching...</div>
          ) : suggestions.length > 0 ? (
            suggestions.map((player, index) => (
              <button
                key={index}
                onClick={() => handlePlayerSelect(player)}
                className="block w-full text-left px-4 py-2 text-sm text-gray-900 hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
              >
                {player}
              </button>
            ))
          ) : query.length >= 2 ? (
            <div className="px-4 py-2 text-sm text-gray-500">No players found</div>
          ) : null}
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="mt-1 text-sm text-red-600">{error}</div>
      )}

      {/* Selected player info */}
      {selectedPlayer && (
        <div className="mt-2 text-sm text-green-600">
          ✓ Selected: {selectedPlayer}
        </div>
      )}
    </div>
  );
};

export default PlayerSearch; 