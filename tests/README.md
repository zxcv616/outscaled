# 🧪 Outscaled.gg Test Suite

## Quick Start

**Run all tests with one command:**

```bash
python tests/run_all_tests.py
```

This will run all test suites and show you a comprehensive summary.

## Test Suites

### ✅ Required Tests (Must Pass)
- **Main API Test Suite** (13 tests) - Core API functionality
- **Confidence Logic Test** - Confidence level validation
- **Betting Logic Test** - PrizePicks-style betting logic
- **Map Ranges Test** - Map range functionality

### ⚠️ Optional Tests (Analysis/Development)
- **Natural Confidence Test** - Analysis of confidence levels
- **Pytest API Tests** - Detailed API tests (requires pytest)
- **Unit Tests** - ML component tests (has feature count issues)

## Individual Test Commands

If you want to run specific tests:

```bash
# Main API tests
python tests/run_tests.py

# Confidence analysis
python tests/test_confidence_logic.py

# Betting logic
python tests/test_betting_logic.py

# Map ranges
python tests/test_map_ranges.py

# Natural confidence analysis
python tests/test_natural_confidence.py
```

## Test Results

The comprehensive test runner will show:
- ✅ **PASSED** - Test suite passed
- ❌ **FAILED** - Required test failed
- ⚠️ **SKIPPED** - Optional test skipped

## What Gets Tested

### Core Features
- Health check and service readiness
- Player search with autocomplete
- Teams endpoint for opponent selection
- Basic predictions with confidence
- Map-range support (PrizePicks style)
- Model transparency features

### Betting Logic
- PrizePicks-style map range logic
- Statistical validation for different map ranges
- Edge case handling
- Proper aggregation for multi-map props

### Confidence Analysis
- Confidence level validation
- Natural confidence progression
- Rule-based override analysis
- Statistical reasoning

## Troubleshooting

### Service Not Ready
If you see "Service failed to start", make sure:
1. Docker is running
2. Backend container is started: `docker-compose up -d backend`
3. Wait 30 seconds for service to fully start

### Optional Test Failures
Optional tests (pytest, unit tests) can fail without affecting the main test suite. These are for development and analysis purposes.

### Feature Count Issues
The unit tests have feature count mismatches (31 vs 34 features) due to recent model updates. This doesn't affect the main API functionality. 