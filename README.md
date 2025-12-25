# Omnivore

Quantitative research and trading platform

## Quick Start Guide

This guide walks you through setting up Omnivore and generating your first prediction.

### Prerequisites

- Python 3.13+
- PostgreSQL 14+
- Redis 7+

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:matthiase/omnivore.git 
cd omnivore

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
### 2. Database Setup

```bash
# Create the PostgreSQL user and database
sudo -u postgres createuser omnivore --createdb
sudo -u postgres createdb omnivore --owner=omnivore

# Run migrations
psql postgresql://omnivore:@localhost:5432/omnivore -f migrations/001_initial_schema.sql

```
### 3. Configuration
Create a `.env` with your configuration. For example:

```env
OMNIVORE_ENV=development
DATABASE_URL=postgresql://omnivore:@localhost:5432/omnivore
REDIS_URL=redis://localhost:6379/0
MODEL_STORAGE_PATH=./models
FEATURES_CONFIG_PATH=./config/features.json
```

### 4. Initialize Data

```bash
# Seed initial instruments (SPY, QQQ, IWM, DIA)
python scripts/seed_instruments.py

# Backfill 5 years of historical data and compute features
python scripts/backfill_historical.py
```

This will take a few minutes as it downloads data from Yahoo Finance.

### 5. Start Services

You'll need three terminal windows:

**Terminal 1 - Redis:**
```bash
redis-server
```

**Terminal 2 - RQ Worker:**
```bash
source .venv/bin/activate
rq worker default training
```

**Terminal 3 - Flask API:**
```bash
source .venv/bin/activate
flask --app src.omnivore.app run --debug
```

The API is now running at `http://localhost:5000`.

### 6. Verify Setup

```bash
# Check API health
curl http://localhost:5000/api/health

# List instruments
curl http://localhost:5000/api/instruments
```

You should see the four seeded ETFs.

---

## Running Tests

Create a `.env.test` with your configuration. For example:

```env
OMNIVORE_ENV=test
DATABASE_URL=postgresql://omnivore:@localhost:5432/omnivore_test
REDIS_URL=redis://localhost:6379/1
MODEL_STORAGE_PATH=./models_test
FEATURES_CONFIG_PATH=./config/features.json
```

```bash
OMNIVORE_ENV=test python -m pytest
```

---

## Creating Your First Model

### Step 1: Define a Model

Create an XGBoost model that predicts next-day returns using RSI and moving averages:

```bash
curl -X POST http://localhost:5000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "spy_daily_xgb_v1",
    "description": "XGBoost model for SPY next-day return prediction",
    "target": "return_1d",
    "model_type": "xgboost",
    "feature_config": ["rsi_14", "ma_10", "ma_20", "ma_50", "atr_14"],
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 4,
      "learning_rate": 0.1
    }
  }'
```

Response:
```json
{
  "id": 1,
  "name": "spy_daily_xgb_v1",
  "target": "return_1d",
  "model_type": "xgboost",
  ...
}
```

### Step 2: Train the Model

Train using the last 3 years of SPY data (instrument_id=1):

```bash
curl -X POST http://localhost:5000/api/models/1/train \
  -H "Content-Type: application/json" \
  -d '{
    "instrument_id": 1,
    "training_start": "2022-01-01",
    "training_end": "2024-12-31",
    "test_size": 0.2
  }'
```

Response:
```json
{
  "job_id": "abc123...",
  "status": "queued"
}
```

### Step 3: Check Training Status

Poll the job until it completes:

```bash
curl http://localhost:5000/api/jobs/abc123...
```

When finished:
```json
{
  "job_id": "abc123...",
  "status": "finished",
  "result": {
    "model_id": 1,
    "version": 1,
    "metrics": {
      "test": {
        "rmse": 0.0142,
        "mae": 0.0098,
        "r2": 0.023,
        "directional_accuracy": 0.54
      }
    }
  }
}
```

### Step 4: Activate the Model Version

```bash
curl -X POST http://localhost:5000/api/models/1/versions/1/activate
```

### Step 5: Generate a Prediction

Generate a prediction for today:

```bash
curl -X POST http://localhost:5000/api/predictions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "instrument_ids": [1],
    "horizons": ["1d"]
  }'
```

Check the job result to see the prediction:

```json
{
  "prediction": {
    "id": 1,
    "predicted_value": 0.0023,
    "target_date": "2025-01-02",
    "horizon": "1d"
  },
  "direction": "up",
  "predicted_return": 0.0023
}
```

---

## Creating a 5-Day Model (Weekly Predictions)

For Saturday predictions of the upcoming week:

```bash
curl -X POST http://localhost:5000/api/models \
  -H "Content-Type: application/json" \
  -d '{
    "name": "spy_weekly_lgbm_v1",
    "description": "LightGBM model for SPY 5-day return prediction",
    "target": "return_5d",
    "model_type": "lightgbm",
    "feature_config": ["rsi_14", "ma_10", "ma_20", "ma_50", "atr_14", "ma_cross_10_50"],
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 5,
      "learning_rate": 0.05
    }
  }'
```

Then train and activate as shown above.

---

## Comparing Models

To compare different model versions or approaches:

```bash
# List all versions of a model
curl http://localhost:5000/api/models/1/versions

# Compare metrics across versions
curl http://localhost:5000/api/models/1/compare
```

Response:
```json
[
  {
    "version": 1,
    "is_active": false,
    "test_rmse": 0.0142,
    "test_dir_acc": 0.54
  },
  {
    "version": 2,
    "is_active": true,
    "test_rmse": 0.0138,
    "test_dir_acc": 0.56
  }
]
```

---

## Tracking Prediction Accuracy

### Backfill Actuals

After predictions have resolved (the target date has passed), record actual outcomes:

```bash
curl -X POST http://localhost:5000/api/predictions/backfill-actuals \
  -H "Content-Type: application/json" \
  -d '{"instrument_id": 1}'
```

### View Performance Summary

```bash
curl "http://localhost:5000/api/predictions/performance?model_id=1&horizon=1d"
```

Response:
```json
{
  "total_predictions": 45,
  "predictions_with_actuals": 42,
  "mae": 0.0105,
  "directional_accuracy": 0.55,
  "correct_directions": 23,
  "incorrect_directions": 19
}
```

---

## Monitoring for Drift

Analyze whether features or prediction accuracy have drifted:

```bash
# Get model version ID first
curl http://localhost:5000/api/models/1/versions

# Trigger drift analysis (via direct Python for now)
python -c "
from omnivore.jobs import analyze_drift_job
result = analyze_drift_job(
    model_version_id=1,
    instrument_id=1,
    reference_days=90,
    current_days=30
)
print(result)
"
```

Query drift reports:

```bash
curl "http://localhost:5000/api/jobs/drift/reports?model_version_id=1"
```

---

## Daily Operations

### Refresh Data (run daily after market close)

```bash
# Refresh SPY data
curl -X POST http://localhost:5000/api/instruments/1/refresh \
  -H "Content-Type: application/json" \
  -d '{}'

# Recompute features
curl -X POST http://localhost:5000/api/instruments/1/features \
  -H "Content-Type: application/json" \
  -d '{}'

# Generate daily predictions
curl -X POST http://localhost:5000/api/predictions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "instrument_ids": [1],
    "horizons": ["1d"]
  }'
```

### Generate Daily Predictions

```bash
curl -X POST http://localhost:5000/api/predictions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "instrument_ids": [1],
    "horizons": ["1d"]
  }'
```

### Suggested Cron Schedule

```cron
# Refresh data daily at 6 PM ET (after market close)
0 18 * * 1-5 curl -X POST http://localhost:5000/api/instruments/1/refresh

# Compute features at 6:15 PM ET
15 18 * * 1-5 curl -X POST http://localhost:5000/api/instruments/1/features

# Generate predictions at 6:30 PM ET
30 18 * * 1-5 curl -X POST http://localhost:5000/api/predictions/generate -H "Content-Type: application/json" -d '{"model_id":1,"instrument_ids":[1],"horizons":["1d"]}'

# Saturday 5-day predictions at 10 AM ET
0 10 * * 6 curl -X POST http://localhost:5000/api/predictions/generate -H "Content-Type: application/json" -d '{"model_id":2,"instrument_ids":[1],"horizons":["5d"]}'

# Backfill actuals daily at 7 PM ET
0 19 * * 1-5 curl -X POST http://localhost:5000/api/predictions/backfill-actuals -H "Content-Type: application/json" -d '{"instrument_id":1}'
```

---

## Troubleshooting

### "No features available" error when predicting

Features need to be computed after data is refreshed:

```bash
curl -X POST http://localhost:5000/api/instruments/1/features
```

### "Insufficient training data" error

Ensure you have enough historical data. The model requires at least 50 rows after dropping NaN values (indicators need lookback periods):

```bash
python scripts/backfill_historical.py
```

### Job stuck in "queued" status

Make sure the RQ worker is running:

```bash
rq worker default training
```

### Check RQ queue status

```bash
rq info
```

---

## Next Steps

1. **Experiment with features** — Edit `config/features.json` to add new indicators
2. **Try different models** — Create models with `ridge`, `xgboost`, or `lightgbm`
3. **Add more instruments** — Use the POST `/api/instruments` endpoint
4. **Set up alerting** — Monitor drift reports for `is_alert: true`
5. **Automate with cron** — Use the suggested schedule above
