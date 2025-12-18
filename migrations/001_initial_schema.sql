-- Omnivore Initial Schema
-- Run with: psql $DATABASE_URL -f migrations/001_initial_schema.sql

BEGIN;

-- Instruments registry
CREATE TABLE IF NOT EXISTS instruments (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(20) UNIQUE NOT NULL,
    name            VARCHAR(255),
    asset_type      VARCHAR(20) NOT NULL CHECK (asset_type IN ('etf', 'stock')),
    exchange        VARCHAR(20),
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_instruments_symbol ON instruments(symbol);
CREATE INDEX idx_instruments_active ON instruments(is_active) WHERE is_active = true;

-- Raw OHLCV price data
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    id              SERIAL PRIMARY KEY,
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    date            DATE NOT NULL,
    open            NUMERIC(12,4),
    high            NUMERIC(12,4),
    low             NUMERIC(12,4),
    close           NUMERIC(12,4),
    adj_close       NUMERIC(12,4),
    volume          BIGINT,
    source          VARCHAR(50) DEFAULT 'yfinance',
    fetched_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(instrument_id, date)
);

CREATE INDEX idx_ohlcv_instrument_date ON ohlcv_daily(instrument_id, date DESC);

-- Computed features (wide format)
CREATE TABLE IF NOT EXISTS features_daily (
    id              SERIAL PRIMARY KEY,
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    date            DATE NOT NULL,
    -- Technical indicators
    rsi_14          NUMERIC(8,4),
    ma_10           NUMERIC(12,4),
    ma_20           NUMERIC(12,4),
    ma_50           NUMERIC(12,4),
    atr_14          NUMERIC(12,4),
    -- Derived signals
    ma_cross_10_50  NUMERIC(8,4),
    rsi_oversold    BOOLEAN,
    rsi_overbought  BOOLEAN,
    -- Target variables
    return_1d       NUMERIC(10,6),
    return_5d       NUMERIC(10,6),
    -- Extensibility for experimental features
    extra_features  JSONB DEFAULT '{}',
    computed_at     TIMESTAMPTZ DEFAULT now(),
    UNIQUE(instrument_id, date)
);

CREATE INDEX idx_features_instrument_date ON features_daily(instrument_id, date DESC);

-- Model definitions
CREATE TABLE IF NOT EXISTS models (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) UNIQUE NOT NULL,
    description     TEXT,
    target          VARCHAR(50) NOT NULL,
    model_type      VARCHAR(50) NOT NULL,
    feature_config  JSONB NOT NULL,
    hyperparameters JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Trained model versions
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    model_id        INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    training_start  DATE NOT NULL,
    training_end    DATE NOT NULL,
    metrics         JSONB NOT NULL,
    artifact_path   VARCHAR(500) NOT NULL,
    is_active       BOOLEAN DEFAULT false,
    trained_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(model_id, version)
);

CREATE INDEX idx_model_versions_active ON model_versions(model_id, is_active) WHERE is_active = true;

-- Ensure only one active version per model
CREATE UNIQUE INDEX idx_one_active_version ON model_versions(model_id) WHERE is_active = true;

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    instrument_id   INTEGER NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_date     DATE NOT NULL,
    horizon         VARCHAR(10) NOT NULL CHECK (horizon IN ('1d', '5d')),
    predicted_value NUMERIC(10,6),
    confidence      NUMERIC(6,4),
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_predictions_lookup ON predictions(instrument_id, target_date, horizon);
CREATE INDEX idx_predictions_model_version ON predictions(model_version_id);

-- Actual outcomes for measuring accuracy
CREATE TABLE IF NOT EXISTS prediction_actuals (
    id              SERIAL PRIMARY KEY,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE UNIQUE,
    actual_value    NUMERIC(10,6),
    error           NUMERIC(10,6),
    absolute_error  NUMERIC(10,6),
    direction_correct BOOLEAN,
    recorded_at     TIMESTAMPTZ DEFAULT now()
);

-- Drift monitoring reports
CREATE TABLE IF NOT EXISTS drift_reports (
    id              SERIAL PRIMARY KEY,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    report_date     DATE NOT NULL,
    drift_type      VARCHAR(50) NOT NULL CHECK (drift_type IN ('feature', 'prediction', 'performance')),
    metrics         JSONB NOT NULL,
    is_alert        BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_drift_reports_model ON drift_reports(model_version_id, report_date DESC);

-- Job tracking (optional, RQ handles most of this)
CREATE TABLE IF NOT EXISTS job_history (
    id              SERIAL PRIMARY KEY,
    job_id          VARCHAR(100) NOT NULL,
    job_type        VARCHAR(50) NOT NULL,
    status          VARCHAR(20) NOT NULL,
    params          JSONB,
    result          JSONB,
    error           TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_job_history_type_status ON job_history(job_type, status);

COMMIT;
