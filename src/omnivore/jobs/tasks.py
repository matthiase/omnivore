# ============================================================================
# FILE: src/omnivore/jobs/tasks.py
# ============================================================================
"""RQ task definitions for background job processing."""
from datetime import date, timedelta
from omnivore.services import (
    DataService,
    FeatureEngine,
    ModelRegistry,
    PredictionService,
    DriftMonitor,
)


def refresh_data_job(symbol: str, start_date: str = None, end_date: str = None) -> dict:
    """Fetch and store latest OHLCV data for an instrument."""
    data_service = DataService()

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None

    return data_service.refresh_instrument(symbol, start, end)


def compute_features_job(instrument_id: int, start_date: str = None, end_date: str = None) -> dict:
    """Compute and store features for an instrument."""
    feature_engine = FeatureEngine()

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None

    return feature_engine.compute_and_store(instrument_id, start, end)


def train_model_job(
    model_id: int,
    instrument_id: int,
    training_start: str,
    training_end: str,
    test_size: float = 0.2,
) -> dict:
    """Train a new version of a model."""
    model_registry = ModelRegistry()

    return model_registry.train(
        model_id=model_id,
        instrument_id=instrument_id,
        training_start=date.fromisoformat(training_start),
        training_end=date.fromisoformat(training_end),
        test_size=test_size,
    )


def generate_predictions_job(
    model_id: int,
    instrument_ids: list[int],
    prediction_date: str,
    horizons: list[str] = None,
) -> list[dict]:
    """Generate predictions for multiple instruments."""
    prediction_service = PredictionService()

    return prediction_service.generate_predictions_batch(
        model_id=model_id,
        instrument_ids=instrument_ids,
        prediction_date=date.fromisoformat(prediction_date),
        horizons=horizons,
    )


def backfill_actuals_job(instrument_id: int) -> dict:
    """Backfill actual outcomes for past predictions."""
    prediction_service = PredictionService()
    return prediction_service.backfill_actuals(instrument_id)


def analyze_drift_job(
    model_version_id: int,
    instrument_id: int,
    reference_days: int = 90,
    current_days: int = 30,
) -> dict:
    """Analyze feature and prediction drift."""
    drift_monitor = DriftMonitor()

    today = date.today()
    current_end = today
    current_start = today - timedelta(days=current_days)
    reference_end = current_start - timedelta(days=1)
    reference_start = reference_end - timedelta(days=reference_days)

    # Compute feature drift
    feature_drift = drift_monitor.compute_feature_drift(
        model_version_id=model_version_id,
        instrument_id=instrument_id,
        reference_start=reference_start,
        reference_end=reference_end,
        current_start=current_start,
        current_end=current_end,
    )

    # Store feature drift report
    drift_monitor.create_report(
        model_version_id=model_version_id,
        instrument_id=instrument_id,
        drift_type="feature",
        metrics=feature_drift,
    )

    # Compute prediction drift
    prediction_drift = drift_monitor.compute_prediction_drift(
        model_version_id=model_version_id,
        window_days=current_days,
    )

    # Store prediction drift report
    if prediction_drift.get("status") == "computed":
        drift_monitor.create_report(
            model_version_id=model_version_id,
            instrument_id=instrument_id,
            drift_type="prediction",
            metrics=prediction_drift,
        )

    return {
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
    }
