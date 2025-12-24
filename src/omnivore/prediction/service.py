from datetime import date, timedelta

from omnivore import db
from omnivore.model.registry import ModelRegistry
from omnivore.services.feature_engine import FeatureEngine


class PredictionService:
    """Handles prediction generation and tracking."""

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.model_registry = ModelRegistry()

    def generate_prediction(
        self,
        model_id: int,
        instrument_id: int,
        prediction_date: date,
        horizon: str = "1d",
    ) -> dict:
        """Generate a prediction for a specific date and horizon."""

        # Get active model version
        version = self.model_registry.get_active_version(model_id)
        if not version:
            raise ValueError(f"No active version for model {model_id}")

        # Get model definition for target info
        self.model_registry.get_model(model_id)

        # Determine target date based on horizon
        days = int(horizon.replace("d", ""))
        target_date = prediction_date + timedelta(days=days)

        # Get features for prediction date
        features = self.feature_engine.get_features(
            instrument_id=instrument_id,
            start_date=prediction_date,
            end_date=prediction_date,
            include_targets=False,
        )

        if features.empty:
            raise ValueError(f"No features available for {prediction_date}")

        # Generate prediction
        predicted_value = self.model_registry.predict(
            model_version_id=version["id"],
            features=features,
        )[0]

        # Store prediction
        prediction = db.fetch_one(
            """
            INSERT INTO predictions
                (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                version["id"],
                instrument_id,
                prediction_date,
                target_date,
                horizon,
                float(predicted_value),
            ),
        )

        return {
            "prediction": prediction,
            "model_version": version["version"],
            "predicted_return": float(predicted_value),
            "direction": "up" if predicted_value > 0 else "down",
        }

    def generate_predictions_batch(
        self,
        model_id: int,
        instrument_ids: list[int],
        prediction_date: date,
        horizons: list[str] = None,
    ) -> list[dict]:
        """Generate predictions for multiple instruments and horizons."""
        horizons = horizons or ["1d"]
        results = []

        for instrument_id in instrument_ids:
            for horizon in horizons:
                try:
                    result = self.generate_prediction(
                        model_id=model_id,
                        instrument_id=instrument_id,
                        prediction_date=prediction_date,
                        horizon=horizon,
                    )
                    results.append(result)
                except Exception as e:
                    results.append(
                        {
                            "instrument_id": instrument_id,
                            "horizon": horizon,
                            "error": str(e),
                        }
                    )

        return results

    def record_actual(self, prediction_id: int, actual_value: float) -> dict:
        """Record the actual outcome for a prediction."""
        prediction = db.fetch_one("SELECT * FROM predictions WHERE id = %s", (prediction_id,))

        if not prediction:
            raise ValueError(f"Prediction {prediction_id} not found")

        predicted_value = float(prediction["predicted_value"])
        error = predicted_value - actual_value
        direction_correct = (predicted_value > 0) == (actual_value > 0)

        return db.fetch_one(
            """
            INSERT INTO prediction_actuals
                (prediction_id, actual_value, error, absolute_error, direction_correct)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (prediction_id) DO UPDATE SET
                actual_value = EXCLUDED.actual_value,
                error = EXCLUDED.error,
                absolute_error = EXCLUDED.absolute_error,
                direction_correct = EXCLUDED.direction_correct,
                recorded_at = now()
            RETURNING *
            """,
            (prediction_id, actual_value, error, abs(error), direction_correct),
        )

    def backfill_actuals(self, instrument_id: int) -> dict:
        """Backfill actual outcomes for past predictions using OHLCV data."""
        # Get predictions without actuals
        predictions = db.fetch_all(
            """
            SELECT p.*, i.symbol FROM predictions p
            LEFT JOIN instruments i ON i.id = p.instrument_id
            LEFT JOIN prediction_actuals pa ON pa.prediction_id = p.id
            WHERE p.instrument_id = %s
              AND pa.id IS NULL
              AND p.target_date <= CURRENT_DATE
            ORDER BY p.target_date
            """,
            (instrument_id,),
        )

        if not predictions:
            return {"updated": 0, "message": "No predictions to backfill"}

        # Fetch the OHLCV data from the upstream provider
        from omnivore.ohlcv import OhlcvService

        ohlcv = OhlcvService()

        updated = 0
        for pred in predictions:
            # Get price data for prediction_date and target_date
            df = ohlcv.fetch(
                symbol=pred["symbol"],
                start_date=pred["prediction_date"],
                end_date=pred["target_date"],
            )

            if len(df) < 2:
                continue

            start_price = df.iloc[0]["adj_close"]
            end_price = df.iloc[-1]["adj_close"]
            actual_return = (end_price - start_price) / start_price

            self.record_actual(pred["id"], float(actual_return))
            updated += 1

        return {"updated": updated, "total_pending": len(predictions)}

    def get_predictions(
        self,
        instrument_id: int = None,
        model_id: int = None,
        start_date: date = None,
        end_date: date = None,
        horizon: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query predictions with various filters."""
        query = """
            SELECT
                p.*,
                pa.actual_value,
                pa.error,
                pa.direction_correct,
                mv.version as model_version,
                m.name as model_name,
                i.symbol
            FROM predictions p
            JOIN model_versions mv ON mv.id = p.model_version_id
            JOIN models m ON m.id = mv.model_id
            JOIN instruments i ON i.id = p.instrument_id
            LEFT JOIN prediction_actuals pa ON pa.prediction_id = p.id
            WHERE 1=1
        """
        params = []

        if instrument_id:
            query += " AND p.instrument_id = %s"
            params.append(instrument_id)
        if model_id:
            query += " AND m.id = %s"
            params.append(model_id)
        if start_date:
            query += " AND p.prediction_date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND p.prediction_date <= %s"
            params.append(end_date)
        if horizon:
            query += " AND p.horizon = %s"
            params.append(horizon)

        query += " ORDER BY p.prediction_date DESC LIMIT %s"
        params.append(limit)

        return db.fetch_all(query, tuple(params))

    def get_performance_summary(
        self,
        model_id: int,
        instrument_id: int = None,
        horizon: str = None,
    ) -> dict:
        """Get performance summary for a model's predictions."""
        query = """
            SELECT
                COUNT(*) as total_predictions,
                COUNT(pa.id) as predictions_with_actuals,
                AVG(pa.absolute_error) as mae,
                AVG(pa.error) as mean_error,
                AVG(CASE WHEN pa.direction_correct THEN 1.0 ELSE 0.0 END) as directional_accuracy,
                SUM(CASE WHEN pa.direction_correct THEN 1 ELSE 0 END) as correct_directions,
                SUM(CASE WHEN NOT pa.direction_correct THEN 1 ELSE 0 END) as incorrect_directions
            FROM predictions p
            JOIN model_versions mv ON mv.id = p.model_version_id
            LEFT JOIN prediction_actuals pa ON pa.prediction_id = p.id
            WHERE mv.model_id = %s
        """
        params = [model_id]

        if instrument_id:
            query += " AND p.instrument_id = %s"
            params.append(instrument_id)
        if horizon:
            query += " AND p.horizon = %s"
            params.append(horizon)

        return db.fetch_one(query, tuple(params))
