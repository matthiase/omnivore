# ============================================================================
# FILE: src/omnivore/repositories/prediction_repository.py
# ============================================================================
from typing import List, Optional
from omnivore import db


class PredictionRepository:
    """
    Repository for prediction-related data access.
    Encapsulates all SQL and queries for the predictions and prediction_actuals tables.
    """

    def get_latest_predictions(self, horizon: str = "1d") -> List[dict]:
        """
        Get the latest prediction for each instrument for a given horizon.
        Returns a list of dicts with instrument_id, predicted_value, target_date, symbol, and name.
        """
        return db.fetch_all(
            """
            SELECT p.instrument_id, p.predicted_value, p.target_date, i.symbol, i.name
            FROM predictions p
            JOIN instruments i ON p.instrument_id = i.id
            WHERE p.horizon = %s
            AND p.target_date = (
                SELECT MAX(p2.target_date)
                FROM predictions p2
                WHERE p2.instrument_id = p.instrument_id
                  AND p2.horizon = %s
            )
            ORDER BY i.symbol
            """,
            (horizon, horizon)
        )

    def get_accuracy_summary(self, horizon: str = "1d") -> List[dict]:
        """
        Get accuracy summary for each instrument for a given horizon.
        Returns a list of dicts with instrument_id, symbol, n_predictions, directional_accuracy, and mae.
        """
        return db.fetch_all(
            """
            SELECT
                i.id as instrument_id,
                i.symbol,
                COUNT(a.id) as n_predictions,
                AVG(a.direction_correct::int) as directional_accuracy,
                AVG(a.absolute_error) as mae
            FROM instruments i
            LEFT JOIN predictions p ON p.instrument_id = i.id
            LEFT JOIN prediction_actuals a ON a.prediction_id = p.id
            WHERE i.is_active = true
              AND p.horizon = %s
            GROUP BY i.id, i.symbol
            ORDER BY i.symbol
            """,
            (horizon,)
        )

    def create_prediction(
        self,
        model_version_id: int,
        instrument_id: int,
        prediction_date,
        target_date,
        horizon: str,
        predicted_value: float,
        confidence: float = None,
    ) -> dict:
        """
        Insert a new prediction and return the inserted row.
        """
        return db.fetch_one(
            """
            INSERT INTO predictions (
                model_version_id, instrument_id, prediction_date, target_date,
                horizon, predicted_value, confidence
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                model_version_id,
                instrument_id,
                prediction_date,
                target_date,
                horizon,
                predicted_value,
                confidence,
            )
        )

    def get_by_id(self, prediction_id: int) -> Optional[dict]:
        """
        Get a prediction by its ID.
        """
        return db.fetch_one(
            "SELECT * FROM predictions WHERE id = %s",
            (prediction_id,)
        )

    def get_actuals_for_prediction(self, prediction_id: int) -> Optional[dict]:
        """
        Get the actuals record for a given prediction.
        """
        return db.fetch_one(
            "SELECT * FROM prediction_actuals WHERE prediction_id = %s",
            (prediction_id,)
        )