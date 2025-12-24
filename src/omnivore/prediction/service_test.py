"""
Integration tests for PredictionService.

Run with: OMNIVORE_ENV=test pytest src/omnivore/prediction/service_test.py -v
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from omnivore.prediction.service import PredictionService


@pytest.fixture
def sample_model_version(db_connection, sample_model) -> dict:
    """Create a model version for testing predictions."""
    from psycopg.rows import dict_row

    with db_connection.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO model_versions
                (model_id, version, training_start, training_end, metrics, artifact_path, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                sample_model["id"],
                1,
                date(2023, 1, 1),
                date(2023, 12, 31),
                '{"test_rmse": 0.01}',
                "/models/test/1/model.joblib",
                True,
            ),
        )
        return cur.fetchone()


@pytest.fixture
def sample_features(db_connection, sample_instrument) -> dict:
    """Create sample features for an instrument."""
    from psycopg.rows import dict_row

    feature_date = date(2024, 1, 15)

    with db_connection.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO features_daily
                (instrument_id, date, rsi_14, ma_10, ma_20, ma_50, atr_14, return_1d, return_5d)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                sample_instrument["id"],
                feature_date,
                55.0,
                105.0,
                103.0,
                100.0,
                1.5,
                0.005,
                0.02,
            ),
        )
        return {"instrument": sample_instrument, "date": feature_date, "features": cur.fetchone()}


@pytest.fixture
def sample_prediction(db_connection, sample_instrument, sample_model_version) -> dict:
    """Create a sample prediction."""
    from psycopg.rows import dict_row

    with db_connection.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO predictions
                (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                sample_model_version["id"],
                sample_instrument["id"],
                date(2024, 1, 15),
                date(2024, 1, 16),
                "1d",
                0.0123,
            ),
        )
        return cur.fetchone()


class TestGeneratePrediction:
    """Tests for PredictionService.generate_prediction()"""

    @pytest.mark.parametrize(
        "horizon,expected_days",
        [
            ("1d", 1),
            ("5d", 5),
        ],
    )
    def test_generate_prediction_success(
        self,
        db_connection,
        sample_instrument,
        sample_model_version,
        sample_features,
        horizon,
        expected_days,
    ):
        service = PredictionService()
        prediction_date = sample_features["date"]

        with patch.object(service.model_registry, "predict", return_value=np.array([0.0150])):
            result = service.generate_prediction(
                model_id=sample_model_version["model_id"],
                instrument_id=sample_instrument["id"],
                prediction_date=prediction_date,
                horizon=horizon,
            )

        assert result["prediction"] is not None
        assert result["prediction"]["horizon"] == horizon
        assert result["prediction"]["prediction_date"] == prediction_date
        assert result["prediction"]["target_date"] == date(2024, 1, 15 + expected_days)
        assert result["predicted_return"] == pytest.approx(0.0150)
        assert result["direction"] == "up"
        assert result["model_version"] == sample_model_version["version"]

    def test_generate_prediction_negative_return(
        self, db_connection, sample_instrument, sample_model_version, sample_features
    ):
        service = PredictionService()

        with patch.object(service.model_registry, "predict", return_value=np.array([-0.0100])):
            result = service.generate_prediction(
                model_id=sample_model_version["model_id"],
                instrument_id=sample_instrument["id"],
                prediction_date=sample_features["date"],
                horizon="1d",
            )

        assert result["predicted_return"] == pytest.approx(-0.0100)
        assert result["direction"] == "down"

    def test_generate_prediction_no_active_version_raises(
        self, db_connection, sample_model, sample_instrument, sample_features
    ):
        service = PredictionService()
        # sample_model exists but has no active version

        with pytest.raises(ValueError, match="No active version"):
            service.generate_prediction(
                model_id=sample_model["id"],
                instrument_id=sample_instrument["id"],
                prediction_date=sample_features["date"],
                horizon="1d",
            )

    def test_generate_prediction_no_features_raises(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()
        # No features exist for this date

        with pytest.raises(ValueError, match="No features available"):
            service.generate_prediction(
                model_id=sample_model_version["model_id"],
                instrument_id=sample_instrument["id"],
                prediction_date=date(2024, 6, 1),  # Date with no features
                horizon="1d",
            )

    def test_generate_prediction_stores_in_database(
        self, db_connection, sample_instrument, sample_model_version, sample_features
    ):
        service = PredictionService()

        with patch.object(service.model_registry, "predict", return_value=np.array([0.02])):
            result = service.generate_prediction(
                model_id=sample_model_version["model_id"],
                instrument_id=sample_instrument["id"],
                prediction_date=sample_features["date"],
                horizon="1d",
            )

        # Verify stored in database
        from omnivore.prediction import PredictionRepository

        repo = PredictionRepository()
        stored = repo.get_by_id(result["prediction"]["id"])

        assert stored is not None
        assert float(stored["predicted_value"]) == pytest.approx(0.02)


class TestGeneratePredictionsBatch:
    """Tests for PredictionService.generate_predictions_batch()"""

    def test_generate_predictions_batch_success(
        self, db_connection, sample_instruments, sample_model_version
    ):
        service = PredictionService()
        prediction_date = date(2024, 1, 15)

        # Create features for all instruments
        with db_connection.cursor() as cur:
            for instrument in sample_instruments:
                cur.execute(
                    """
                    INSERT INTO features_daily
                        (instrument_id, date, rsi_14, ma_10, ma_20, ma_50, atr_14)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (instrument["id"], prediction_date, 50.0, 100.0, 100.0, 100.0, 1.0),
                )

        with patch.object(service.model_registry, "predict", return_value=np.array([0.01])):
            results = service.generate_predictions_batch(
                model_id=sample_model_version["model_id"],
                instrument_ids=[i["id"] for i in sample_instruments],
                prediction_date=prediction_date,
                horizons=["1d"],
            )

        assert len(results) == len(sample_instruments)
        assert all("prediction" in r for r in results)

    @pytest.mark.parametrize(
        "horizons,expected_count_per_instrument",
        [
            (["1d"], 1),
            (["1d", "5d"], 2),
            (None, 1),  # Default to ["1d"]
        ],
    )
    def test_generate_predictions_batch_multiple_horizons(
        self,
        db_connection,
        sample_instrument,
        sample_model_version,
        sample_features,
        horizons,
        expected_count_per_instrument,
    ):
        service = PredictionService()

        with patch.object(service.model_registry, "predict", return_value=np.array([0.01])):
            results = service.generate_predictions_batch(
                model_id=sample_model_version["model_id"],
                instrument_ids=[sample_instrument["id"]],
                prediction_date=sample_features["date"],
                horizons=horizons,
            )

        assert len(results) == expected_count_per_instrument

    def test_generate_predictions_batch_handles_errors(
        self, db_connection, sample_instruments, sample_model_version
    ):
        service = PredictionService()
        prediction_date = date(2024, 1, 15)

        # Only create features for first instrument
        with db_connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO features_daily
                    (instrument_id, date, rsi_14, ma_10, ma_20, ma_50, atr_14)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (sample_instruments[0]["id"], prediction_date, 50.0, 100.0, 100.0, 100.0, 1.0),
            )

        with patch.object(service.model_registry, "predict", return_value=np.array([0.01])):
            results = service.generate_predictions_batch(
                model_id=sample_model_version["model_id"],
                instrument_ids=[i["id"] for i in sample_instruments],
                prediction_date=prediction_date,
                horizons=["1d"],
            )

        # First should succeed, others should have errors
        assert "prediction" in results[0]
        assert all("error" in r for r in results[1:])


class TestRecordActual:
    """Tests for PredictionService.record_actual()"""

    @pytest.mark.parametrize(
        "predicted,actual,expected_direction_correct",
        [
            (0.0123, 0.0100, True),  # Both positive
            (-0.0123, -0.0100, True),  # Both negative
            (0.0123, -0.0100, False),  # Opposite signs
            (-0.0123, 0.0100, False),  # Opposite signs
        ],
    )
    def test_record_actual_success(
        self,
        db_connection,
        sample_instrument,
        sample_model_version,
        predicted,
        actual,
        expected_direction_correct,
    ):
        service = PredictionService()

        # Create prediction with specific predicted value
        from psycopg.rows import dict_row

        with db_connection.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 15),
                    date(2024, 1, 16),
                    "1d",
                    predicted,
                ),
            )
            prediction = cur.fetchone()

        result = service.record_actual(prediction["id"], actual)

        assert result is not None
        assert result["prediction_id"] == prediction["id"]
        assert float(result["actual_value"]) == pytest.approx(actual)
        assert float(result["error"]) == pytest.approx(predicted - actual)
        assert float(result["absolute_error"]) == pytest.approx(abs(predicted - actual))
        assert result["direction_correct"] is expected_direction_correct

    def test_record_actual_updates_existing(self, db_connection, sample_prediction):
        service = PredictionService()

        # Record first actual
        service.record_actual(sample_prediction["id"], 0.0100)

        # Update with new actual
        result = service.record_actual(sample_prediction["id"], 0.0200)

        assert float(result["actual_value"]) == pytest.approx(0.0200)

    def test_record_actual_prediction_not_found_raises(self, db_connection):
        service = PredictionService()

        with pytest.raises(ValueError, match="not found"):
            service.record_actual(99999, 0.01)


class TestBackfillActuals:
    """Tests for PredictionService.backfill_actuals()"""

    def test_backfill_actuals_success(self, db_connection, sample_instrument, sample_model_version):
        service = PredictionService()

        # Create a prediction that needs backfilling
        from psycopg.rows import dict_row

        with db_connection.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    "1d",
                    0.01,
                ),
            )
            prediction = cur.fetchone()

        # Mock yfinance to return price data
        mock_yf_data = pd.DataFrame(
            [
                {
                    "Date": "2024-01-01",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.0,
                    "Volume": 1000000,
                },
                {
                    "Date": "2024-01-02",
                    "Open": 101.0,
                    "High": 102.0,
                    "Low": 100.0,
                    "Close": 102.0,
                    "Volume": 1100000,
                },
            ]
        )
        mock_yf_data["Date"] = pd.to_datetime(mock_yf_data["Date"])
        mock_yf_data = mock_yf_data.set_index("Date")

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            result = service.backfill_actuals(sample_instrument["id"])

        assert result["updated"] == 1
        assert result["total_pending"] == 1

        # Verify actual was recorded
        from omnivore.prediction import PredictionRepository

        repo = PredictionRepository()
        actual = repo.get_actuals_for_prediction(prediction["id"])
        assert actual is not None

    def test_backfill_actuals_no_predictions(self, db_connection, sample_instrument):
        service = PredictionService()

        result = service.backfill_actuals(sample_instrument["id"])

        assert result["updated"] == 0
        assert result["message"] == "No predictions to backfill"

    def test_backfill_actuals_skips_insufficient_data(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create a prediction
        with db_connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    "1d",
                    0.01,
                ),
            )

        # Mock yfinance to return only 1 row (need 2)
        mock_yf_data = pd.DataFrame(
            [
                {
                    "Date": "2024-01-01",
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.0,
                    "Volume": 1000000,
                },
            ]
        )
        mock_yf_data["Date"] = pd.to_datetime(mock_yf_data["Date"])
        mock_yf_data = mock_yf_data.set_index("Date")

        with patch("omnivore.ohlcv.service.yf.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_yf_data
            mock_ticker_class.return_value = mock_ticker

            result = service.backfill_actuals(sample_instrument["id"])

        assert result["updated"] == 0
        assert result["total_pending"] == 1


class TestGetPredictions:
    """Tests for PredictionService.get_predictions()"""

    def test_get_predictions_no_filters(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create some predictions
        with db_connection.cursor() as cur:
            for i in range(3):
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sample_model_version["id"],
                        sample_instrument["id"],
                        date(2024, 1, i + 1),
                        date(2024, 1, i + 2),
                        "1d",
                        0.01,
                    ),
                )

        result = service.get_predictions()

        assert len(result) == 3
        assert "symbol" in result[0]
        assert "model_name" in result[0]
        assert "model_version" in result[0]

    @pytest.mark.parametrize(
        "filter_key,filter_value,expected_count",
        [
            ("horizon", "1d", 2),
            ("horizon", "5d", 1),
        ],
    )
    def test_get_predictions_with_horizon_filter(
        self,
        db_connection,
        sample_instrument,
        sample_model_version,
        filter_key,
        filter_value,
        expected_count,
    ):
        service = PredictionService()

        # Create predictions with different horizons
        with db_connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                VALUES
                    (%s, %s, %s, %s, '1d', %s),
                    (%s, %s, %s, %s, '1d', %s),
                    (%s, %s, %s, %s, '5d', %s)
                """,
                (
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    0.01,
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    0.01,
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 1),
                    date(2024, 1, 6),
                    0.05,
                ),
            )

        result = service.get_predictions(**{filter_key: filter_value})

        assert len(result) == expected_count

    def test_get_predictions_with_date_range(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create predictions across different dates
        with db_connection.cursor() as cur:
            for i in range(10):
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sample_model_version["id"],
                        sample_instrument["id"],
                        date(2024, 1, i + 1),
                        date(2024, 1, i + 2),
                        "1d",
                        0.01,
                    ),
                )

        result = service.get_predictions(start_date=date(2024, 1, 3), end_date=date(2024, 1, 7))

        assert len(result) == 5

    def test_get_predictions_respects_limit(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create more predictions than limit
        with db_connection.cursor() as cur:
            for i in range(10):
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sample_model_version["id"],
                        sample_instrument["id"],
                        date(2024, 1, i + 1),
                        date(2024, 1, i + 2),
                        "1d",
                        0.01,
                    ),
                )

        result = service.get_predictions(limit=5)

        assert len(result) == 5

    def test_get_predictions_includes_actuals(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create prediction with actual
        from psycopg.rows import dict_row

        with db_connection.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    sample_model_version["id"],
                    sample_instrument["id"],
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    "1d",
                    0.01,
                ),
            )
            prediction = cur.fetchone()

            cur.execute(
                """
                INSERT INTO prediction_actuals
                    (prediction_id, actual_value, error, absolute_error, direction_correct)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (prediction["id"], 0.008, 0.002, 0.002, True),
            )

        result = service.get_predictions()

        assert len(result) == 1
        assert float(result[0]["actual_value"]) == pytest.approx(0.008)
        assert result[0]["direction_correct"] is True


class TestGetPerformanceSummary:
    """Tests for PredictionService.get_performance_summary()"""

    def test_get_performance_summary_success(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create predictions with actuals
        predictions_data = [
            {"predicted": 0.01, "actual": 0.008, "direction_correct": True},
            {"predicted": 0.02, "actual": 0.015, "direction_correct": True},
            {"predicted": -0.01, "actual": 0.005, "direction_correct": False},
        ]

        for data in predictions_data:
            from psycopg.rows import dict_row

            with db_connection.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        sample_model_version["id"],
                        sample_instrument["id"],
                        date(2024, 1, 1),
                        date(2024, 1, 2),
                        "1d",
                        data["predicted"],
                    ),
                )
                prediction = cur.fetchone()

                error = data["predicted"] - data["actual"]
                cur.execute(
                    """
                    INSERT INTO prediction_actuals
                        (prediction_id, actual_value, error, absolute_error, direction_correct)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        prediction["id"],
                        data["actual"],
                        error,
                        abs(error),
                        data["direction_correct"],
                    ),
                )

        result = service.get_performance_summary(model_id=sample_model_version["model_id"])

        assert result["total_predictions"] == 3
        assert result["predictions_with_actuals"] == 3
        assert float(result["directional_accuracy"]) == pytest.approx(2 / 3)
        assert result["correct_directions"] == 2
        assert result["incorrect_directions"] == 1

    def test_get_performance_summary_no_predictions(self, db_connection, sample_model_version):
        service = PredictionService()

        result = service.get_performance_summary(model_id=sample_model_version["model_id"])

        assert result["total_predictions"] == 0

    def test_get_performance_summary_filters_by_horizon(
        self, db_connection, sample_instrument, sample_model_version
    ):
        service = PredictionService()

        # Create predictions with different horizons
        for horizon in ["1d", "1d", "5d"]:
            from psycopg.rows import dict_row

            with db_connection.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (model_version_id, instrument_id, prediction_date, target_date, horizon, predicted_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        sample_model_version["id"],
                        sample_instrument["id"],
                        date(2024, 1, 1),
                        date(2024, 1, 2),
                        horizon,
                        0.01,
                    ),
                )
                prediction = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO prediction_actuals
                        (prediction_id, actual_value, error, absolute_error, direction_correct)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (prediction["id"], 0.01, 0.0, 0.0, True),
                )

        result_1d = service.get_performance_summary(
            model_id=sample_model_version["model_id"], horizon="1d"
        )
        result_5d = service.get_performance_summary(
            model_id=sample_model_version["model_id"], horizon="5d"
        )

        assert result_1d["total_predictions"] == 2
        assert result_5d["total_predictions"] == 1
