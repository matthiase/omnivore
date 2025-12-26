"""
Integration tests for PredictionRepository.

Run with: OMNIVORE_ENV=test pytest src/omnivore/prediction/repository_test.py -v
"""

from datetime import date

import pytest

from omnivore.prediction.repository import PredictionRepository


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
def sample_prediction(db_connection, sample_instrument, sample_model_version) -> dict:
    """Create a single prediction for testing."""
    repo = PredictionRepository()
    return repo.create_prediction(
        model_version_id=sample_model_version["id"],
        instrument_id=sample_instrument["id"],
        prediction_date=date(2024, 1, 1),
        target_date=date(2024, 1, 2),
        horizon="1d",
        predicted_value=0.0123,
        confidence=0.85,
    )


@pytest.fixture
def sample_prediction_with_actual(db_connection, sample_prediction) -> dict:
    """Create a prediction with an associated actual outcome."""
    from psycopg.rows import dict_row

    with db_connection.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO prediction_actuals
                (prediction_id, actual_value, error, absolute_error, direction_correct)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                sample_prediction["id"],
                0.0100,  # actual_value
                0.0023,  # error (predicted - actual)
                0.0023,  # absolute_error
                True,  # direction_correct (both positive)
            ),
        )
        actual = cur.fetchone()

    return {"prediction": sample_prediction, "actual": actual}


class TestCreatePrediction:
    """Tests for PredictionRepository.create_prediction()"""

    @pytest.mark.parametrize(
        "horizon,predicted_value,confidence",
        [
            ("1d", 0.0150, 0.90),
            ("5d", -0.0200, 0.75),
            ("1d", 0.0001, None),  # No confidence
        ],
    )
    def test_create_prediction_success(
        self,
        db_connection,
        sample_instrument,
        sample_model_version,
        horizon,
        predicted_value,
        confidence,
    ):
        repo = PredictionRepository()

        result = repo.create_prediction(
            model_version_id=sample_model_version["id"],
            instrument_id=sample_instrument["id"],
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 2),
            horizon=horizon,
            predicted_value=predicted_value,
            confidence=confidence,
        )

        assert result["id"] is not None
        assert result["model_version_id"] == sample_model_version["id"]
        assert result["instrument_id"] == sample_instrument["id"]
        assert result["prediction_date"] == date(2024, 1, 1)
        assert result["target_date"] == date(2024, 1, 2)
        assert result["horizon"] == horizon
        assert float(result["predicted_value"]) == pytest.approx(predicted_value)
        if confidence is not None:
            assert float(result["confidence"]) == pytest.approx(confidence)
        else:
            assert result["confidence"] is None
        assert result["created_at"] is not None

    def test_create_prediction_invalid_model_version_raises(self, db_connection, sample_instrument):
        repo = PredictionRepository()

        with pytest.raises(Exception):  # ForeignKeyViolation
            repo.create_prediction(
                model_version_id=99999,
                instrument_id=sample_instrument["id"],
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                horizon="1d",
                predicted_value=0.01,
            )

    def test_create_prediction_invalid_instrument_raises(self, db_connection, sample_model_version):
        repo = PredictionRepository()

        with pytest.raises(Exception):  # ForeignKeyViolation
            repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=99999,
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                horizon="1d",
                predicted_value=0.01,
            )


class TestGetById:
    """Tests for PredictionRepository.get_by_id()"""

    @pytest.mark.parametrize("horizon", ["1d", "5d"])
    def test_get_by_id_success(
        self, db_connection, sample_instrument, sample_model_version, horizon
    ):
        repo = PredictionRepository()
        created = repo.create_prediction(
            model_version_id=sample_model_version["id"],
            instrument_id=sample_instrument["id"],
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 2),
            horizon=horizon,
            predicted_value=0.01,
        )

        result = repo.get_by_id(created["id"])

        assert result is not None
        assert result["id"] == created["id"]
        assert result["horizon"] == horizon

    def test_get_by_id_not_found(self, db_connection):
        repo = PredictionRepository()

        result = repo.get_by_id(99999)

        assert result is None


class TestGetLatestPredictions:
    """Tests for PredictionRepository.get_latest_predictions()"""

    @pytest.mark.parametrize("horizon", ["1d", "5d"])
    def test_get_latest_predictions_success(
        self, db_connection, sample_instruments, sample_model_version, horizon
    ):
        repo = PredictionRepository()

        # Create predictions for each instrument with different target dates
        for i, instrument in enumerate(sample_instruments):
            # Older prediction
            repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=instrument["id"],
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                horizon=horizon,
                predicted_value=0.01 + i * 0.001,
            )
            # Newer prediction (should be returned)
            repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=instrument["id"],
                prediction_date=date(2024, 1, 5),
                target_date=date(2024, 1, 6),
                horizon=horizon,
                predicted_value=0.02 + i * 0.001,
            )

        result = repo.get_latest_predictions(horizon=horizon)

        assert len(result) == len(sample_instruments)
        # All should have the later target_date
        for r in result:
            assert r["target_date"] == date(2024, 1, 6)

    def test_get_latest_predictions_filters_by_horizon(
        self, db_connection, sample_instrument, sample_model_version
    ):
        repo = PredictionRepository()

        # Create predictions with different horizons
        repo.create_prediction(
            model_version_id=sample_model_version["id"],
            instrument_id=sample_instrument["id"],
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 2),
            horizon="1d",
            predicted_value=0.01,
        )
        repo.create_prediction(
            model_version_id=sample_model_version["id"],
            instrument_id=sample_instrument["id"],
            prediction_date=date(2024, 1, 1),
            target_date=date(2024, 1, 6),
            horizon="5d",
            predicted_value=0.05,
        )

        result_1d = repo.get_latest_predictions(horizon="1d")
        result_5d = repo.get_latest_predictions(horizon="5d")

        assert len(result_1d) == 1
        assert result_1d[0]["target_date"] == date(2024, 1, 2)

        assert len(result_5d) == 1
        assert result_5d[0]["target_date"] == date(2024, 1, 6)

    def test_get_latest_predictions_empty(self, db_connection):
        repo = PredictionRepository()

        result = repo.get_latest_predictions()

        assert result == []

    def test_get_latest_predictions_ordered_by_symbol(
        self, db_connection, sample_instruments, sample_model_version
    ):
        repo = PredictionRepository()

        # Create predictions in reverse symbol order
        for instrument in reversed(sample_instruments):
            repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=instrument["id"],
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                horizon="1d",
                predicted_value=0.01,
            )

        result = repo.get_latest_predictions()

        symbols = [r["symbol"] for r in result]
        assert symbols == sorted(symbols)


class TestGetAccuracySummary:
    """Tests for PredictionRepository.get_accuracy_summary()"""

    def test_get_accuracy_summary_success(
        self, db_connection, sample_instrument, sample_model_version
    ):
        repo = PredictionRepository()

        # Create predictions with actuals
        predictions_data = [
            {"predicted": 0.01, "actual": 0.008, "direction_correct": True},
            {"predicted": 0.02, "actual": 0.015, "direction_correct": True},
            {"predicted": -0.01, "actual": 0.005, "direction_correct": False},
        ]

        for data in predictions_data:
            pred = repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=sample_instrument["id"],
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2),
                horizon="1d",
                predicted_value=data["predicted"],
            )

            with db_connection.cursor() as cur:
                error = data["predicted"] - data["actual"]
                cur.execute(
                    """
                    INSERT INTO prediction_actuals
                        (prediction_id, actual_value, error, absolute_error, direction_correct)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (pred["id"], data["actual"], error, abs(error), data["direction_correct"]),
                )

        result = repo.get_accuracy_summary(horizon="1d")

        assert len(result) == 1
        assert result[0]["instrument_id"] == sample_instrument["id"]
        assert result[0]["n_predictions"] == 3
        assert float(result[0]["directional_accuracy"]) == pytest.approx(2 / 3)

    @pytest.mark.parametrize("horizon", ["1d", "5d"])
    def test_get_accuracy_summary_filters_by_horizon(
        self, db_connection, sample_instrument, sample_model_version, horizon
    ):
        repo = PredictionRepository()

        # Create predictions for both horizons
        for h in ["1d", "5d"]:
            pred = repo.create_prediction(
                model_version_id=sample_model_version["id"],
                instrument_id=sample_instrument["id"],
                prediction_date=date(2024, 1, 1),
                target_date=date(2024, 1, 2) if h == "1d" else date(2024, 1, 6),
                horizon=h,
                predicted_value=0.01,
            )

            with db_connection.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_actuals
                        (prediction_id, actual_value, error, absolute_error, direction_correct)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (pred["id"], 0.01, 0.0, 0.0, True),
                )

        result = repo.get_accuracy_summary(horizon=horizon)

        assert len(result) == 1
        assert result[0]["n_predictions"] == 1

    def test_get_accuracy_summary_empty(self, db_connection):
        repo = PredictionRepository()

        result = repo.get_accuracy_summary()

        assert result == []


class TestGetActualsForPrediction:
    """Tests for PredictionRepository.get_actuals_for_prediction()"""

    def test_get_actuals_for_prediction_success(self, db_connection, sample_prediction_with_actual):
        repo = PredictionRepository()
        prediction = sample_prediction_with_actual["prediction"]

        result = repo.get_actuals_for_prediction(prediction["id"])

        assert result is not None
        assert result["prediction_id"] == prediction["id"]
        assert float(result["actual_value"]) == pytest.approx(0.0100)
        assert float(result["error"]) == pytest.approx(0.0023)
        assert result["direction_correct"] is True

    def test_get_actuals_for_prediction_not_found(self, db_connection, sample_prediction):
        repo = PredictionRepository()

        result = repo.get_actuals_for_prediction(sample_prediction["id"])

        assert result is None

    def test_get_actuals_for_prediction_invalid_id(self, db_connection):
        repo = PredictionRepository()

        result = repo.get_actuals_for_prediction(99999)

        assert result is None
