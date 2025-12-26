"""
Integration tests for ModelRegistry.

Run with: OMNIVORE_ENV=test pytest src/omnivore/model/registry_test.py -v
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from omnivore.model.registry import ModelRegistry


@pytest.fixture
def registry(db_connection, tmp_path):
    """Create a ModelRegistry with temporary storage."""
    from omnivore import config as cfg

    original_path = cfg.config.model_storage_path
    cfg.config.model_storage_path = tmp_path / "models"

    yield ModelRegistry()

    cfg.config.model_storage_path = original_path


@pytest.fixture
def sample_training_data(db_connection, sample_instrument):
    """Create sufficient training data (features) for model training."""
    from datetime import timedelta

    base_date = date(2023, 1, 1)

    with db_connection.cursor() as cur:
        for i in range(100):  # Need at least 50 rows
            current_date = base_date + timedelta(days=i)
            # Skip weekends
            if current_date.weekday() >= 5:
                continue

            cur.execute(
                """
                INSERT INTO features_daily
                    (instrument_id, date, rsi_14, ma_10, ma_20, ma_50, atr_14, return_1d, return_5d)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sample_instrument["id"],
                    current_date,
                    50.0 + (i % 30),
                    100.0 + i * 0.5,
                    100.0 + i * 0.45,
                    100.0 + i * 0.4,
                    1.5 + (i % 10) * 0.1,
                    0.01 * (1 if i % 2 == 0 else -1),
                    0.02 * (1 if i % 3 == 0 else -1),
                ),
            )

    return {
        "instrument": sample_instrument,
        "start_date": base_date,
        "end_date": base_date + timedelta(days=100),
    }


class TestCreateModel:
    """Tests for ModelRegistry.create_model()"""

    @pytest.mark.parametrize("model_type", ["ridge", "xgboost", "lightgbm"])
    def test_create_model_success(self, registry, model_type):
        result = registry.create_model(
            name=f"test_{model_type}",
            description="Test model",
            target="return_1d",
            model_type=model_type,
            feature_config=["rsi_14", "ma_10", "ma_20"],
            hyperparameters={"alpha": 1.0} if model_type == "ridge" else None,
        )

        assert result["id"] is not None
        assert result["name"] == f"test_{model_type}"
        assert result["model_type"] == model_type

    def test_create_model_invalid_type_raises(self, registry):
        with pytest.raises(ValueError, match="Unknown model type"):
            registry.create_model(
                name="invalid",
                description="Test",
                target="return_1d",
                model_type="invalid_type",
                feature_config=["rsi_14"],
            )


class TestTrain:
    """Tests for ModelRegistry.train()"""

    @pytest.mark.parametrize("model_type", ["ridge", "xgboost", "lightgbm"])
    def test_train_success(self, registry, sample_training_data, model_type):
        # Create model
        model = registry.create_model(
            name=f"train_test_{model_type}",
            description="Test",
            target="return_1d",
            model_type=model_type,
            feature_config=["rsi_14", "ma_10", "ma_20"],
        )

        result = registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
            test_size=0.2,
        )

        assert result["model_id"] == model["id"]
        assert result["version"] == 1
        assert "metrics" in result
        assert "train" in result["metrics"]
        assert "test" in result["metrics"]
        assert result["metrics"]["train"]["rmse"] >= 0
        assert result["metrics"]["test"]["rmse"] >= 0

    def test_train_increments_version(self, registry, sample_training_data):
        model = registry.create_model(
            name="version_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14", "ma_10"],
        )

        result1 = registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )
        result2 = registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )

        assert result1["version"] == 1
        assert result2["version"] == 2

    def test_train_model_not_found_raises(self, registry, sample_training_data):
        with pytest.raises(ValueError, match="not found"):
            registry.train(
                model_id=99999,
                instrument_id=sample_training_data["instrument"]["id"],
                training_start=sample_training_data["start_date"],
                training_end=sample_training_data["end_date"],
            )

    def test_train_insufficient_data_raises(self, registry, sample_instrument):
        model = registry.create_model(
            name="insufficient_data_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14"],
        )

        with pytest.raises(ValueError, match="Insufficient training data"):
            registry.train(
                model_id=model["id"],
                instrument_id=sample_instrument["id"],
                training_start=date(2023, 1, 1),
                training_end=date(2023, 1, 10),  # Not enough data
            )


class TestActivateVersion:
    """Tests for ModelRegistry.activate_version()"""

    def test_activate_version_success(self, registry, sample_training_data):
        model = registry.create_model(
            name="activate_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14", "ma_10"],
        )

        registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )

        result = registry.activate_version(model["id"], version=1)

        assert result["is_active"] is True
        assert result["version"] == 1

    def test_activate_version_deactivates_others(self, registry, sample_training_data):
        model = registry.create_model(
            name="deactivate_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14", "ma_10"],
        )

        # Train two versions
        registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )
        registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )

        # Activate v1, then v2
        registry.activate_version(model["id"], version=1)
        registry.activate_version(model["id"], version=2)

        # Check only v2 is active
        v1 = registry.get_version(model["id"], 1)
        v2 = registry.get_version(model["id"], 2)

        assert v1["is_active"] is False
        assert v2["is_active"] is True

    def test_activate_version_not_found_raises(self, registry, sample_model):
        with pytest.raises(ValueError, match="not found"):
            registry.activate_version(sample_model["id"], version=99)


class TestPredict:
    """Tests for ModelRegistry.predict()"""

    def test_predict_success(self, registry, sample_training_data):
        model = registry.create_model(
            name="predict_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14", "ma_10", "ma_20"],
        )

        train_result = registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )

        # Create test features
        features = pd.DataFrame(
            [
                {"rsi_14": 55.0, "ma_10": 105.0, "ma_20": 103.0},
                {"rsi_14": 60.0, "ma_10": 110.0, "ma_20": 108.0},
            ]
        )

        predictions = registry.predict(
            model_version_id=train_result["version_record"]["id"],
            features=features,
        )

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2

    def test_predict_version_not_found_raises(self, registry):
        features = pd.DataFrame([{"rsi_14": 55.0, "ma_10": 105.0}])

        with pytest.raises(ValueError, match="not found"):
            registry.predict(model_version_id=99999, features=features)


class TestCompareVersions:
    """Tests for ModelRegistry.compare_versions()"""

    def test_compare_versions_success(self, registry, sample_training_data):
        model = registry.create_model(
            name="compare_test",
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=["rsi_14", "ma_10"],
        )

        # Train two versions
        registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )
        registry.train(
            model_id=model["id"],
            instrument_id=sample_training_data["instrument"]["id"],
            training_start=sample_training_data["start_date"],
            training_end=sample_training_data["end_date"],
        )

        result = registry.compare_versions(model["id"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "version" in result.columns
        assert "test_rmse" in result.columns
        assert "test_dir_acc" in result.columns

    def test_compare_versions_empty(self, registry, sample_model):
        result = registry.compare_versions(sample_model["id"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
