"""
Integration tests for ModelRepository.

Run with: OMNIVORE_ENV=test pytest src/omnivore/model/repository_test.py -v
"""

import json
from datetime import date

import pytest
from psycopg.errors import UniqueViolation

from omnivore.model.repository import ModelRepository


class TestCreate:
    """Tests for ModelRepository.create()"""

    @pytest.mark.parametrize(
        "name,target,model_type",
        [
            ("test_model_1", "return_1d", "xgboost"),
            ("test_model_2", "return_5d", "lightgbm"),
            ("test_model_3", "return_1d", "ridge"),
        ],
    )
    def test_create_success(self, db_connection, name, target, model_type):
        repo = ModelRepository()
        feature_config = json.dumps(["rsi_14", "ma_10", "ma_20"])
        hyperparameters = json.dumps({"n_estimators": 100})

        result = repo.create(
            name=name,
            description="Test model",
            target=target,
            model_type=model_type,
            feature_config=feature_config,
            hyperparameters=hyperparameters,
        )

        assert result["id"] is not None
        assert result["name"] == name
        assert result["target"] == target
        assert result["model_type"] == model_type
        assert result["created_at"] is not None

    def test_create_with_no_hyperparameters(self, db_connection):
        repo = ModelRepository()

        result = repo.create(
            name="minimal_model",
            description="Minimal",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps(["rsi_14"]),
            hyperparameters=None,
        )

        assert result["id"] is not None
        assert result["hyperparameters"] == {}

    def test_create_duplicate_name_raises(self, db_connection):
        repo = ModelRepository()
        feature_config = json.dumps(["rsi_14"])

        repo.create(
            name="duplicate_model",
            description="First",
            target="return_1d",
            model_type="ridge",
            feature_config=feature_config,
        )

        with pytest.raises(UniqueViolation):
            repo.create(
                name="duplicate_model",
                description="Second",
                target="return_1d",
                model_type="ridge",
                feature_config=feature_config,
            )


class TestGetById:
    """Tests for ModelRepository.get_by_id()"""

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "ridge"])
    def test_get_by_id_success(self, db_connection, model_type):
        repo = ModelRepository()
        created = repo.create(
            name=f"model_{model_type}",
            description="Test",
            target="return_1d",
            model_type=model_type,
            feature_config=json.dumps(["rsi_14"]),
        )

        result = repo.get_by_id(created["id"])

        assert result is not None
        assert result["id"] == created["id"]
        assert result["model_type"] == model_type

    def test_get_by_id_not_found(self, db_connection):
        repo = ModelRepository()

        result = repo.get_by_id(99999)

        assert result is None


class TestGetByName:
    """Tests for ModelRepository.get_by_name()"""

    @pytest.mark.parametrize("name", ["my_model", "spy_daily_v1", "test-model-123"])
    def test_get_by_name_success(self, db_connection, name):
        repo = ModelRepository()
        repo.create(
            name=name,
            description="Test",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps(["rsi_14"]),
        )

        result = repo.get_by_name(name)

        assert result is not None
        assert result["name"] == name

    def test_get_by_name_not_found(self, db_connection):
        repo = ModelRepository()

        result = repo.get_by_name("nonexistent_model")

        assert result is None


class TestList:
    """Tests for ModelRepository.list()"""

    @pytest.mark.parametrize("num_models", [1, 3, 5])
    def test_list_success(self, db_connection, num_models):
        repo = ModelRepository()

        for i in range(num_models):
            repo.create(
                name=f"model_{i}",
                description=f"Model {i}",
                target="return_1d",
                model_type="ridge",
                feature_config=json.dumps(["rsi_14"]),
            )

        result = repo.list()

        assert len(result) == num_models

    def test_list_empty(self, db_connection):
        repo = ModelRepository()

        result = repo.list()

        assert result == []


class TestCreateVersion:
    """Tests for ModelRepository.create_version()"""

    @pytest.mark.parametrize(
        "version,is_active",
        [
            (1, False),
            (2, True),
            (10, False),
        ],
    )
    def test_create_version_success(self, db_connection, sample_model, version, is_active):
        repo = ModelRepository()
        metrics = json.dumps({"rmse": 0.01, "mae": 0.008})

        result = repo.create_version(
            model_id=sample_model["id"],
            version=version,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=metrics,
            artifact_path=f"/models/{sample_model['id']}/{version}/model.joblib",
            is_active=is_active,
        )

        assert result["id"] is not None
        assert result["model_id"] == sample_model["id"]
        assert result["version"] == version
        assert result["training_start"] == date(2023, 1, 1)
        assert result["training_end"] == date(2023, 12, 31)
        assert result["is_active"] is is_active
        assert result["artifact_path"] == f"/models/{sample_model['id']}/{version}/model.joblib"

    def test_create_version_invalid_model_raises(self, db_connection):
        repo = ModelRepository()

        with pytest.raises(Exception):  # ForeignKeyViolation
            repo.create_version(
                model_id=99999,
                version=1,
                training_start=date(2023, 1, 1),
                training_end=date(2023, 12, 31),
                metrics=json.dumps({}),
                artifact_path="/models/fake/model.joblib",
            )


class TestGetVersionById:
    """Tests for ModelRepository.get_version_by_id()"""

    def test_get_version_by_id_success(self, db_connection, sample_model):
        repo = ModelRepository()
        created = repo.create_version(
            model_id=sample_model["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({"rmse": 0.01}),
            artifact_path="/models/test/model.joblib",
        )

        result = repo.get_version_by_id(created["id"])

        assert result is not None
        assert result["id"] == created["id"]
        assert result["version"] == 1

    def test_get_version_by_id_not_found(self, db_connection):
        repo = ModelRepository()

        result = repo.get_version_by_id(99999)

        assert result is None


class TestListVersions:
    """Tests for ModelRepository.list_versions()"""

    @pytest.mark.parametrize("num_versions", [1, 3, 5])
    def test_list_versions_success(self, db_connection, sample_model, num_versions):
        repo = ModelRepository()

        for i in range(1, num_versions + 1):
            repo.create_version(
                model_id=sample_model["id"],
                version=i,
                training_start=date(2023, 1, 1),
                training_end=date(2023, 12, 31),
                metrics=json.dumps({"rmse": 0.01}),
                artifact_path=f"/models/test/{i}/model.joblib",
            )

        result = repo.list_versions(sample_model["id"])

        assert len(result) == num_versions

    def test_list_versions_empty(self, db_connection, sample_model):
        repo = ModelRepository()

        result = repo.list_versions(sample_model["id"])

        assert result == []

    def test_list_versions_ordered_by_version_desc(self, db_connection, sample_model):
        repo = ModelRepository()

        for i in [1, 3, 2]:  # Create out of order
            repo.create_version(
                model_id=sample_model["id"],
                version=i,
                training_start=date(2023, 1, 1),
                training_end=date(2023, 12, 31),
                metrics=json.dumps({}),
                artifact_path=f"/models/test/{i}/model.joblib",
            )

        result = repo.list_versions(sample_model["id"])

        versions = [r["version"] for r in result]
        assert versions == [3, 2, 1]

    def test_list_versions_only_for_specified_model(self, db_connection):
        repo = ModelRepository()

        # Create two models
        model1 = repo.create(
            name="model_1",
            description="",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps([]),
        )
        model2 = repo.create(
            name="model_2",
            description="",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps([]),
        )

        # Create versions for each
        repo.create_version(
            model_id=model1["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/m1/v1",
        )
        repo.create_version(
            model_id=model1["id"],
            version=2,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/m1/v2",
        )
        repo.create_version(
            model_id=model2["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/m2/v1",
        )

        result1 = repo.list_versions(model1["id"])
        result2 = repo.list_versions(model2["id"])

        assert len(result1) == 2
        assert len(result2) == 1


class TestGetActiveVersion:
    """Tests for ModelRepository.get_active_version()"""

    def test_get_active_version_success(self, db_connection, sample_model):
        repo = ModelRepository()

        # Create inactive and active versions
        repo.create_version(
            model_id=sample_model["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 6, 30),
            metrics=json.dumps({}),
            artifact_path="/v1",
            is_active=False,
        )
        repo.create_version(
            model_id=sample_model["id"],
            version=2,
            training_start=date(2023, 7, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v2",
            is_active=True,
        )

        result = repo.get_active_version(sample_model["id"])

        assert result is not None
        assert result["version"] == 2
        assert result["is_active"] is True

    def test_get_active_version_none_active(self, db_connection, sample_model):
        repo = ModelRepository()

        repo.create_version(
            model_id=sample_model["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v1",
            is_active=False,
        )

        result = repo.get_active_version(sample_model["id"])

        assert result is None

    def test_get_active_version_no_versions(self, db_connection, sample_model):
        repo = ModelRepository()

        result = repo.get_active_version(sample_model["id"])

        assert result is None


class TestActivateVersion:
    """Tests for ModelRepository.activate_version()"""

    def test_activate_version_success(self, db_connection, sample_model):
        repo = ModelRepository()

        repo.create_version(
            model_id=sample_model["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v1",
            is_active=False,
        )
        repo.create_version(
            model_id=sample_model["id"],
            version=2,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v2",
            is_active=False,
        )

        result = repo.activate_version(sample_model["id"], version=2)

        assert result is not None
        assert result["version"] == 2
        assert result["is_active"] is True

    def test_activate_version_deactivates_others(self, db_connection, sample_model):
        repo = ModelRepository()

        repo.create_version(
            model_id=sample_model["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v1",
            is_active=True,  # Start as active
        )
        repo.create_version(
            model_id=sample_model["id"],
            version=2,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v2",
            is_active=False,
        )

        repo.activate_version(sample_model["id"], version=2)

        # Check version 1 is now inactive
        v1 = repo.list_versions(sample_model["id"])
        v1_record = next(v for v in v1 if v["version"] == 1)
        v2_record = next(v for v in v1 if v["version"] == 2)

        assert v1_record["is_active"] is False
        assert v2_record["is_active"] is True

    def test_activate_version_not_found(self, db_connection, sample_model):
        repo = ModelRepository()

        result = repo.activate_version(sample_model["id"], version=99)

        assert result is None

    def test_activate_version_wrong_model(self, db_connection):
        repo = ModelRepository()

        # Create two models
        model1 = repo.create(
            name="model_1",
            description="",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps([]),
        )
        model2 = repo.create(
            name="model_2",
            description="",
            target="return_1d",
            model_type="ridge",
            feature_config=json.dumps([]),
        )

        # Create version for model1
        repo.create_version(
            model_id=model1["id"],
            version=1,
            training_start=date(2023, 1, 1),
            training_end=date(2023, 12, 31),
            metrics=json.dumps({}),
            artifact_path="/v1",
        )

        # Try to activate for model2
        result = repo.activate_version(model2["id"], version=1)

        assert result is None
