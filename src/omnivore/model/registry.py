import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from omnivore import db
from omnivore.config import config
from omnivore.services.feature_engine import FeatureEngine


class ModelRegistry:
    """Manages model definitions, training, versioning, and inference."""

    MODEL_TYPES = {
        "ridge": Ridge,
        "xgboost": XGBRegressor,
        "lightgbm": LGBMRegressor,
    }

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.storage_path = config.model_storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_model(
        self,
        name: str,
        description: str,
        target: str,
        model_type: str,
        feature_config: list[str],
        hyperparameters: dict = None,
    ) -> dict:
        """Create a new model definition."""
        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. Valid: {list(self.MODEL_TYPES.keys())}"
            )

        return db.fetch_one(
            """
            INSERT INTO models (name, description, target, model_type, feature_config, hyperparameters)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                name,
                description,
                target,
                model_type,
                json.dumps(feature_config),
                json.dumps(hyperparameters or {}),
            ),
        )

    def get_model(self, model_id: int) -> dict | None:
        """Get model definition by ID."""
        return db.fetch_one("SELECT * FROM models WHERE id = %s", (model_id,))

    def get_model_by_name(self, name: str) -> dict | None:
        """Get model definition by name."""
        return db.fetch_one("SELECT * FROM models WHERE name = %s", (name,))

    def list_models(self) -> list[dict]:
        """List all model definitions."""
        return db.fetch_all("SELECT * FROM models ORDER BY name")

    def _get_artifact_path(self, model_id: int, version: int) -> Path:
        """Get the filesystem path for a model artifact."""
        return self.storage_path / str(model_id) / str(version)

    def _save_artifact(
        self,
        model_id: int,
        version: int,
        model: Any,
        feature_config: list[str],
        metadata: dict,
    ) -> str:
        """Save model artifact to filesystem."""
        artifact_dir = self._get_artifact_path(model_id, version)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = artifact_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save feature config
        with open(artifact_dir / "feature_config.json", "w") as f:
            json.dump(feature_config, f)

        # Save metadata
        with open(artifact_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, default=str)

        return str(model_path)

    def _load_artifact(self, artifact_path: str) -> tuple[Any, list[str]]:
        """Load model artifact from filesystem."""
        artifact_dir = Path(artifact_path).parent

        model = joblib.load(artifact_path)

        with open(artifact_dir / "feature_config.json") as f:
            feature_config = json.load(f)

        return model, feature_config

    def train(
        self,
        model_id: int,
        instrument_id: int,
        training_start: date,
        training_end: date,
        test_size: float = 0.2,
    ) -> dict:
        """Train a new version of a model."""
        model_def = self.get_model(model_id)
        if not model_def:
            raise ValueError(f"Model {model_id} not found")

        # Get training data
        feature_names = model_def["feature_config"]
        if isinstance(feature_names, str):
            feature_names = json.loads(feature_names)

        X, y = self.feature_engine.get_training_data(
            instrument_id=instrument_id,
            target=model_def["target"],
            start_date=training_start,
            end_date=training_end,
            feature_names=feature_names,
        )

        if len(X) < 50:
            raise ValueError(f"Insufficient training data: {len(X)} rows")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=False,  # Time series: no shuffle
        )

        # Create and train model
        hyperparams = model_def["hyperparameters"]
        if isinstance(hyperparams, str):
            hyperparams = json.loads(hyperparams)

        model_class = self.MODEL_TYPES[model_def["model_type"]]
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "train": {
                "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                "mae": float(mean_absolute_error(y_train, y_pred_train)),
                "r2": float(r2_score(y_train, y_pred_train)),
                "directional_accuracy": float((np.sign(y_train) == np.sign(y_pred_train)).mean()),
            },
            "test": {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                "mae": float(mean_absolute_error(y_test, y_pred_test)),
                "r2": float(r2_score(y_test, y_pred_test)),
                "directional_accuracy": float((np.sign(y_test) == np.sign(y_pred_test)).mean()),
            },
            "data": {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            },
        }

        # Determine version number
        latest = db.fetch_one(
            "SELECT COALESCE(MAX(version), 0) as v FROM model_versions WHERE model_id = %s",
            (model_id,),
        )
        new_version = latest["v"] + 1

        # Save artifact
        artifact_path = self._save_artifact(
            model_id=model_id,
            version=new_version,
            model=model,
            feature_config=feature_names,
            metadata={
                "training_start": training_start,
                "training_end": training_end,
                "instrument_id": instrument_id,
                "metrics": metrics,
            },
        )

        # Record version
        version_record = db.fetch_one(
            """
            INSERT INTO model_versions
                (model_id, version, training_start, training_end, metrics, artifact_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                model_id,
                new_version,
                training_start,
                training_end,
                json.dumps(metrics),
                artifact_path,
            ),
        )

        return {
            "model_id": model_id,
            "version": new_version,
            "metrics": metrics,
            "artifact_path": artifact_path,
            "version_record": version_record,
        }

    def activate_version(self, model_id: int, version: int) -> dict:
        """Set a specific version as the active version for a model."""
        # Deactivate all versions for this model
        db.execute("UPDATE model_versions SET is_active = false WHERE model_id = %s", (model_id,))

        # Activate the specified version
        result = db.fetch_one(
            """
            UPDATE model_versions
            SET is_active = true
            WHERE model_id = %s AND version = %s
            RETURNING *
            """,
            (model_id, version),
        )

        if not result:
            raise ValueError(f"Version {version} not found for model {model_id}")

        return result

    def get_active_version(self, model_id: int) -> dict | None:
        """Get the currently active version for a model."""
        return db.fetch_one(
            """
            SELECT * FROM model_versions
            WHERE model_id = %s AND is_active = true
            """,
            (model_id,),
        )

    def get_version(self, model_id: int, version: int) -> dict | None:
        """Get a specific model version."""
        return db.fetch_one(
            "SELECT * FROM model_versions WHERE model_id = %s AND version = %s", (model_id, version)
        )

    def list_versions(self, model_id: int) -> list[dict]:
        """List all versions for a model."""
        return db.fetch_all(
            "SELECT * FROM model_versions WHERE model_id = %s ORDER BY version DESC", (model_id,)
        )

    def predict(
        self,
        model_version_id: int,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Generate predictions using a trained model version."""
        version = db.fetch_one("SELECT * FROM model_versions WHERE id = %s", (model_version_id,))

        if not version:
            raise ValueError(f"Model version {model_version_id} not found")

        model, feature_config = self._load_artifact(version["artifact_path"])

        # Ensure features are in correct order
        X = features[feature_config]

        return model.predict(X)

    def compare_versions(self, model_id: int) -> pd.DataFrame:
        """Compare metrics across all versions of a model."""
        versions = self.list_versions(model_id)

        rows = []
        for v in versions:
            metrics = v["metrics"]
            if isinstance(metrics, str):
                metrics = json.loads(metrics)

            rows.append(
                {
                    "version": v["version"],
                    "is_active": v["is_active"],
                    "trained_at": v["trained_at"],
                    "train_rmse": metrics.get("train", {}).get("rmse"),
                    "test_rmse": metrics.get("test", {}).get("rmse"),
                    "train_r2": metrics.get("train", {}).get("r2"),
                    "test_r2": metrics.get("test", {}).get("r2"),
                    "train_dir_acc": metrics.get("train", {}).get("directional_accuracy"),
                    "test_dir_acc": metrics.get("test", {}).get("directional_accuracy"),
                }
            )

        return pd.DataFrame(rows)
