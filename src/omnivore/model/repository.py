import json
from typing import List, Optional

from omnivore import db


class ModelRepository:
    """
    Repository for model and model_version-related data access.
    Encapsulates all SQL and queries for the models and model_versions tables.
    """

    def get_by_id(self, model_id: int) -> Optional[dict]:
        """Get a model by its ID."""
        return db.fetch_one("SELECT * FROM models WHERE id = %s", (model_id,))

    def get_by_name(self, name: str) -> Optional[dict]:
        """Get a model by its unique name."""
        return db.fetch_one("SELECT * FROM models WHERE name = %s", (name,))

    def list(self) -> List[dict]:
        """List all models."""
        return db.fetch_all("SELECT * FROM models ORDER BY created_at DESC")

    def create(
        self,
        name: str,
        description: str,
        target: str,
        model_type: str,
        feature_config,
        hyperparameters=None,
    ) -> dict:
        """Create a new model."""
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
                feature_config,
                json.dumps(hyperparameters or {}),
            ),
        )

    # Model Versions

    def get_version_by_id(self, version_id: int) -> Optional[dict]:
        """Get a model version by its ID."""
        return db.fetch_one("SELECT * FROM model_versions WHERE id = %s", (version_id,))

    def list_versions(self, model_id: int) -> List[dict]:
        """List all versions for a given model."""
        return db.fetch_all(
            "SELECT * FROM model_versions WHERE model_id = %s ORDER BY version DESC", (model_id,)
        )

    def get_active_version(self, model_id: int) -> Optional[dict]:
        """Get the active version for a given model."""
        return db.fetch_one(
            "SELECT * FROM model_versions WHERE model_id = %s AND is_active = true", (model_id,)
        )

    def create_version(
        self,
        model_id: int,
        version: int,
        training_start,
        training_end,
        metrics,
        artifact_path: str,
        is_active: bool = False,
    ) -> dict:
        """Create a new model version."""
        return db.fetch_one(
            """
            INSERT INTO model_versions (
                model_id, version, training_start, training_end,
                metrics, artifact_path, is_active
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (model_id, version, training_start, training_end, metrics, artifact_path, is_active),
        )

    def activate_version(self, model_id: int, version: int) -> Optional[dict]:
        """
        Activate a specific version for a model (deactivate others).
        Returns the activated version.
        """
        # Deactivate all other versions
        db.execute("UPDATE model_versions SET is_active = false WHERE model_id = %s", (model_id,))
        # Activate the specified version
        return db.fetch_one(
            "UPDATE model_versions SET is_active = true WHERE model_id = %s AND version = %s RETURNING *",
            (model_id, version),
        )
