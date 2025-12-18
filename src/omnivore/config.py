# ============================================================================
# FILE: src/omnivore/config.py
# ============================================================================
import os
import json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    database_url: str
    redis_url: str
    model_storage_path: Path
    features_config_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            database_url=os.environ["DATABASE_URL"],
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            model_storage_path=Path(os.environ.get("MODEL_STORAGE_PATH", "./models")),
            features_config_path=Path(os.environ.get("FEATURES_CONFIG_PATH", "./config/features.json")),
        )

    def load_features_config(self) -> dict:
        with open(self.features_config_path) as f:
            return json.load(f)


config = Config.from_env()
