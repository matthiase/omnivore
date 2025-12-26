from flask import Blueprint, current_app, jsonify, request

from omnivore.jobs import train_model_job
from omnivore.model.registry import ModelRegistry

bp = Blueprint("models", __name__)
model_registry = ModelRegistry()


@bp.route("", methods=["GET"])
def list_models():
    """List all model definitions."""
    models = model_registry.list_models()
    return jsonify(models)


@bp.route("", methods=["POST"])
def create_model():
    """Create a new model definition."""
    data = request.get_json()

    model = model_registry.create_model(
        name=data["name"],
        description=data.get("description"),
        target=data["target"],
        model_type=data["model_type"],
        feature_config=data["feature_config"],
        hyperparameters=data.get("hyperparameters"),
    )

    return jsonify(model), 201


@bp.route("/<int:model_id>", methods=["GET"])
def get_model(model_id: int):
    """Get model definition."""
    model = model_registry.get_model(model_id)
    if not model:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(model)


@bp.route("/<int:model_id>/train", methods=["POST"])
def train_model(model_id: int):
    """Trigger model training."""
    model = model_registry.get_model(model_id)
    if not model:
        return jsonify({"error": "Model not found"}), 404

    data = request.get_json()

    job = current_app.training_queue.enqueue(
        train_model_job,
        model_id=model_id,
        instrument_id=data["instrument_id"],
        training_start=data["training_start"],
        training_end=data["training_end"],
        test_size=data.get("test_size", 0.2),
        job_timeout="30m",
    )

    return jsonify(
        {
            "job_id": job.id,
            "status": "queued",
        }
    ), 202


@bp.route("/<int:model_id>/versions", methods=["GET"])
def list_versions(model_id: int):
    """List all trained versions of a model."""
    versions = model_registry.list_versions(model_id)
    return jsonify(versions)


@bp.route("/<int:model_id>/versions/<int:version>/activate", methods=["POST"])
def activate_version(model_id: int, version: int):
    """Set a version as active."""
    try:
        result = model_registry.activate_version(model_id, version)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/<int:model_id>/compare", methods=["GET"])
def compare_versions(model_id: int):
    """Compare performance across model versions."""
    df = model_registry.compare_versions(model_id)
    return jsonify(df.to_dict(orient="records"))
