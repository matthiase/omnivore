# ============================================================================
# FILE: src/omnivore/api/routes/predictions.py
# ============================================================================
from datetime import date

from flask import Blueprint, current_app, jsonify, request

from omnivore.jobs import backfill_actuals_job, generate_predictions_job
from omnivore.prediction import PredictionService

bp = Blueprint("predictions", __name__)
prediction_service = PredictionService()


@bp.route("", methods=["GET"])
def list_predictions():
    """List predictions with filters."""
    predictions = prediction_service.get_predictions(
        instrument_id=request.args.get("instrument_id", type=int),
        model_id=request.args.get("model_id", type=int),
        start_date=request.args.get("start_date"),
        end_date=request.args.get("end_date"),
        horizon=request.args.get("horizon"),
        limit=request.args.get("limit", 100, type=int),
    )
    return jsonify(predictions)


@bp.route("/generate", methods=["POST"])
def generate_predictions():
    """Generate predictions for instruments."""
    data = request.get_json()

    job = current_app.task_queue.enqueue(
        generate_predictions_job,
        model_id=data["model_id"],
        instrument_ids=data["instrument_ids"],
        prediction_date=data.get("prediction_date", str(date.today())),
        horizons=data.get("horizons", ["1d"]),
    )

    return jsonify(
        {
            "job_id": job.id,
            "status": "queued",
        }
    ), 202


@bp.route("/backfill-actuals", methods=["POST"])
def backfill_actuals():
    """Backfill actual outcomes for past predictions."""
    data = request.get_json()

    job = current_app.task_queue.enqueue(
        backfill_actuals_job,
        instrument_id=data["instrument_id"],
    )

    return jsonify(
        {
            "job_id": job.id,
            "status": "queued",
        }
    ), 202


@bp.route("/performance", methods=["GET"])
def get_performance():
    """Get performance summary for a model."""
    model_id = request.args.get("model_id", type=int)
    if not model_id:
        return jsonify({"error": "model_id required"}), 400

    summary = prediction_service.get_performance_summary(
        model_id=model_id,
        instrument_id=request.args.get("instrument_id", type=int),
        horizon=request.args.get("horizon"),
    )
    return jsonify(summary)
