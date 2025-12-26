from flask import Blueprint, current_app, jsonify, request

from omnivore.instrument.repository import InstrumentRepository
from omnivore.jobs import compute_features_job, refresh_data_job

bp = Blueprint("instruments", __name__)

instrument_repo = InstrumentRepository()


@bp.route("", methods=["GET"])
def list_instruments():
    """List all instruments."""
    active_only = request.args.get("active", "true").lower() == "true"
    instruments = instrument_repo.list(active_only=active_only)
    return jsonify(instruments)


@bp.route("", methods=["POST"])
def create_instrument():
    """Create a new instrument."""
    data = request.get_json()

    instrument = instrument_repo.create(
        symbol=data["symbol"],
        name=data.get("name"),
        asset_type=data.get("asset_type", "stock"),
        exchange=data.get("exchange"),
    )

    return jsonify(instrument), 201


@bp.route("/<int:instrument_id>", methods=["GET"])
def get_instrument(instrument_id: int):
    """Get instrument by ID."""
    instrument = instrument_repo.get_by_id(instrument_id)
    if not instrument:
        return jsonify({"error": "Instrument not found"}), 404
    return jsonify(instrument)


@bp.route("/<int:instrument_id>/refresh", methods=["POST"])
def refresh_instrument(instrument_id: int):
    """Trigger data refresh for an instrument."""
    instrument = instrument_repo.get_by_id(instrument_id)
    if not instrument:
        return jsonify({"error": "Instrument not found"}), 404

    data = request.get_json() or {}

    # Queue the refresh job
    job = current_app.task_queue.enqueue(
        refresh_data_job,
        symbol=instrument["symbol"],
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
    )

    return jsonify(
        {
            "job_id": job.id,
            "status": "queued",
            "instrument": instrument["symbol"],
        }
    ), 202


@bp.route("/<int:instrument_id>/features", methods=["POST"])
def compute_features(instrument_id: int):
    """Trigger feature computation for an instrument."""
    instrument = instrument_repo.get_by_id(instrument_id)
    if not instrument:
        return jsonify({"error": "Instrument not found"}), 404

    data = request.get_json() or {}

    job = current_app.task_queue.enqueue(
        compute_features_job,
        instrument_id=instrument_id,
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
    )

    return jsonify(
        {
            "job_id": job.id,
            "status": "queued",
        }
    ), 202
