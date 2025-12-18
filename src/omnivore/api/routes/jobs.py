# ============================================================================
# FILE: src/omnivore/api/routes/jobs.py
# ============================================================================
from flask import Blueprint, jsonify, current_app
from rq.job import Job

bp = Blueprint("jobs", __name__)


@bp.route("/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Get status of a job."""
    try:
        job = Job.fetch(job_id, connection=current_app.redis)
    except Exception:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job.id,
        "status": job.get_status(),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
    }

    if job.is_finished:
        response["result"] = job.result
    elif job.is_failed:
        response["error"] = str(job.exc_info)

    return jsonify(response)


@bp.route("/drift/reports", methods=["GET"])
def get_drift_reports():
    """Get drift reports."""
    from flask import request
    from omnivore.services import DriftMonitor

    drift_monitor = DriftMonitor()
    reports = drift_monitor.get_reports(
        model_version_id=request.args.get("model_version_id", type=int),
        drift_type=request.args.get("drift_type"),
        alerts_only=request.args.get("alerts_only", "false").lower() == "true",
        limit=request.args.get("limit", 50, type=int),
    )
    return jsonify(reports)
