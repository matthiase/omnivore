from flask import Flask
from redis import Redis
from rq import Queue

from omnivore.config import config


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)

    # Configure Redis and RQ
    app.redis = Redis.from_url(config.redis_url)
    app.task_queue = Queue("default", connection=app.redis)
    app.training_queue = Queue("training", connection=app.redis)

    # Register blueprints
    from omnivore.routes.instruments import bp as instruments_bp
    from omnivore.routes.jobs import bp as jobs_bp
    from omnivore.routes.models import bp as models_bp
    from omnivore.routes.predictions import bp as predictions_bp

    app.register_blueprint(instruments_bp, url_prefix="/api/instruments")
    app.register_blueprint(models_bp, url_prefix="/api/models")
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(jobs_bp, url_prefix="/api/jobs")

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    return app


# For flask run command
app = create_app()
