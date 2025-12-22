from omnivore.instrument import InstrumentRepository
from omnivore.ohlcv import OhlcvRepository
from omnivore.repositories.model_repository import ModelRepository
from omnivore.repositories.prediction_repository import PredictionRepository


class DataService:
    """
    Handles business logic and orchestration, using repositories for data access.
    """

    def __init__(self):
        self.instruments = InstrumentRepository()
        self.predictions = PredictionRepository()
        self.ohlcv = OhlcvRepository()
        self.models = ModelRepository()
