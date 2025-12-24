from omnivore.instrument import InstrumentRepository
from omnivore.ohlcv import OhlcvRepository
from omnivore.prediction import ModelRepository, PredictionRepository


class DataService:
    """
    Handles business logic and orchestration, using repositories for data access.
    """

    def __init__(self):
        self.instruments = InstrumentRepository()
        self.predictions = PredictionRepository()
        self.ohlcv = OhlcvRepository()
        self.models = ModelRepository()
