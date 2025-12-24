from omnivore.instrument.repository import InstrumentRepository
from omnivore.model.repository import ModelRepository
from omnivore.ohlcv.repository import OhlcvRepository
from omnivore.prediction.repository import PredictionRepository


class DataService:
    """
    Handles business logic and orchestration, using repositories for data access.
    """

    def __init__(self):
        self.instruments = InstrumentRepository()
        self.predictions = PredictionRepository()
        self.ohlcv = OhlcvRepository()
        self.models = ModelRepository()
