from abc import abstractmethod
from src.data.data_factory import DataFactory

class BaseTrainer:
    def __init__(self, model_name: str, data_directory: str) -> None:
        self.model_name = model_name
        self.data_directory = data_directory

    def load(self, model_directory: str):
        self._load(model_directory=model_directory)
    
    def train(self):
        self._train()

    @abstractmethod
    def _load(self, model_directory: str):
        pass
    
    @abstractmethod
    def _train(self):
        pass

    def _load_dataset(self, trainer_type: str):
        data_factory = DataFactory(model_name=self.model_name)
        dataset = data_factory.get_data(trainer_type)
        return dataset.load(data_directory=self.data_directory)
    