from src.trainer.base_trainer import BaseTrainer
from src.trainer.ner_trainer import NerTrainer

class TrainerFactory:
    def __init__(self, model_name: str, data_directory: str) -> None:
         self._creators = {
            "ner": NerTrainer(model_name=model_name, data_directory=data_directory)
        }
    
    def get_trainer(self, format: str) -> BaseTrainer:
        return self._creators[format]