from src.data.ner_data import NerData
from src.data.base_data import BaseData

class DataFactory:
    def __init__(self, model_name: str) -> None:
        self._creators = {
            "ner": NerData(model_name=model_name)
        }

    def get_data(self, format: str) -> BaseData:
        return self._creators[format]
    