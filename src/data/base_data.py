import logging
from abc import abstractmethod
from transformers import AutoTokenizer

_logger = logging.getLogger(__name__)

class BaseData:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load(self, data_directory: str):
        _logger.debug("Load Data From {}".format(data_directory))
        return self._load(data_directory)

    @abstractmethod
    def _load(self, data_directory: str):
        pass