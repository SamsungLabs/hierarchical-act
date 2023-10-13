from abc import *
from typing import Dict

class BaseLoss(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, model)->Dict[str, float]:
        raise NotImplementedError