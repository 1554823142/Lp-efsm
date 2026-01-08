from abc import ABC, abstractmethod
from typing import List, Any

class MessageAbstractor(ABC):
    """
    Feature vector -> protocol symbol
    """

    @abstractmethod
    def fit(self, features: List[List[float]]) -> None:
        pass

    @abstractmethod
    def abstract(self, feature: List[float]) -> Any:
        pass
