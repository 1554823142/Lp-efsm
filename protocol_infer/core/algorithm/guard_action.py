from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable

class GuardActionLearner(ABC):
    @abstractmethod
    def learn(self, var_instances: List[Dict[str, float]]) -> Tuple[Optional[Callable], Optional[Callable]]:
        pass
