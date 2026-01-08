from abc import ABC, abstractmethod
from typing import Dict, List
from protocol_infer.core.datamodel.session import SessionKey
class FSMInfer(ABC):

    @abstractmethod
    def infer(self, sequences: Dict[SessionKey, List[str]]):
        pass