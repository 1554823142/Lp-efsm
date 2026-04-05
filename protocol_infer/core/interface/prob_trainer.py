from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.model.pefsm import PEFSM


class ProbTrainer(ABC):
    @abstractmethod
    def train(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
    ) -> PEFSM:
        pass
