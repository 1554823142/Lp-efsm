from abc import ABC, abstractmethod
from protocol_infer.core.model.fsm import FSM

class StateMerger(ABC):
    @abstractmethod
    def merge(self, fsm: FSM) -> FSM:
        pass
