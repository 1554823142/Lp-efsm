from abc import ABC, abstractmethod
from typing import Dict, List
from protocol_infer.core.model.efsm import FSM, EFSM
from protocol_infer.core.datamodel.session import SessionKey

class EFSMBuilder(ABC):
    @abstractmethod
    def build(self, fsm: FSM, 
              sequences: Dict[SessionKey, List[Dict]]) ->EFSM:
        pass