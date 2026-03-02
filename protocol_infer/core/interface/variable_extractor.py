from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.event import Direction


class VariableAbstractor(ABC):
    @abstractmethod
    def extract(self, raw_msg) -> Dict[str, Any]:
        pass