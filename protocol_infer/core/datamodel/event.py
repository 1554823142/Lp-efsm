from dataclasses import dataclass
from enum import Enum
from .session import SessionKey

class Direction(Enum):
    C2S = 1
    S2C = 2


    # 为了迎合特征向量全float的要求
    def to_feature(self) -> float:      
        return 0.0 if self == Direction.C2S else 1.0

@dataclass(frozen=True)
class MessageEvent:
    session_key: SessionKey         
    timestamp: float
    payload: bytes
    direction: Direction
