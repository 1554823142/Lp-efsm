from dataclasses import dataclass
from enum import Enum
from .session import SessionKey

class Direction(Enum):
    C2S = 1
    S2C = 2

@dataclass(frozen=True)
class MessageEvent:
    session_key: SessionKey         
    timestamp: float
    payload: bytes
    direction: Direction
