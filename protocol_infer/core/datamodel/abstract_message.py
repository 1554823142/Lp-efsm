from dataclasses import dataclass
from typing import Dict
from .event import Direction
from .session import SessionKey

@dataclass
class AbstractMessage:
    session_key: SessionKey
    timestamp: float
    direction: Direction

    symbol: str                 # FSM 用
    vars: Dict[str, float|int]  # EFSM 用