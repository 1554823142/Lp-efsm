from dataclasses import dataclass
from typing import List
from .event import MessageEvent

@dataclass
class Trace:
    events: List[MessageEvent]
