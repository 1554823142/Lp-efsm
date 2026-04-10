from dataclasses import dataclass
from .raw_packet import Rawpacket
from typing import List

@dataclass(frozen=True)
class SessionKey:
    ip1: str
    port1: int
    ip2: str
    port2: int
    protocol: str
    segment_id: int = 0


@dataclass
class Session:
    key: SessionKey
    packets: List[Rawpacket]
