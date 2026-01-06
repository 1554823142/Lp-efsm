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


@dataclass
class Session:
    key: SessionKey
    packets: List[Rawpacket]