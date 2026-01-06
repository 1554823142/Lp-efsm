from dataclasses import dataclass

@dataclass(frozen=True)         # 只读数据
class Rawpacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str      # TCP / UDP
    payload: bytes