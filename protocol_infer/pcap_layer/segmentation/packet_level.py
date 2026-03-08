
from typing import List, Set, Optional
from protocol_infer.core.interface.pcap_analysis import Segmenter
from protocol_infer.core.datamodel.event import MessageEvent, Direction
from protocol_infer.core.datamodel.session import Session

class PacketLevelSegmenter(Segmenter):

    def __init__(self, server_ports: Set[int] = None):
        # 默认使用常见工控端口: 502(Modbus), 20000(DNP3), 2404(IEC104)
        self.server_ports = server_ports or {502, 20000, 2404}

    def segment(self, session: Session) -> List[MessageEvent]:
        if not session.packets:
            return []

        client_ip = self._infer_client_ip(session)
        events = []

        for pkt in session.packets:
            if not pkt.payload:
                continue

            direction = Direction.C2S if pkt.src_ip == client_ip else Direction.S2C
            
            events.append(
                MessageEvent(
                    session_key=session.key,
                    timestamp=pkt.timestamp,
                    payload=pkt.payload,
                    direction=direction
                )
            )

        return events

    def _infer_client_ip(self, session: Session) -> str:
        """
        推断 Client IP:
        1. 优先检查是否存在发往已知服务端口的包 (Src -> ServerPort)
        2. 其次检查是否存在来自已知服务端口的包 (ServerPort -> Dst)
        3. 兜底：使用第一个携带 payload 的包的 Src IP
        4. 最后的兜底：使用 SessionKey 中的 ip1
        """
        # 优先：找发往已知服务端口的包
        for pkt in session.packets:
            if not pkt.payload:
                continue
            if pkt.dst_port in self.server_ports:
                return pkt.src_ip
            if pkt.src_port in self.server_ports:
                return pkt.dst_ip
        
        # 兜底：第一个有 payload 的包的 src_ip
        first = next((p for p in session.packets if p.payload), None)
        return first.src_ip if first else session.key.ip1
