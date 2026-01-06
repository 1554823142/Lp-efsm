from core.interface.pcap_analysis import Segmenter
from core.datamodel.event import MessageEvent, Direction

class PacketLevelSegmenter(Segmenter):

    def segment(self, session):
        events = []

        for pkt in session.packets:
            direction = Direction.C2S       # 默认设为客户端到服务器
            events.append(
                MessageEvent(
                    session_key=session.key,
                    timestamp=pkt.timestamp,
                    payload=pkt.payload,
                    direction=direction
                )
            )

        return events
