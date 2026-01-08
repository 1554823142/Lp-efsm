from typing import List
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.pcap_layer.parser.scapy_parser import ScapyParser
from protocol_infer.pcap_layer.session.tuple5_builder import FiveTupleBuilder
from protocol_infer.pcap_layer.segmentation.packet_level import PacketLevelSegmenter


class PCAPPipeline:
    def run(self, pcap_path: str) -> Trace:
        parser = ScapyParser()
        session_builder = FiveTupleBuilder()
        segmenter = PacketLevelSegmenter()

        raw_packets = parser.parse(pcap_path)
        sessions = session_builder.build(raw_packets)

        events = []
        for session in sessions:
            events.extend(segmenter.segment(session))

        events.sort(key=lambda e: e.timestamp)
        return Trace(events=events)

        
        