from typing import Iterable

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP

from protocol_infer.core.datamodel.raw_packet import Rawpacket
from protocol_infer.core.interface.pcap_analysis import PCAPParser


class ScapyParser(PCAPParser):
    
    def parse(self, path: str) -> Iterable:
        # Stream packets so large captures do not need to be loaded into memory
        # before sessionization can start.
        with PcapReader(path) as packets:
            for packet in packets:
                if IP not in packet:
                    continue

                ip = packet[IP]

                if TCP in packet:
                    l4 = packet[TCP]
                    prot = "TCP"
                elif UDP in packet:
                    l4 = packet[UDP]
                    prot = "UDP"
                else:
                    continue

                yield Rawpacket(
                    timestamp=float(packet.time),
                    src_ip=ip.src,
                    src_port=l4.sport,
                    dst_ip=ip.dst,
                    dst_port=l4.dport,
                    protocol=prot,
                    payload=bytes(l4.payload),
                )
