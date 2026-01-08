from typing import Iterable
from protocol_infer.core.interface.pcap_analysis import PCAPParser
from protocol_infer.core.datamodel.raw_packet import Rawpacket
from scapy.all import rdpcap, PcapReader
from scapy.layers.inet import IP, TCP, UDP, ICMP

class ScapyParser(PCAPParser):
    
    def parse(self, path: str) -> Iterable:
        
        packets = rdpcap(path)      # 解析pcap文件

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
                payload=bytes(l4.payload)
            )
        

