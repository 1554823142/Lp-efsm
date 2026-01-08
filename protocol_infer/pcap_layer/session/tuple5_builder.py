from typing import Iterable, List
from protocol_infer.core.interface.pcap_analysis import SessionBuilder
from protocol_infer.core.datamodel.session import Session, SessionKey
from protocol_infer.core.datamodel.raw_packet import Rawpacket
from collections import defaultdict


class FiveTupleBuilder(SessionBuilder):

    ''' raw_packet ---> Session/Sessionkey
        由于流(会话)内的包不一定连续, 所以需要先收集各个流中的包, 
        收集到的包按时间戳排序得到完整且独立的一个个会话
    '''
    def build(self, packets: Iterable[Rawpacket]) -> List[Session]:
        
        temp_flow = defaultdict(list)       # 先用字典收集, 构造出5元组, 再排序

        for pkt in packets:

            # 构建session key(5元组)
            key = SessionKey(
                ip1=pkt.src_ip,
                port1=pkt.src_port,
                ip2=pkt.dst_ip,
                port2=pkt.dst_port,
                protocol=pkt.protocol
            )

            temp_flow[key].append(pkt)
        
        sessions = []
        for key, pkts in temp_flow.items():
            if len(pkts) > 1:                               # 单个包不需要排序
                pkts.sort(key=lambda p : p.timestamp)       # 每个流内部排序
            sessions.append(Session(key=key, packets=pkts))

        return sessions