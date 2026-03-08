
from typing import Iterable, List
from protocol_infer.core.interface.pcap_analysis import SessionBuilder
from protocol_infer.core.datamodel.session import Session, SessionKey
from protocol_infer.core.datamodel.raw_packet import Rawpacket
from collections import defaultdict


class FiveTupleBuilder(SessionBuilder):

    '''
        基于5元组(IP1/Port1, IP2/Port2, Protocol)构建会话
        raw_packet ---> Session/Sessionkey
        由于流(会话)内的包不一定连续, 所以需要先收集各个流中的包, 
        收集到的包按时间戳排序得到完整且独立的一个个会话
    '''
    def build(self, packets: Iterable[Rawpacket]) -> List[Session]:
        
        temp_flow = defaultdict(list)       # 先用字典收集, 构造出5元组, 再排序

        for pkt in packets:

            # 构建session key(5元组)
            # 保证 key 的唯一性: ip1/port1 始终小于 ip2/port2 (简单的规范化)
            # 如一个 TCP 连接会产生两个方向的包, 而两个包的方向不同, 构造出的ip1/ip2就会相反
            # 需要整合到一个sessionkey, 这样可以减少冗余
            # 这里的大小比较仅仅是合并重复key, ip1/2并不确定方向
            if pkt.src_ip < pkt.dst_ip or (pkt.src_ip == pkt.dst_ip and pkt.src_port <= pkt.dst_port):
                key = SessionKey(
                    ip1=pkt.src_ip,
                    port1=pkt.src_port,
                    ip2=pkt.dst_ip,
                    port2=pkt.dst_port,
                    protocol=pkt.protocol
                )
            else:
                key = SessionKey(
                    ip1=pkt.dst_ip,
                    port1=pkt.dst_port,
                    ip2=pkt.src_ip,
                    port2=pkt.src_port,
                    protocol=pkt.protocol
                )

            temp_flow[key].append(pkt)
        
        sessions = []
        for key, pkts in temp_flow.items():
            if len(pkts) > 0:                               
                pkts.sort(key=lambda p : p.timestamp)       # 每个流内部排序
                sessions.append(Session(key=key, packets=pkts))

        return sessions
