import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root / "protocol_infer"))
sys.path.insert(0, str(project_root))
from typing import List
from protocol_infer.core.datamodel.session import Session, SessionKey
from protocol_infer.core.datamodel.raw_packet import Rawpacket
from protocol_infer.pcap_layer.segmentation.packet_level import PacketLevelSegmenter
from protocol_infer.core.datamodel.event import Direction

def test_packet_segmentation():
    # 模拟 Session
    session_key = SessionKey("192.168.1.100", 12345, "192.168.1.200", 502, "TCP")
    
    # 场景 1: 标准 Modbus 流量 (Dst Port = 502)
    packets1 = [
        Rawpacket(0.1, "192.168.1.100", "192.168.1.200", 12345, 502, "TCP", b"query"),   # C -> S
        Rawpacket(0.2, "192.168.1.200", "192.168.1.100", 502, 12345, "TCP", b"response") # S -> C
    ]
    session1 = Session(key=session_key, packets=packets1)
    
    segmenter = PacketLevelSegmenter(server_ports={502})
    events1 = segmenter.segment(session1)
    
    print(f"Test 1 (Standard Modbus): {len(events1)} events")
    if len(events1) == 2:
        print(f"  Event 1 Direction: {events1[0].direction} (Expected C2S)")
        print(f"  Event 2 Direction: {events1[1].direction} (Expected S2C)")

    # 场景 2: 乱序/Server 先发包 (但端口已知)
    packets2 = [
        Rawpacket(0.1, "192.168.1.200", "192.168.1.100", 502, 12345, "TCP", b"welcome"), # S -> C
        Rawpacket(0.2, "192.168.1.100", "192.168.1.200", 12345, 502, "TCP", b"login")    # C -> S
    ]
    session2 = Session(key=session_key, packets=packets2)
    events2 = segmenter.segment(session2)
    
    print(f"\nTest 2 (Server First): {len(events2)} events")
    if len(events2) == 2:
        print(f"  Event 1 Direction: {events2[0].direction} (Expected S2C)")
        print(f"  Event 2 Direction: {events2[1].direction} (Expected C2S)")

    # 场景 3: 未知端口 (兜底逻辑)
    packets3 = [
        Rawpacket(0.1, "10.0.0.1", "10.0.0.2", 33333, 44444, "UDP", b"msg1"),
        Rawpacket(0.2, "10.0.0.2", "10.0.0.1", 44444, 33333, "UDP", b"msg2")
    ]
    session3 = Session(key=SessionKey("10.0.0.1", 33333, "10.0.0.2", 44444, "UDP"), packets=packets3)
    events3 = segmenter.segment(session3)
    
    print(f"\nTest 3 (Unknown Ports): {len(events3)} events")
    if len(events3) == 2:
        # 第一个发包的被认为是 Client
        print(f"  Event 1 Direction: {events3[0].direction} (Expected C2S)")
        print(f"  Event 2 Direction: {events3[1].direction} (Expected S2C)")

if __name__ == "__main__":
    test_packet_segmentation()
