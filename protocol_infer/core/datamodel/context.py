from dataclasses import dataclass
from typing import Dict, List
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.event import MessageEvent, Direction
from collections import Counter
import math


@dataclass
class SessionContext:

    # 方向判定变量
    is_client: bool = False
    server_port: int = 0
    
    # 统计变量
    packet_ratio: float = 0.0  # C2S / S2C 报文比例
    byte_ratio: float = 0.0    # C2S / S2C 字节比例
    pair_ratio: float = 0.0    # 请求-响应配对比例
    
    # 协议特征
    avg_message_len: float = 0.0
    message_len_std: float = 0.0
    total_messages: int = 0
    
    # 时序特征
    avg_interval: float = 0.0
    burstiness: float = 0.0
    
    # 协议指纹
    is_request_response: bool = False
    is_streaming: bool = False
    suspected_protocol: str = ""

class ContextExtractor:
    """提取单个消息的变量特征"""
    
    def extract_vars(self, event: MessageEvent) -> Dict[str, float]:
        payload = event.payload

        vars = {
            "len": float(len(payload)),
            "direction": event.direction.to_feature(),
            "entropy": self._calculate_entropy(payload),
            "b0": float(payload[0]) if len(payload) > 0 else -1.0,
            "b1": float(payload[1]) if len(payload) > 1 else -1.0,
        }
        return vars
    
    def _calculate_entropy(self, data: bytes) -> float:
        """计算字节序列的熵"""
        if not data:
            return 0.0
        
        freq = Counter(data)
        length = len(data)
        entropy = -sum((count / length) * math.log2(count / length) for count in freq.values() if count > 0)
        return entropy


class SessionContextBuilder:
    """构建会话级上下文信息"""
    
    MIN_EVENTS_FOR_PATTERN = 4  # 请求-响应或流式判断的最小事件数
    
    def build(self, events: List[MessageEvent]) -> SessionContext:
        """从事件列表构建SessionContext"""
        if not events:
            return SessionContext()
        
        # 按方向分组
        c2s_events = [e for e in events if e.direction == Direction.C2S]
        s2c_events = [e for e in events if e.direction == Direction.S2C]
        
        # 方向判定
        session_key = events[0].session_key
        is_client, server_port = self._determine_direction(session_key, c2s_events, s2c_events)
        
        # 统计变量
        packet_ratio = len(c2s_events) / len(s2c_events) if s2c_events else float('inf')
        byte_ratio = (sum(len(e.payload) for e in c2s_events) / sum(len(e.payload) for e in s2c_events)) if s2c_events else float('inf')
        pair_ratio = self._calculate_pair_ratio(events)
        
        # 协议特征
        lengths = [len(e.payload) for e in events]
        avg_message_len = sum(lengths) / len(lengths) if lengths else 0.0
        message_len_std = self._calculate_std(lengths) if len(lengths) > 1 else 0.0
        total_messages = len(events)
        
        # 时序特征
        intervals = [events[i+1].timestamp - events[i].timestamp for i in range(len(events)-1)] if len(events) > 1 else []
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        burstiness = self._calculate_burstiness(intervals)
        
        # 协议指纹
        is_request_response = self._is_request_response_pattern(events)
        is_streaming = self._is_streaming_pattern(events, avg_interval)
        suspected_protocol = self._detect_protocol(events, session_key)
        
        return SessionContext(
            is_client=is_client,
            server_port=server_port,
            packet_ratio=packet_ratio,
            byte_ratio=byte_ratio,
            pair_ratio=pair_ratio,
            avg_message_len=avg_message_len,
            message_len_std=message_len_std,
            total_messages=total_messages,
            avg_interval=avg_interval,
            burstiness=burstiness,
            is_request_response=is_request_response,
            is_streaming=is_streaming,
            suspected_protocol=suspected_protocol
        )
    
    def _determine_direction(self, session_key: SessionKey, 
                            c2s_events: List[MessageEvent],
                            s2c_events: List[MessageEvent]) -> tuple:
        """判断客户端和服务器端口，更稳健"""
        if not c2s_events or not s2c_events:
            # 没有对端数据，退化使用端口判断
            if session_key.port1 < session_key.port2:
                return True, session_key.port2
            else:
                return False, session_key.port1
        
        # 比较数据量
        bytes_c2s = sum(len(e.payload) for e in c2s_events)
        bytes_s2c = sum(len(e.payload) for e in s2c_events)
        if bytes_c2s >= bytes_s2c:
            return True, session_key.port2
        else:
            return False, session_key.port1
    
    def _calculate_pair_ratio(self, events: List[MessageEvent]) -> float:
        """计算请求-响应配对比例"""
        if len(events) < 2:
            return 0.0
        
        pairs = 0
        i = 0
        while i < len(events) - 1:
            if events[i].direction != events[i+1].direction:
                pairs += 1
                i += 2
            else:
                i += 1
        
        return pairs / (len(events)/2) if len(events) >= 2 else 0.0
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _calculate_burstiness(self, intervals: List[float]) -> float:
        """计算突发性（标准差/均值）"""
        if not intervals:
            return 0.0
        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return 0.0
        std = self._calculate_std(intervals)
        return std / mean
    
    def _is_request_response_pattern(self, events: List[MessageEvent]) -> bool:
        """判断是否为请求-响应模式"""
        if len(events) < self.MIN_EVENTS_FOR_PATTERN:
            return False
        
        alternations = sum(1 for i in range(len(events)-1) if events[i].direction != events[i+1].direction)
        return alternations / (len(events)-1) > 0.5
    
    def _is_streaming_pattern(self, events: List[MessageEvent], avg_interval: float) -> bool:
        """判断是否为流式传输模式"""
        if len(events) < self.MIN_EVENTS_FOR_PATTERN or avg_interval == 0:
            return False
        lengths = [len(e.payload) for e in events]
        if len(lengths) > 1:
            std = self._calculate_std(lengths)
            mean = sum(lengths) / len(lengths)
            cv = std / mean if mean > 0 else 0
            return avg_interval < 0.1 and cv < 0.5
        return False
    
    def _detect_protocol(self, events: List[MessageEvent], session_key: SessionKey) -> str:
        """基于特征检测协议类型"""
        if session_key.protocol != "TCP" or not events:
            return "Unknown"
        
        first_payload = events[0].payload
        if not first_payload:
            return "Unknown"
        
        if session_key.port1 in (80, 443) or session_key.port2 in (80, 443):
            return "HTTP" if 80 in (session_key.port1, session_key.port2) else "HTTPS"
        if session_key.port1 in (502,) or session_key.port2 in (502,):
            return "Modbus"
        
        return "Unknown"