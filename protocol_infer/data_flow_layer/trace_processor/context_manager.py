from protocol_infer.core.datamodel.context import SessionContext
from typing import TYPE_CHECKING, Dict, List, Optional
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.event import MessageEvent, Direction
from collections import defaultdict
import math

if TYPE_CHECKING:
    from protocol_infer.algorithm.field_detection.dynamic_field_detector import DynamicField


class ContextExtractor:
    def __init__(
        self,
        byte_positions: Optional[List[int]] = None,
        static_items: Optional[Dict[int, int]] = None,
        dynamic_fields: Optional[List["DynamicField"]] = None,
    ):
        self.byte_positions = byte_positions if byte_positions is not None else [0, 1]
        self.static_items = static_items or {}
        self.dynamic_fields: List["DynamicField"] = dynamic_fields or []

    def extract_vars(self, event: MessageEvent) -> Dict[str, float]:
        payload = event.payload or b""

        vars_dict: Dict[str, float] = {
            "len": float(len(payload)),
            "direction": float(event.direction.to_feature()),
            "entropy": self._calculate_entropy(payload),
        }

        for pos in self.byte_positions:
            vars_dict[f"b{pos}"] = float(payload[pos]) if pos < len(payload) else 0.0
        for pos in sorted(self.static_items.keys()):        # 静态偏移pos, 生成变量名s{pos}
            vars_dict[f"s{pos}"] = float(payload[pos]) if pos < len(payload) else 0.0

        # 动态字段（单字节/多字节），变量名格式：dyn_{pos} 或 dyn_{pos}_{width}{b/l}
        for field in self.dynamic_fields:
            val = field.extract_value(payload)
            vars_dict[field.var_name] = val if val is not None else 0.0

        return vars_dict
    
    def _calculate_entropy(self, data: bytes) -> float:
        """计算字节序列的熵"""
        if len(data) == 0:
            return 0.0
        
        # 计算每个字节的频率
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1
        
        # 计算熵
        entropy = 0.0
        length = len(data)
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy


class SessionContextBuilder:

    def __init__(self, server_ports: Optional[set] = None):
        self.server_ports = server_ports or {502, 20000, 2404}
    
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
        packet_ratio = len(c2s_events) / len(s2c_events) if len(s2c_events) > 0 else float('inf')
        byte_ratio = sum(len(e.payload) for e in c2s_events) / sum(len(e.payload) for e in s2c_events) if s2c_events else float('inf')
        pair_ratio = self._calculate_pair_ratio(events)
        
        # 协议特征
        lengths = [len(e.payload) for e in events]
        avg_message_len = sum(lengths) / len(lengths) if lengths else 0.0
        message_len_std = self._calculate_std(lengths) if len(lengths) > 1 else 0.0
        total_messages = len(events)
        
        # 时序特征
        intervals = [events[i+1].timestamp - events[i].timestamp 
                    for i in range(len(events)-1)] if len(events) > 1 else []
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        burstiness = self._calculate_burstiness(intervals) if intervals else 0.0
        
        # 协议模式识别
        is_request_response = self._is_request_response_pattern(events)
        is_streaming = self._is_streaming_pattern(events, avg_interval) # (平均间隔不小，长度变化大)
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
    
    def _determine_direction(
        self,
        session_key: SessionKey,
        c2s_events: List[MessageEvent],
        s2c_events: List[MessageEvent],
    ) -> tuple:
        port1 = session_key.port1
        port2 = session_key.port2

        port1_is_server = port1 in self.server_ports
        port2_is_server = port2 in self.server_ports

        if port1_is_server and not port2_is_server:
            server_port = port1
        elif port2_is_server and not port1_is_server:
            server_port = port2
        else:
            server_port = min(port1, port2)

        is_client = server_port != port1
        return is_client, server_port
    
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
        
        return pairs / (len(events) / 2) if len(events) > 0 else 0.0
    
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
        if len(events) < 2:
            return False
        
        # 检查是否频繁出现方向交替
        alternations = 0
        for i in range(len(events) - 1):
            if events[i].direction != events[i+1].direction:
                alternations += 1
        
        return alternations / len(events) > 0.5 if len(events) > 0 else False
    
    def _is_streaming_pattern(self, events: List[MessageEvent], avg_interval: float) -> bool:
        """判断是否为流式传输模式"""
        # 如果平均间隔很小且消息长度变化不大，可能是流式
        if avg_interval < 0.1:  # 小于100ms
            lengths = [len(e.payload) for e in events]
            if len(lengths) > 1:
                std = self._calculate_std(lengths)
                mean = sum(lengths) / len(lengths)
                cv = std / mean if mean > 0 else 0  # 变异系数
                return cv < 0.5  # 长度变化小
        return False
    
    def _detect_protocol(self, events: List[MessageEvent], session_key: SessionKey) -> str:
        """未知协议场景：不做具体协议命名，统一返回 Unknown"""
        return "Unknown"
