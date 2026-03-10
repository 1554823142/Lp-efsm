from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionContext:
    '''
        会话级别上下文信息, 描述整个session的宏观特征
    '''

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
