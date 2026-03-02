from dataclasses import dataclass
from typing import List, Dict
from protocol_infer.core.datamodel.event import MessageEvent, SessionKey
from typing import Optional
from protocol_infer.core.datamodel.abstract_message import AbstractMessage
from protocol_infer.core.datamodel.context import SessionContext

@dataclass
class Trace:
    events: List[MessageEvent]

    # 拓展EFSM的输入
    abstract_messages: Optional[List[AbstractMessage]] = None            # 离散符号, 变量序列
    session_contexts: Optional[Dict[SessionKey, SessionContext]] = None  # 存储的上下文
