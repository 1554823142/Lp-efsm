from collections import defaultdict
from typing import Dict, List, Optional
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.data_flow_layer.trace_processor.context_manager import ContextExtractor, SessionContextBuilder

class FeatureProcessor:

    _BASE_VARS = ["len", "direction", "entropy"]

    def __init__(self, apriori_positions: Optional[List[int]] = None):
        self.ctx_extractor = ContextExtractor()
        self.session_context_builder = SessionContextBuilder()
        
        # 用 Apriori 发现的偏移替代硬编码的 b0, b1
        # 如果未提供，默认为 [0, 1] 以兼容旧逻辑
        self.apriori_positions = apriori_positions if apriori_positions is not None else [0, 1]
        
        # 动态构建变量顺序
        self._var_order = self._BASE_VARS + [f"b{p}" for p in self.apriori_positions]

    def prepare_sessions(self, trace: Trace, sessions: Optional[Dict[SessionKey, List]] = None) -> Dict[SessionKey, List]:
        if sessions is None:
            sessions = defaultdict(list)
            for ev in trace.events:
                sessions[ev.session_key].append(ev)

        for events in sessions.values():
            events.sort(key=lambda e: e.timestamp)

        return dict(sessions)

    def extract_vars(self, ev) -> Dict:
        # 1. 提取基础变量 (len, direction, entropy)
        base_vars = self.ctx_extractor.extract_vars(ev)
        
        # 2. 提取指定偏移位置的字节值
        payload = ev.payload or b""
        for pos in self.apriori_positions:
            # 提取 payload[pos]，如果越界则补 0.0
            val = float(payload[pos]) if pos < len(payload) else 0.0
            base_vars[f"b{pos}"] = val
            
        return base_vars

    def var_list(self, vars_dict: Dict) -> List[float]:
        return [vars_dict.get(k, 0.0) for k in self._var_order]

    def build_session_contexts(self, trace: Trace, sessions: Dict[SessionKey, List]):
        session_contexts = {sk: self.session_context_builder.build(events) for sk, events in sessions.items()}
        trace.session_contexts = session_contexts        # 将上下文添加到trace
        return session_contexts
