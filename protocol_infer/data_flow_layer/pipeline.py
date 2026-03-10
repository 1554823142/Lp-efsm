
from collections import defaultdict
from typing import Dict, List, Optional

from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.fsm import FSM
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.interface.message_abstraction import MessageAbstractor
from protocol_infer.core.interface.feature_extractor import FeatureExtractor
from protocol_infer.core.datamodel.abstract_message import AbstractMessage

from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor
from protocol_infer.data_flow_layer.abstraction.trace_abstractor import AbstractionProcessor
from protocol_infer.data_flow_layer.inference.efsm_infer import EFSMInferencer


class DataFlowPipeline:


    def __init__(
        self,
        abstractor: Optional[MessageAbstractor] = None,
        symbol_featureer: Optional[FeatureExtractor] = None,
        feature_processor: Optional[FeatureProcessor] = None,
    ):
        """
        数据流层管道：
            1) 从 Trace / sessions 中提取消息级变量与会话上下文（FeatureProcessor）
            2) 抽象变量 → 符号，写入 Trace.abstract_messages（AbstractionProcessor）
            3) 基于 FSM + 变量序列 构建 EFSM（EFSMInferencer）
        """
        self.feature_processor = feature_processor or FeatureProcessor()    # 提取特征, 供EFSM的guard学习
        self.abstraction_processor = AbstractionProcessor(abstractor)       # 特征向量 -> symbol(保证数据/控制流层一致)
        self.symbol_featureer = symbol_featureer or FeatureProcessor()      # 复用控制流层的特征提取器以生成symbol
        self.efsm_inferencer = EFSMInferencer()                             # 在FSM的基础上学习guard和action


    def _prepare_sessions(
        self,
        trace: Trace,
        sessions: Optional[Dict[SessionKey, List]] = None
    ) -> Dict[SessionKey, List]:
        '''
            统一会话准备逻辑：
                - 如果上层已经按 SessionKey 分好组（如 ControlFlowPipeline），直接使用
                - 否则使用 FeatureProcessor 重新分组
        '''

        return self.feature_processor.prepare_sessions(trace, sessions)

    def run(
        self,
        trace: Trace,
        fsm: FSM,
        sessions: Optional[Dict[SessionKey, List]] = None,
        precomputed_sess_features: Optional[Dict[SessionKey, tuple]] = None,
        apriori_positions: Optional[List[int]] = None
    ) -> EFSM:
        """
        从 Trace(增强版) + FSM 构建 EFSM。
        增强版 Trace 包含：
            - abstract_messages：抽象消息序列（由 AbstractionProcessor 写入）
            - session_contexts：会话级上下文（由 FeatureProcessor 写入）
            
        apriori_positions: 从控制流层传入的有效载荷偏移位置，如果提供则会更新 feature_processor
        """
        
        # 如果提供了新的偏移位置，重新初始化 FeatureProcessor
        if apriori_positions is not None:
            self.feature_processor = FeatureProcessor(apriori_positions=apriori_positions)

        sessions = self._prepare_sessions(trace, sessions)

        # 1) 构建会话上下文（写入 trace.session_contexts）
        self.feature_processor.build_session_contexts(trace, sessions)

        # 2) 抽象消息（写入 trace.abstract_messages）
        # 若提供了已训练的抽象器与其对应的特征提取器，则复用以保证符号与FSM一致
        if self.abstraction_processor.abstractor is not None and (self.symbol_featureer is not None or precomputed_sess_features is not None):
            abstract_msgs = []
            if precomputed_sess_features is not None:
                # 直接复用控制流层已计算好的符号特征
                for events, symbol_features in precomputed_sess_features.values():
                    for ev, feat in zip(events, symbol_features):
                        vars_dict = self.feature_processor.extract_vars(ev)
                        symbol = self.abstraction_processor.abstractor.abstract(feat)   # feature->symbol(string类型), 只abstract, 不fit
                        abstract_msgs.append(
                            AbstractMessage(
                                session_key=ev.session_key,
                                timestamp=ev.timestamp,
                                symbol=symbol,
                                vars=vars_dict,             # 单条消息级别的变量,分析哪些变量取值规律可以作为转移的触发条件
                                direction=ev.direction
                            )
                        )
            else:
                for events in sessions.values():
                    # 使用控制流层的特征提取器生成符号特征
                    symbol_features = self.symbol_featureer.extract(events)
                    for ev, feat in zip(events, symbol_features):
                        vars_dict = self.feature_processor.extract_vars(ev)
                        symbol = self.abstraction_processor.abstractor.abstract(feat)
                        abstract_msgs.append(
                            AbstractMessage(
                                session_key=ev.session_key,
                                timestamp=ev.timestamp,
                                symbol=symbol,
                                vars=vars_dict,
                                direction=ev.direction
                            )
                        )
            trace.abstract_messages = abstract_msgs     # 写入trace, 供EFSMInferencer使用
        else:               # 重新训练abstractor
            trace = self.abstraction_processor.fit_and_abstract(
                trace, sessions, self.feature_processor
            )

        # 3) 构建 EFSM：根据 FSM + 每个会话的变量序列
        sequences: Dict[SessionKey, List[tuple]] = defaultdict(list)
        for ev in trace.abstract_messages:
            sequences[ev.session_key].append((ev.symbol, ev.vars))

        fsm_dfa = fsm.determinize()         # 确定性化 FSM
        efsm = self.efsm_inferencer.build_efsm(fsm_dfa, sequences, variable_names=self.feature_processor.var_names())
        return efsm
