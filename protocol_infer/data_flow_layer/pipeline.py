
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.core.model.fsm import FSM
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.interface.message_abstraction import MessageAbstractor
from protocol_infer.core.interface.feature_extractor import FeatureExtractor
from protocol_infer.core.datamodel.abstract_message import AbstractMessage

from protocol_infer.core.algorithm.guard_action import GuardActionLearner
from protocol_infer.algorithm.guard_action import AprioriGuardLearner
from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor
from protocol_infer.data_flow_layer.abstraction.trace_abstractor import AbstractionProcessor
from protocol_infer.data_flow_layer.inference.efsm_infer import EFSMInferencer
from protocol_infer.algorithm.field_detection.dynamic_field_detector import DynamicFieldDetector


class DataFlowPipeline:

    def __init__(
        self,
        abstractor: Optional[MessageAbstractor] = None,
        symbol_featureer: Optional[FeatureExtractor] = None,
        feature_processor: Optional[FeatureProcessor] = None,
        guard_action_learner: Optional[GuardActionLearner] = None,
    ):
        """
        数据流层管道：
            1) 从 Trace / sessions 中提取消息级变量与会话上下文（FeatureProcessor）
            2) 抽象变量 → 符号，写入 Trace.abstract_messages（AbstractionProcessor）
            2.5) 动态字段检测：在 Apriori 已知位置基础上扫描被遗漏的结构化动态字段
            3) 基于 FSM + 变量序列 构建 EFSM（EFSMInferencer）
        """
        self.feature_processor = feature_processor or FeatureProcessor()    # 提取特征, 供EFSM的guard学习
        self.abstraction_processor = AbstractionProcessor(abstractor)       # 特征向量 -> symbol(保证数据/控制流层一致)
        self.symbol_featureer = symbol_featureer or FeatureProcessor()      # 复用控制流层的特征提取器以生成symbol
        self.efsm_inferencer = EFSMInferencer(
            learner=guard_action_learner if guard_action_learner is not None else AprioriGuardLearner()
        )                             # 在FSM的基础上学习guard和action

    # ------------------------------------------------------------------
    # 内部辅助：收集 (event, symbol) 对
    # ------------------------------------------------------------------

    def _collect_ev_sym_pairs(
        self,
        sessions: Dict[SessionKey, List[MessageEvent]],
        precomputed_sess_features: Optional[Dict[SessionKey, tuple]],
    ) -> List[Tuple[MessageEvent, str]]:
        """
        第一趟遍历：仅获取 (event, symbol) 对，不提取变量。
        这样在检测到动态字段、更新 feature_processor 之后，
        第二趟才用最新的 feature_processor 提取完整变量向量。
        """
        pairs: List[Tuple[MessageEvent, str]] = []
        if precomputed_sess_features is not None:
            for events, symbol_features in precomputed_sess_features.values():
                for ev, feat in zip(events, symbol_features):
                    symbol = self.abstraction_processor.abstractor.abstract(feat)
                    pairs.append((ev, symbol))
        else:
            for events in sessions.values():
                symbol_features = self.symbol_featureer.extract(events)
                for ev, feat in zip(events, symbol_features):
                    symbol = self.abstraction_processor.abstractor.abstract(feat)
                    pairs.append((ev, symbol))
        return pairs

    # ------------------------------------------------------------------
    # 内部辅助：动态字段检测并更新 feature_processor
    # ------------------------------------------------------------------

    def _detect_and_update_dynamic_fields(
        self,
        ev_sym_pairs: List[Tuple[MessageEvent, str]],
        known_positions: Optional[List[int]],
    ) -> None:
        """
        按 symbol 分组事件，调用 DynamicFieldDetector，
        若检测到新字段则更新 feature_processor（变量顺序同步重建）。
        """
        symbol_events: Dict[str, List[MessageEvent]] = defaultdict(list)
        for ev, sym in ev_sym_pairs:
            symbol_events[sym].append(ev)

        detector = DynamicFieldDetector()
        # 同时纳入动态位置（apriori_positions）和静态位置（apriori_static_items.keys()）
        # 两类位置都已被 Apriori 解释，不应被 DynamicFieldDetector 重复处理。
        # 若只纳入动态位置（原逻辑），静态字节（如 b0=s0=0x00）不在 known_positions 中，
        # 导致包含静态字节的多字节滑动窗口（如 dyn_0_2b = b0-b1）被误判为新字段。
        dyn_positions = set(
            known_positions if known_positions is not None
            else self.feature_processor.apriori_positions
        )
        static_positions = set(self.feature_processor.apriori_static_items.keys())
        base_positions = dyn_positions | static_positions
        dynamic_fields = detector.detect_from_symbol_groups(
            symbol_events=dict(symbol_events),
            known_positions=base_positions,
        )
        if dynamic_fields:
            self.feature_processor.update_dynamic_fields(dynamic_fields)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

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
        apriori_positions: Optional[List[int]] = None,
        apriori_static_items: Optional[Dict[int, int]] = None,
    ) -> EFSM:
        """
        从 Trace(增强版) + FSM 构建 EFSM。

        apriori_positions / apriori_static_items:
            从控制流层传入的有效载荷偏移位置与静态字节值，
            如果提供则重建 feature_processor（在此基础上再做动态字段增量检测）。
        """
        if apriori_positions is not None or apriori_static_items is not None:
            self.feature_processor = FeatureProcessor(
                apriori_positions=apriori_positions,
                apriori_static_items=apriori_static_items,
            )

        sessions = self._prepare_sessions(trace, sessions)

        # 1) 构建会话上下文（写入 trace.session_contexts）
        self.feature_processor.build_session_contexts(trace, sessions)

        # 2) 抽象消息（写入 trace.abstract_messages）
        if self.abstraction_processor.abstractor is not None and (
            self.symbol_featureer is not None or precomputed_sess_features is not None
        ):
            # ── 第一趟：仅获取 (event, symbol) 对 ──────────────────
            ev_sym_pairs = self._collect_ev_sym_pairs(sessions, precomputed_sess_features)

            # ── 步骤 2.5：动态字段检测，增量更新 feature_processor ──
            self._detect_and_update_dynamic_fields(ev_sym_pairs, apriori_positions)

            # ── 第二趟：用更新后的 feature_processor 提取完整变量 ───
            abstract_msgs = [
                AbstractMessage(
                    session_key=ev.session_key,
                    timestamp=ev.timestamp,
                    symbol=sym,
                    vars=self.feature_processor.extract_vars(ev),
                    direction=ev.direction,
                )
                for ev, sym in ev_sym_pairs
            ]
            trace.abstract_messages = abstract_msgs

        else:
            # ── 回退路径：重新训练 abstractor ────────────────────────
            trace = self.abstraction_processor.fit_and_abstract(
                trace, sessions, self.feature_processor
            )
            # 对回退路径同样做动态字段检测并更新 vars
            if trace.abstract_messages:
                ev_by_key = {
                    (e.session_key, e.timestamp): e for e in trace.events
                }
                ev_sym_pairs = [
                    (ev_by_key[(m.session_key, m.timestamp)], m.symbol)
                    for m in trace.abstract_messages
                    if (m.session_key, m.timestamp) in ev_by_key
                ]
                self._detect_and_update_dynamic_fields(ev_sym_pairs, apriori_positions)
                if self.feature_processor.dynamic_fields:
                    for msg in trace.abstract_messages:
                        ev = ev_by_key.get((msg.session_key, msg.timestamp))
                        if ev is not None:
                            msg.vars = self.feature_processor.extract_vars(ev)

        # 3) 构建 EFSM：根据 FSM + 每个会话的变量序列
        sequences: Dict[SessionKey, List[tuple]] = defaultdict(list)
        for ev in trace.abstract_messages:
            sequences[ev.session_key].append((ev.symbol, ev.vars))

        fsm_dfa = fsm.determinize()         # 确定性化 FSM
        efsm = self.efsm_inferencer.build_efsm(
            fsm_dfa, sequences,
            variable_names=self.feature_processor.var_names()
        )
        return efsm
