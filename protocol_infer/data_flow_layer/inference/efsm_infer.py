from collections import defaultdict
from typing import Dict, List, Optional, Callable, Tuple
from protocol_infer.core.model.fsm import FSM, Transition
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.algorithm.guard_action import GuardActionLearner
from protocol_infer.algorithm.guard_action import IntervalDeltaLearner

class EFSMInferencer:
    """
    EFSM 推断模块
    根据 FSM + 会话变量序列，生成 EFSM、guard 和 action。
    """

    def __init__(self, learner: Optional[GuardActionLearner] = None):
        self.learner = learner if learner is not None else IntervalDeltaLearner()

    def build_efsm(
        self,
        fsm: FSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]
    ) -> EFSM:
        """
        构建 EFSM：
            - 收集每个转移的变量实例
            - 学习 guard/action
            - 返回带 guard/action 的 EFSM
        """
        efsm = EFSM(base_fsm=fsm)
        transition_vars: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)

        for session_key, pairs in sequences.items():
            current_state = fsm.start_state
            for symbol, vars_dict in pairs:
                candidates = fsm.get_transitions(current_state, symbol)
                if not candidates:
                    break
                tran = candidates[0]
                key = (tran.src, tran.symbol)
                transition_vars[key].append(vars_dict)
                current_state = tran.dst

        for tran in efsm.transitions:
            key = (tran.src, tran.symbol)
            var_instances = transition_vars.get(key, [])
            if var_instances:
                guard, action = self.learner.learn(var_instances)
                efsm.register_guard_action(tran, guard, action)

        if sequences:
            first_seq = next(iter(sequences.values()))
            if first_seq:
                efsm.variable_defs = set(first_seq[0][1].keys())

        return efsm

    def _learn_guard(self, var_instances: List[Dict[str, float]]) -> Optional[Callable]:
        guard, _ = self.learner.learn(var_instances)
        return guard

    def _learn_action(self, var_instances: List[Dict[str, float]]) -> Optional[Callable]:
        _, action = self.learner.learn(var_instances)
        return action
