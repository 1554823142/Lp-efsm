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
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
        variable_names: Optional[List[str]] = None,
    ) -> EFSM:

        '''
            传入的fsm已经是确定的fsm, 所以不再需要再选用visit最大的转移路径
        '''
        efsm = EFSM(base_fsm=fsm)
        # transition_vars: {(src状态, symbol) : vars}
        transition_vars: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
        all_vars: set = set(variable_names) if variable_names is not None else set()

        for session_key, pairs in sequences.items():
            current_state = fsm.start_state

            # 沿着fsm遍历, 记录每个转移的变量序列
            for symbol, vars_dict in pairs:
                if variable_names is None:
                    all_vars.update(vars_dict.keys())
                candidates = fsm.get_transitions(current_state, symbol)
                if not candidates:
                    break
                tran = candidates[0]
                transition_vars[(tran.src, tran.symbol)].append(vars_dict)
                current_state = tran.dst

        # guard/action学习
        for tran in efsm.transitions:
            var_instances = transition_vars.get((tran.src, tran.symbol), [])
            if var_instances:
                guard, action = self.learner.learn(var_instances)
                efsm.register_guard_action(tran, guard, action)

        efsm.variable_defs = all_vars
        return efsm
