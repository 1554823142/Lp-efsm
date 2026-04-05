from collections import defaultdict
from typing import Dict, List, Optional, Callable, Tuple
from protocol_infer.core.model.fsm import FSM, Transition
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.algorithm.guard_action import GuardActionLearner
from protocol_infer.algorithm.guard_action import IntervalDeltaLearner
from protocol_infer.algorithm.guard_action.cross_message import CrossMessageLearner

class EFSMInferencer:
    """
    EFSM 推断模块
    根据 FSM + 会话变量序列，生成 EFSM、guard 和 action。
    """

    def __init__(
        self,
        learner: Optional[GuardActionLearner] = None,
        cross_learner: Optional[CrossMessageLearner] = None,       # 跨消息守卫
    ):
        self.learner = learner if learner is not None else IntervalDeltaLearner()
        self.cross_learner = cross_learner if cross_learner is not None else CrossMessageLearner()

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
        # transition_vars: {(src状态, symbol) : vars} 单条消息变量列表
        transition_vars: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
        # 相邻消息配对列表    {(src状态, symbol) : (rc.vars, dst.vars)}
        transition_pairs: Dict[tuple, List[Tuple[Dict[str, float], Dict[str, float]]]] = defaultdict(list)
        
        all_vars: set = set(variable_names) if variable_names is not None else set()

        for session_key, pairs in sequences.items():
            current_state = fsm.start_state
            prev_vars: Optional[Dict[str, float]] = None

            # 沿着fsm遍历, 记录每个转移的变量序列
            for symbol, vars_dict in pairs:
                if variable_names is None:
                    all_vars.update(vars_dict.keys())
                candidates = fsm.get_transitions(current_state, symbol)
                if not candidates:
                    break
                tran = candidates[0]
                transition_vars[(tran.src, tran.symbol)].append(vars_dict)
                if prev_vars is not None:
                    transition_pairs[(tran.src, tran.symbol)].append((prev_vars, vars_dict))        # 收集相邻消息对
                current_state = tran.dst
                prev_vars = vars_dict       # 记录上一条消息

        # guard/action学习
        for tran in efsm.transitions:
            var_instances = transition_vars.get((tran.src, tran.symbol), [])
            pair_instances = transition_pairs.get((tran.src, tran.symbol), [])

            base_guard = None
            base_action = None
            if var_instances:
                base_guard, base_action = self.learner.learn(var_instances)

            memory_guard, memory_action = (
                self.cross_learner.learn(pair_instances) if pair_instances else (None, None)
            )

            if base_guard is None and base_action is None and memory_guard is None:
                continue

            def wrapped_guard(
                vars: Dict[str, float],
                memory=None,
                _base_guard=base_guard,
                _memory_guard=memory_guard,
            ) -> bool:
                # Layer1: 单消息内部
                if _base_guard is not None and not _base_guard(vars):
                    return False

                # Layer2: 跨消息
                # 无跨消息或第一条消息则直接通过
                if _memory_guard is None:
                    return True
                if memory is None:
                    return True
                mem = memory.data if hasattr(memory, "data") else memory
                return _memory_guard(vars, mem)

            def wrapped_action(
                vars: Dict[str, float],
                memory=None,
                _base_action=base_action,
                _memory_action=memory_action,
            ) -> Dict[str, float]:
                # 单消息内部更新
                new_vars = _base_action(vars.copy()) if _base_action is not None else vars.copy()

                # 写入 memory（全量覆盖，原始变量值；供跨消息 guard 使用）
                if memory is not None:
                    mem = memory.data if hasattr(memory, "data") else memory
                    if hasattr(mem, "update"):
                        mem.update(vars)
                    # 若有跨消息 action，执行它（当前实现与 mem.update 等价，接口预留扩展）
                    if _memory_action is not None:
                        _memory_action(vars, memory)
                return new_vars

            efsm.register_guard_action(tran, wrapped_guard, wrapped_action)

        efsm.variable_defs = all_vars
        return efsm
