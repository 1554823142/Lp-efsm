import os
from collections import defaultdict
from typing import Dict, List, Optional, Callable, Tuple, Any
import concurrent.futures
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
        # transition_session_vars: {(src状态, symbol) : {session_key: [vars]}} 按会话分组
        transition_session_vars: Dict[tuple, Dict[SessionKey, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
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
                transition_session_vars[(tran.src, tran.symbol)][session_key].append(vars_dict)
                if prev_vars is not None:
                    transition_pairs[(tran.src, tran.symbol)].append((prev_vars, vars_dict))        # 收集相邻消息对
                current_state = tran.dst
                prev_vars = vars_dict       # 记录上一条消息

        # 全局统计：识别会话相关和计数器变量
        volatile_vars = set()
        counter_vars = set()
        if len(sequences) > 1:
            for var_name in all_vars:
                session_values = defaultdict(list)
                for session_key, pairs in sequences.items():
                    for _, vars_dict in pairs:
                        if var_name in vars_dict:
                            session_values[session_key].append(vars_dict[var_name])
                
                # 1. 检测计数器：单会话内唯一值比例很高
                total_unique_ratio = 0
                for vals in session_values.values():
                    if len(vals) >= 2:
                        unique_ratio = len(set(vals)) / len(vals)
                        # 降低门槛：如果 60% 以上的包值不同，极可能是计数器
                        if unique_ratio > 0.6:
                            total_unique_ratio += 1
                if len(session_values) > 0 and total_unique_ratio / len(session_values) > 0.3:
                    counter_vars.add(var_name)

                # 2. 检测会话相关：单会话常数但跨会话不同
                const_sessions = 0
                global_values = set()
                for vals in session_values.values():
                    if len(set(vals)) == 1:
                        const_sessions += 1
                    global_values.update(vals)
                
                if len(session_values) > 1 and const_sessions / len(session_values) > 0.8 and len(global_values) > 1:
                    volatile_vars.add(var_name)
                    
        # 3. 汇总黑名单
        guard_blacklist = volatile_vars | counter_vars

        # 4. 全局统计：每个变量的取值分布（用于衡量信息增益）
        global_var_values = defaultdict(set)
        symbol_var_values = defaultdict(lambda: defaultdict(set))
        
        for seq in sequences.values():
            for symbol, vars_dict in seq:
                for name, val in vars_dict.items():
                    global_var_values[name].add(val)
                    symbol_var_values[symbol][name].add(val)
        
        var_global_diversity = {
            name: len(vals) for name, vals in global_var_values.items()
        }
        
        # 符号级多样性：反映变量在该 Symbol 下的判别能力
        var_symbol_diversity = {}
        for symbol, vars_map in symbol_var_values.items():
            var_symbol_diversity[symbol] = {
                name: len(vals) for name, vals in vars_map.items()
            }

        # 5. 符号全局常量统计：用于过滤无区分度的 guard
        symbol_global_vars = defaultdict(lambda: defaultdict(set))
        for (src, symbol), instances in transition_vars.items():
            for inst in instances:
                for var_name, val in inst.items():
                    symbol_global_vars[symbol][var_name].add(val)
        
        symbol_constants = {}
        for symbol, vars_map in symbol_global_vars.items():
            symbol_constants[symbol] = {}
            for var_name, values in vars_map.items():
                if len(values) == 1:
                    symbol_constants[symbol][var_name] = next(iter(values))

        # 准备待处理的任务数据
        tasks = []
        for tran in efsm.transitions:
            var_instances = transition_vars.get((tran.src, tran.symbol), [])
            session_vars = transition_session_vars.get((tran.src, tran.symbol), {})
            pair_instances = transition_pairs.get((tran.src, tran.symbol), [])
            if var_instances or pair_instances:
                context = {
                    "guard_blacklist": guard_blacklist,
                    "symbol_constants": symbol_constants.get(tran.symbol, {}),
                    "var_global_diversity": var_global_diversity,
                    "var_symbol_diversity": var_symbol_diversity.get(tran.symbol, {}),
                    "session_count": len(session_vars),
                    "total_sessions": len(sequences)
                }
                tasks.append((tran, var_instances, pair_instances, context))

        def process_transition(tran_info):
            tran, var_instances, pair_instances, context = tran_info
            base_guard = None
            base_action = None
            if var_instances:
                base_guard, base_action = self.learner.learn(var_instances, context=context)

            memory_guard, memory_action = (
                self.cross_learner.learn(pair_instances) if pair_instances else (None, None)
            )

            if base_guard is None and base_action is None and memory_guard is None:
                return tran, None, None

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
                new_vars = _base_action(vars.copy()) if _base_action is not None else vars.copy()
                if memory is not None:
                    mem = memory.data if hasattr(memory, "data") else memory
                    if hasattr(mem, "update"):
                        mem.update(vars)
                    if _memory_action is not None:
                        _memory_action(vars, memory)
                return new_vars

            return tran, wrapped_guard, wrapped_action

        # 使用线程池加速学习（因为学习逻辑主要是 CPU 密集型但受限于 Python GIL，
        # 在多核环境下，如果学习逻辑中有部分 IO 或在多进程模式下会更佳，
        # 考虑到代码复用性，先用多线程降低单线程循环延迟，如需极致性能可改用 ProcessPoolExecutor）
        # 鉴于 guard/action 学习包含大量数值计算，ProcessPoolExecutor 往往更适合 CPU 密集型。
        # 但因为 lambda 和内部函数序列化问题，改用 ThreadPoolExecutor 确保兼容性。
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            results = list(executor.map(process_transition, tasks))

        for tran, guard, action in results:
            if guard is not None or action is not None:
                efsm.register_guard_action(tran, guard, action)
                # 提取并存储 action 元数据用于评估
                if action is not None and hasattr(action, "metadata"):
                    tran.action_metadata = action.metadata

        efsm.variable_defs = all_vars
        return efsm
