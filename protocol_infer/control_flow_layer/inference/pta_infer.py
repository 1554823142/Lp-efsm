from typing import Dict, List
from protocol_infer.core.interface.fsm_infer import FSMInfer
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.fsm import FSM, Transition


class PTAInfer(FSMInfer):
    """
    Build a Prefix-Tree Acceptor (PTA) from given symbol sequences.
    特征:
        共享前缀
        确定转移
    Input:
        sequences: Dict[SessionKey, List[str]]
    Output:
        FSM instance representing the PTA
    """

    def infer(self, sequences: Dict[SessionKey, List[str]]) -> FSM:
        fsm = FSM()

        # 创建唯一初始 state（new_state 内部已处理 is_start 时设置 start_state）
        start = fsm.new_state(is_start=True)

        for session_key, seq in sequences.items():
            current = fsm.start_state
            visited_states = set()

            if current not in visited_states:
                fsm.states[current].visit()
                visited_states.add(current)

            for symbol in seq:
                key = (current, symbol)
                existing = fsm._by_state_input.get(key)

                if existing:
                    # 转移已存在，PTA 确定性：直接复用
                    dst = existing[0].dst
                    existing[0].traverse_count += 1
                else:
                    # 新建状态和转移，统一通过 add_transition 维护所有索引
                    dst = fsm.new_state()
                    tran = fsm.add_transition(src=current, dst=dst, symbol=symbol)
                    tran.traverse_count = 1

                current = dst

                if current not in visited_states:
                    fsm.states[current].visit()
                    visited_states.add(current)

            # 标记当前序列的终止状态
            fsm.states[current].is_end = True
            print(f"[PTA] session={session_key} end_state={current}")

        return fsm
