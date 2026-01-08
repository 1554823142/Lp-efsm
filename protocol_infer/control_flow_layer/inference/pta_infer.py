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

        # 创建唯一初始state
        start = fsm.new_state(is_start=True)
        fsm.start_state = start

        # For each session's symbol sequence, extend the PTA
        for session_key, seq in sequences.items():
            current = fsm.start_state

            # track visited states for this sequence to update visit_count once per sequence
            visited_states = set()
            # 避免同一序列重复计数同一状态
            if current not in visited_states:
                fsm.states[current].visit()
                visited_states.add(current)

            for symbol in seq:
                key = (current, symbol)
                existing = fsm._by_state_input.get(key)

                if existing:        # 转移已经存在
                    # PTA的确定性
                    tran = existing[0]
                    dst = tran.dst
                else:
                    # create new state and transition
                    dst = fsm.new_state()
                    tran = Transition(
                        id=len(fsm.transitions),        # transition列表的长度递增性直接作为id
                        src=current,    
                        dst=dst,
                        symbol=symbol,
                        guard=None,
                        action=None
                    )
                    fsm.transitions.append(tran)
                    fsm._by_state_input.setdefault(key, []).append(tran)

                    # 更新前驱后继
                    fsm.states[current].next_states[symbol] = dst
                    fsm.states[dst].prev_states[symbol] = current
                    fsm.states[current].add_transition(tran)

                current = dst

                if current not in visited_states:
                    fsm.states[current].visit()
                    visited_states.add(current)

            # mark accepting (end) state for this sequence
            fsm.states[current].is_end = True
            print(f"[PTA] session={session_key} end_state={current}")

            print("check end_states:")
            for sid, s in fsm.states.items():
                if s.is_end:
                    print("end_state:", sid)
            
        return fsm