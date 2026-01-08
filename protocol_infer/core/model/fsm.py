from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Iterable


@dataclass
class Transition:
    id: int
    src: int
    dst: int
    symbol: str
    guard: Optional[Callable[[Dict], bool]]  # guard(vars) -> bool
    action: Optional[Callable[[Dict], Dict]]  # action(vars) -> new_vars
    output: Optional[str] = None
    prob: Optional[float] = None  # for P-EFSM


class FSMState:
    '''
        FSM状态定义
    '''
    def __init__(self, name: str, is_start=False, is_end=False, 
                hasNo: Optional[int]=None,
                transitions: Optional[List[Transition]]=None):
        self.name = name
        self.is_start = is_start
        self.is_end = is_end

        # 记录前驱后继, 方便后续的计算
        self.next_states = {}      # symbol -> FSMStateID
        self.prev_states = {}      # symbol -> FSMStateID

        # 统计信息（为 merge / coverage 服务）
        self.visit_count = 0        # 出现于多少条序列, 相当于refsm的no_of_seqs

        self.hasNo = hasNo          # 状态的哈希编号（用于状态合并）

        self.transitions = [] if transitions is None else transitions      # 该状态的所有出边（转移）列表

    def visit(self):
        self.visit_count += 1

    def add_transition(self, tran: Transition):
        self.transitions.append(tran)

    
            

class FSM:
    def __init__(self):
        self.states: Dict[int, FSMState] = {}
        self.start_state: Optional[int] = None
        self._next_state_id = 0
        self.transitions: List[Transition] = []
        self._by_state_input: Dict[Tuple[int, str], List[Transition]] = {}  # (源状态ID, 输入符号) → 转移列表

    def __str__(self) -> str:
        lines = []
        lines.append("==== FSM Summary ====")
        lines.append(f"States: {len(self.states)}")
        lines.append(f"Transitions: {len(self.transitions)}")
        lines.append(f"Start state: {self.start_state}")

        end_states = [
            sid for sid, s in self.states.items() if s.is_end
        ]
        
        lines.append(f"End states: {end_states}")
        lines.append("")

        lines.append("---- States ----")
        for sid, state in self.states.items():
            flags = []
            if state.is_start:
                flags.append("START")
            if state.is_end:
                flags.append("END")

            flag_str = f" ({', '.join(flags)})" if flags else ""
            lines.append(
                f"[{sid}] {state.name}{flag_str}, "
                f"visits={state.visit_count}"
            )

            for tran in state.transitions:
                lines.append(
                    f"    --[{tran.symbol}]--> {tran.dst}"
                )

        return "\n".join(lines)

    def new_state(self, is_start=False, is_end=False):
        sid = self._next_state_id
        self._next_state_id += 1

        self.states[sid] = FSMState(
            name=f"s{sid}",
            is_start=is_start,
            is_end=is_end
        )

        return sid
    

    def merge_two_state(self, s1, s2):
        '''
            将s1 s2(都为sid)合并
        '''

        if s1 == s2:
            return

        state1 = self.states[s1]
        state2 = self.states[s2]

        if state1.is_end or state2.is_end:
            # 避免合并 end 状态
            return

        # 合并s2信息到s1(统计信息)
        state1.visit_count += state2.visit_count
        state1.is_end = state1.is_end or state2.is_end

        # 修改state2的transition
        for tran in state2.transitions:
            tran.src = s1
            state1.add_transition(tran)

        # 修改所有指向state2的transition
        for tran in self.transitions:
            if tran.dst == s2:
                tran.dst = s1

        # 合并前驱后继
        for symbol, pid in state2.prev_states.items():
            state1.prev_states[symbol] = pid

        for symbol, nid in state2.next_states.items():
            state1.next_states[symbol] = nid

        # 检查s2是否为start_state
        if self.start_state == s2:
            self.start_state = s1
            state1.is_start = True

            
        # 删除state2
        del self.states[s2]

        self._by_state_input.clear()
        for tran in self.transitions:
            key = (tran.src, tran.symbol)
            self._by_state_input.setdefault(key, []).append(tran)