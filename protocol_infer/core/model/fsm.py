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
                transitions=[]):
        self.name = name
        self.is_start = is_start
        self.is_end = is_end

        # 记录前驱后继, 方便后续的计算
        self.next_states = {}      # symbol -> FSMState
        self.prev_states = {}      # symbol -> FSMState

        # 统计信息（为 merge / coverage 服务）
        self.visit_count = 0        # 出现于多少条序列, 相当于refsm的no_of_seqs

        self.hasNo = hasNo          # 状态的哈希编号（用于状态合并）

        self.transitions = transitions      # 该状态的所有出边（转移）列表

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

    def new_state(self, is_start=False, is_end=False):
        sid = self._next_state_id
        self._next_state_id += 1

        self.states[sid] = FSMState(
            name=f"s{sid}",
            is_start=is_start,
            is_end=is_end
        )

        return sid
    