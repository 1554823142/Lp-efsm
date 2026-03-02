from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional


@dataclass
class Transition:
    id: int
    src: int
    dst: int
    symbol: str
    guard: Optional[Callable[[Dict], bool]] = None
    action: Optional[Callable[[Dict], Dict]] = None
    output: Optional[str] = None
    prob: Optional[float] = None


class FSMState:
    def __init__(
        self,
        name: str,
        is_start: bool = False,
        is_end: bool = False,
        hasNo: Optional[int] = None,
        transitions: Optional[List[Transition]] = None,
    ):
        self.name = name
        self.is_start = is_start
        self.is_end = is_end
        self.visit_count = 0
        self.hasNo = hasNo
        self.transitions: List[Transition] = [] if transitions is None else transitions
        # 由 FSM._register_transition 统一维护，外部不要直接写
        self.next_states: Dict[str, int] = {}   # symbol -> dst state id
        self.prev_states: Dict[str, int] = {}   # symbol -> src state id

    def visit(self):
        self.visit_count += 1

    def add_transition(self, tran: Transition):
        self.transitions.append(tran)


class FSM:
    def __init__(self):
        self.states: Dict[int, FSMState] = {}
        self.start_state: Optional[int] = None
        self._next_state_id = 0
        self._next_tran_id = 0
        self.transitions: List[Transition] = []
        self._by_state_input: Dict[Tuple[int, str], List[Transition]] = {}

    # ------------------------------------------------------------------ #
    # 核心接口                                                              #
    # ------------------------------------------------------------------ #

    def new_state(self, is_start: bool = False, is_end: bool = False) -> int:
        sid = self._next_state_id
        self._next_state_id += 1
        self.states[sid] = FSMState(name=f"s{sid}", is_start=is_start, is_end=is_end)
        if is_start:
            self.start_state = sid
        return sid

    def add_transition(
        self,
        src: int,
        dst: int,
        symbol: str,
        guard: Optional[Callable] = None,
        action: Optional[Callable] = None,
        output: Optional[str] = None,
        prob: Optional[float] = None,
    ) -> Transition:
        """
        统一的转移添加入口，同时维护：
          - self.transitions
          - self._by_state_input
          - FSMState.transitions
          - FSMState.next_states / prev_states
        """
        tid = self._next_tran_id
        self._next_tran_id += 1
        tran = Transition(
            id=tid, src=src, dst=dst, symbol=symbol,
            guard=guard, action=action, output=output, prob=prob
        )
        self._register_transition(tran)
        return tran

    def _register_transition(self, tran: Transition) -> None:
        """内部注册，保证所有索引同步"""
        self.transitions.append(tran)
        self._by_state_input.setdefault((tran.src, tran.symbol), []).append(tran)
        if tran.src in self.states:
            self.states[tran.src].add_transition(tran)
            self.states[tran.src].next_states[tran.symbol] = tran.dst
        if tran.dst in self.states:
            self.states[tran.dst].prev_states[tran.symbol] = tran.src

    # ------------------------------------------------------------------ #
    # 状态合并                                                              #
    # ------------------------------------------------------------------ #

    def merge_two_state(self, s1: int, s2: int) -> None:
        """
        将 s2 合并入 s1：
          - 允许 s1 是 end（s2 非 end）：保留 end 属性
          - 拒绝将 end 状态合并入非 end 状态（避免丢失终止语义）
        """
        if s1 == s2:
            return
        if s1 not in self.states or s2 not in self.states:
            return

        state1 = self.states[s1]
        state2 = self.states[s2]

        if state2.is_end and not state1.is_end:
            return

        # 1. 合并统计信息
        state1.visit_count += state2.visit_count
        if state2.is_end:
            state1.is_end = True

        # 2. 将 s2 的出边改为从 s1 出发
        for tran in state2.transitions:
            tran.src = s1
            state1.transitions.append(tran)
            self._by_state_input.setdefault((s1, tran.symbol), []).append(tran)
            state1.next_states[tran.symbol] = tran.dst
            if tran.dst in self.states:
                self.states[tran.dst].prev_states[tran.symbol] = s1

        # 3. 将所有指向 s2 的入边改为指向 s1
        for tran in self.transitions:
            if tran.dst == s2:
                tran.dst = s1
                if tran.src in self.states:
                    self.states[tran.src].next_states[tran.symbol] = s1
                state1.prev_states[tran.symbol] = tran.src

        # 4. s2 是 start_state 时转移到 s1
        if self.start_state == s2:
            self.start_state = s1
            state1.is_start = True

        # 5. 清理 _by_state_input 中以 s2 为 src 的旧条目
        for k in [k for k in self._by_state_input if k[0] == s2]:
            del self._by_state_input[k]

        # 6. 对合并后 (s1, symbol) 的转移列表去重
        for key in list(self._by_state_input):
            if key[0] == s1:
                seen: set = set()
                self._by_state_input[key] = [
                    t for t in self._by_state_input[key]
                    if t.id not in seen and not seen.add(t.id)  # type: ignore
                ]

        # state1.transitions 同步去重
        seen = set()
        state1.transitions = [
            t for t in state1.transitions
            if t.id not in seen and not seen.add(t.id)  # type: ignore
        ]

        # 7. 删除 s2
        del self.states[s2]

    # ------------------------------------------------------------------ #
    # 工具方法                                                              #
    # ------------------------------------------------------------------ #

    def get_transitions(self, src: int, symbol: str) -> List[Transition]:
        return self._by_state_input.get((src, symbol), [])

    def __str__(self) -> str:
        lines = [
            "==== FSM Summary ====",
            f"States: {len(self.states)}",
            f"Transitions: {len(self.transitions)}",
            f"Start state: {self.start_state}",
            f"End states: {[sid for sid, s in self.states.items() if s.is_end]}",
            "",
            "---- States ----",
        ]
        for sid, state in self.states.items():
            flags = []
            if state.is_start: flags.append("START")
            if state.is_end:   flags.append("END")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"[{sid}] {state.name}{flag_str}, visits={state.visit_count}")
            for tran in state.transitions:
                lines.append(f"    --[{tran.symbol}]--> {tran.dst}")
        return "\n".join(lines)