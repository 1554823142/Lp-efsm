from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
from collections import deque


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
    traverse_count: int = 0


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

    def accepts(self, sequence: List[str]) -> bool:
        """
        Check if the FSM accepts the given sequence of symbols.
        Assumes deterministic transitions for simplicity, or takes the first matching transition.
        """
        if self.start_state is None:
            return False
            
        current_state = self.start_state
        for symbol in sequence:
            # Check transitions from current state with this symbol
            transitions = self._by_state_input.get((current_state, symbol))
            if not transitions:
                return False
            
            # For now, just take the first transition (greedy/deterministic assumption)
            # In a non-deterministic FSM, we might need BFS/DFS to find *any* valid path
            current_state = transitions[0].dst
            
        # Check if we ended in a valid state (usually any state is valid for protocol traces unless we have explicit end states)
        # If we successfully traversed the whole sequence, we consider it "accepted" in terms of coverage
        return True


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

        # 6. 对合并后 (s1, symbol, dst) 的转移进行语义去重
        for key in list(self._by_state_input):
            if key[0] == s1:
                unique_by_dst: Dict[int, Transition] = {}
                for t in self._by_state_input[key]:
                    if t.dst not in unique_by_dst:
                        unique_by_dst[t.dst] = t
                    else:
                        unique_by_dst[t.dst].traverse_count += t.traverse_count
                self._by_state_input[key] = list(unique_by_dst.values())
        
        # state1.transitions 同步语义去重
        unique_pairs: set = set()
        new_transitions: List[Transition] = []
        for t in state1.transitions:
            pair = (t.symbol, t.dst)
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                new_transitions.append(t)
            else:
                # 累加频次到已存在的对应项
                for x in new_transitions:
                    if x.symbol == t.symbol and x.dst == t.dst:
                        x.traverse_count += t.traverse_count
                        break
        state1.transitions = new_transitions

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
                freq = tran.traverse_count
                if freq and freq > 1:
                    lines.append(f"    --[{tran.symbol}]--> {tran.dst} (x{freq})")
                else:
                    lines.append(f"    --[{tran.symbol}]--> {tran.dst}")
        return "\n".join(lines)

    def enforce_determinism(self) -> None:
        new_transitions_all: List[Transition] = []
        for sid, state in self.states.items():
            by_symbol: Dict[str, List[Transition]] = {}
            for t in state.transitions:
                by_symbol.setdefault(t.symbol, []).append(t)
            kept: List[Transition] = []
            for sym, lst in by_symbol.items():
                if len(lst) == 1:
                    chosen = lst[0]
                else:
                    chosen = max(lst, key=lambda x: (x.traverse_count, -x.id))
                kept.append(chosen)
                self._by_state_input[(sid, sym)] = [chosen]
                state.next_states[sym] = chosen.dst
            state.transitions = kept
            new_transitions_all.extend(kept)
        self.transitions = new_transitions_all

    def determinize(self) -> "FSM":
        '''
            确定性化 FSM，返回新的确定性 FSM
        '''

        # 1. 初始化
        new_fsm = FSM()
        if self.start_state is None:
            return new_fsm
        start_set = frozenset([self.start_state])
        subset_map: Dict[frozenset, int] = {}       # frozenset(不可变集合)可哈希

        # 2. DFA初始状态
        start_is_end = any(self.states[s].is_end for s in start_set)    # 只要有一个状态是结束状态，新状态就标记为结束状态
        start_id = new_fsm.new_state(is_start=True, is_end=start_is_end)
        # 新状态的访问次数取所有原状态中最大的(避免互斥分支的虚高)
        new_fsm.states[start_id].visit_count = max(
            (self.states[s].visit_count for s in start_set if s in self.states),
            default=0
        )
        subset_map[start_set] = start_id

        # 3. 构建DFA状态转移
        queue = deque([start_set])
        while queue:                    # 使用队列进行 BFS 遍历
            current = queue.popleft()
            current_id = subset_map[current]
            sym_targets: Dict[str, set] = {}    # 按符号分组收集状态    
            counts: Dict[str, int] = {}         # 记录每个符号的最大遍历次数
            for s in current:                   # 遍历当前状态集中的每个状态
                state = self.states.get(s)
                if not state:
                    continue
                for t in state.transitions:                 # 遍历当前状态的每个转移
                    sym_targets.setdefault(t.symbol, set()).add(t.dst)
                    counts[t.symbol] = max(
                        counts.get(t.symbol, 0), t.traverse_count or 0
                    )
            for sym, targets in sym_targets.items():
                target_set = frozenset(targets)
                if target_set not in subset_map:        # 避免重复状态
                    is_end = any(self.states[x].is_end for x in target_set)
                    new_id = new_fsm.new_state(is_end=is_end)
                    new_fsm.states[new_id].visit_count = max(
                        (self.states[s].visit_count for s in target_set if s in self.states),
                        default=0
                    )
                    subset_map[target_set] = new_id
                    queue.append(target_set)
                tran = new_fsm.add_transition(current_id, subset_map[target_set], sym)
                tran.traverse_count = counts.get(sym, 0)
        return new_fsm

    def remove_duplicate_transitions(self) -> None:
        '''
            遍历所有转移，用 (src, dst, symbol) 三元组作为唯一标识
            清空所有状态的转移列表，重新构建
        '''
        seen: set = set()
        unique: List[Transition] = []           # 仅保留第一次出现的转移
        # 1. 格式化为三元组 (src, dst, symbol)
        for tran in self.transitions:
            key = (tran.src, tran.dst, tran.symbol)     # 三元组
            if key not in seen:
                seen.add(key)
                unique.append(tran)

        # 2. 重新构建转移列表
        self.transitions = unique
        self._by_state_input.clear()
        for sid in self.states:
            self.states[sid].transitions = []
            self.states[sid].next_states.clear()
            self.states[sid].prev_states.clear()
        
        # 3. 构建状态转移索引
        for tran in self.transitions:
            self._by_state_input.setdefault((tran.src, tran.symbol), []).append(tran)
            if tran.src in self.states:
                self.states[tran.src].transitions.append(tran)
                self.states[tran.src].next_states[tran.symbol] = tran.dst
            if tran.dst in self.states:
                self.states[tran.dst].prev_states[tran.symbol] = tran.src
