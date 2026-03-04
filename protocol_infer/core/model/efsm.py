from protocol_infer.core.model.fsm import FSM, FSMState, Transition
from typing import Callable, Dict, Optional, Any, Set, List, Tuple
import copy


class EFSMState(FSMState):
    """
    EFSM 状态，扩展状态变量。
    """
    def __init__(self, *args, variables: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables: Dict[str, Any] = variables if variables is not None else {}


class EFSM(FSM):
    """
    扩展有限状态机。
    - guard/action 统一挂载在 Transition 对象上，字典仅作快速查找用途
    - _by_state_input 是主要转移索引，step() 优先使用
    - 从 base_fsm 构造时正确重建所有状态的 transitions 列表
    """

    def __init__(self, base_fsm: Optional[FSM] = None):
        super().__init__()

        # EFSM 扩展字段
        self.variable_defs: Set[str] = set()
        self._transition_guards: Dict[int, Callable] = {}
        self._transition_actions: Dict[int, Callable] = {}

        if base_fsm:
            self._copy_from_fsm(base_fsm)

    def _copy_from_fsm(self, base_fsm: FSM) -> None:
        """
        从 FSM 构造 EFSM：
          1. 将所有 FSMState 升级为 EFSMState
          2. 深拷贝转移，并重建各状态的 transitions 列表和 _by_state_input 索引
        """
        # 1. 升级状态
        self.states = {}
        for sid, st in base_fsm.states.items():
            efst = EFSMState(
                name=st.name,
                is_start=st.is_start,
                is_end=st.is_end,
                variables={}
            )
            efst.visit_count = st.visit_count
            self.states[sid] = efst
        self.start_state = base_fsm.start_state
        self._next_state_id = base_fsm._next_state_id

        # 2. 深拷贝转移，重建索引和各状态的 transitions 列表
        self.transitions = []
        self._by_state_input = {}

        for tran in base_fsm.transitions:
            new_tran = copy.deepcopy(tran)
            self._register_transition(new_tran)

    def _register_transition(self, tran: Transition) -> None:
        """
        内部方法：将转移加入所有索引。
        单一入口，保证索引一致性。
        """
        self.transitions.append(tran)
        self._by_state_input.setdefault((tran.src, tran.symbol), []).append(tran)

        # 重建 FSMState.transitions（step 依赖此列表）
        if tran.src in self.states:
            self.states[tran.src].transitions.append(tran)

    def new_state(
        self,
        is_start: bool = False,
        is_end: bool = False,
        variables: Optional[Dict[str, Any]] = None
    ) -> int:
        """创建新的 EFSM 状态，返回状态 ID"""
        sid = self._next_state_id
        self._next_state_id += 1
        self.states[sid] = EFSMState(
            name=f"s{sid}",
            is_start=is_start,
            is_end=is_end,
            variables=variables if variables is not None else {}
        )
        return sid

    def add_transition(self, tran: Transition) -> None:
        """添加转移（公开接口）"""
        self._register_transition(tran)

    def register_guard_action(
        self,
        tran: Transition,
        guard: Optional[Callable],
        action: Optional[Callable]
    ) -> None:
        """
        为转移注册 guard 和 action。
        guard/action 同时写入 tran 对象和快速查找字典，保持一致。
        """
        tran.guard = guard
        tran.action = action
        # 字典与 tran 同步，get_guard/get_action 与 step 行为一致
        if guard is not None:
            self._transition_guards[tran.id] = guard
        else:
            self._transition_guards.pop(tran.id, None)

        if action is not None:
            self._transition_actions[tran.id] = action
        else:
            self._transition_actions.pop(tran.id, None)

    def step(
        self,
        sid: int,
        symbol: str,
        vars: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """
        执行一次状态转移。
        使用 _by_state_input 索引快速定位候选转移，
        再按 guard 过滤，找到第一个可触发的转移执行 action。

        返回: (dst_state_id, new_vars) 或 (None, None)
        """
        candidates: List[Transition] = self._by_state_input.get((sid, symbol), [])

        for tran in candidates:
            if tran.guard is not None and not tran.guard(vars):
                continue
            new_vars = tran.action(vars.copy()) if tran.action is not None else vars.copy()
            return tran.dst, new_vars

        return None, None

    def get_guard(self, tid: int) -> Optional[Callable]:
        return self._transition_guards.get(tid)

    def get_action(self, tid: int) -> Optional[Callable]:
        return self._transition_actions.get(tid)

    def get_next_states(self, sid: int, symbol: str) -> List[int]:
        """返回给定状态在 symbol 下所有可能的目标状态 ID"""
        return [t.dst for t in self._by_state_input.get((sid, symbol), [])]

    def get_prev_states(self, sid: int, symbol: str) -> List[int]:
        """返回所有通过 symbol 能到达 sid 的源状态 ID"""
        return [
            t.src
            for (src, sym), trans in self._by_state_input.items()
            if sym == symbol
            for t in trans
            if t.dst == sid
        ]
