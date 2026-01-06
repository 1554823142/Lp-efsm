from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Iterable


@dataclass(frozen=True)
class Transition:
    id: int
    src: int
    dst: int
    symbol: str
    guard: Optional[Callable[[Dict], bool]]  # guard(vars) -> bool
    action: Optional[Callable[[Dict], Dict]]  # action(vars) -> new_vars
    output: Optional[str] = None
    prob: Optional[float] = None  # for P-EFSM

class FSM:
    def __init__(self):
        self.states: Dict[str, int] = {}
        self.start_state: Optional[int] = None
        self._next_state_id = 0
        self.transitions: List[Transition] = []
        self._by_state_input: Dict[Tuple[int, str], List[Transition]] = {}

    def add_state(self, name: str, start: bool=False) -> int: ...
    def add_transition(self, src: str, symbol: str, dst: str, **kwargs) -> Transition: ...
    def successors(self, state: int, symbol: str) -> List[Transition]:
        return self._by_state_input.get((state, symbol), [])
    