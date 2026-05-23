from typing import Dict, List, Optional, Tuple

from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.interface.prob_trainer import ProbTrainer
from protocol_infer.core.model.efsm import EFSM, MemoryContext
from protocol_infer.core.model.fsm import Transition
from protocol_infer.core.model.pefsm import PEFSM


class PEFSMInferencer(ProbTrainer):
    """
    概率层训练器。
    对已有 EFSM 做会话重放统计，得到每条转移的 count / prob / confidence。
    """

    def __init__(
        self,
        min_count: int = 5,
        high_count: int = 30,
        high_ratio: float = 0.05,
    ):
        self.min_count = min_count
        self.high_count = high_count
        self.high_ratio = high_ratio

    def _select_transition(
        self,
        efsm: EFSM,
        sid: int,
        symbol: str,
        vars_dict: Dict[str, float],
        memory: MemoryContext,
    ) -> Optional[Transition]:
        candidates = efsm._by_state_input.get((sid, symbol), [])
        for tran in candidates:
            if tran.guard is None:
                return tran
            try:
                ok = tran.guard(vars_dict, memory)
            except TypeError:
                ok = tran.guard(vars_dict)
            if ok:
                return tran
        return candidates[0] if candidates else None

    def _apply_transition(
        self,
        tran: Transition,
        vars_dict: Dict[str, float],
        memory: MemoryContext,
    ) -> Dict[str, float]:
        """
        执行转移的 action, 更新变量字典和 memory
        """
        if tran.action is None:
            return vars_dict.copy()
        try:
            return tran.action(vars_dict.copy(), memory)
        except TypeError:
            return tran.action(vars_dict.copy())

    def train(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
    ) -> PEFSM:
        pefsm = PEFSM(
            base_efsm=efsm,
            min_count=self.min_count,
            high_count=self.high_count,
            high_ratio=self.high_ratio,
        )

        # 初始化
        for tran in pefsm.transitions:
            tran.traverse_count = 0
            tran.prob = None
            tran.confidence = None

        # 会话回放
        for _, pairs in sequences.items():
            current_state = pefsm.start_state
            if current_state is None:
                break

            memory = MemoryContext()
            for symbol, vars_dict in pairs:
                tran = self._select_transition(pefsm, current_state, symbol, vars_dict, memory)
                if tran is None:
                    break

                tran.traverse_count += 1
                self._apply_transition(tran, vars_dict, memory)
                current_state = tran.dst

        pefsm.compute_probabilities()
        return pefsm
