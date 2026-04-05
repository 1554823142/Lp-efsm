from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import random

from protocol_infer.core.model.efsm import EFSM, MemoryContext
from protocol_infer.core.model.fsm import Transition


class PEFSM(EFSM):
    """
    概率扩展有限状态机。
    在 EFSM 结构不变的基础上，为转移补充：
        - traverse_count: 观测次数
        - prob: 条件概率 P(dst | src, symbol)
        - confidence: high / medium / low
    """

    def __init__(
        self,
        base_efsm: Optional[EFSM] = None,
        min_count: int = 5,
        high_count: int = 30,
        high_ratio: float = 0.05,
    ):
        super().__init__(base_fsm=base_efsm)
        self.min_count = min_count
        self.high_count = high_count
        self.high_ratio = high_ratio

    def set_transition_stats(
        self,
        tran: Transition,
        count: int,
        probability: float,
        confidence: str,
    ) -> None:
        tran.traverse_count = count
        tran.prob = probability
        tran.confidence = confidence

    def assess_confidence(self, count: int, total: int) -> str:
        if total <= 0:
            return "low"
        ratio = count / total
        if count >= self.high_count and ratio >= self.high_ratio:
            return "high"
        if count >= self.min_count:
            return "medium"
        return "low"

    def compute_probabilities(self) -> None:
        grouped: Dict[Tuple[int, str], List[Transition]] = defaultdict(list)
        for tran in self.transitions:
            grouped[(tran.src, tran.symbol)].append(tran)

        for transitions in grouped.values():
            total = sum(max(0, t.traverse_count) for t in transitions)
            if total <= 0:
                fallback = 1.0 / len(transitions) if transitions else 0.0
                for tran in transitions:
                    tran.prob = fallback
                    tran.confidence = "low"
                continue

            for tran in transitions:
                prob = tran.traverse_count / total
                confidence = self.assess_confidence(tran.traverse_count, total)
                self.set_transition_stats(tran, tran.traverse_count, prob, confidence)

    def get_transition_probability(
        self,
        src: int,
        symbol: str,
        dst: int,
    ) -> float:
        for tran in self._by_state_input.get((src, symbol), []):
            if tran.dst == dst:
                return tran.prob if tran.prob is not None else 0.0
        return 0.0

    def get_probabilistic_transitions(self, src: int, symbol: str) -> List[Transition]:
        transitions = list(self._by_state_input.get((src, symbol), []))
        return sorted(
            transitions,
            key=lambda t: (
                -(t.prob if t.prob is not None else 0.0),
                -t.traverse_count,
                t.id,
            ),
        )

    def step_probabilistic(
        self,
        sid: int,
        symbol: str,
        vars: Dict[str, float],
        memory: Optional[MemoryContext] = None,
        rng: Optional[random.Random] = None,
    ) -> Tuple[Optional[int], Optional[Dict[str, float]], Optional[Transition]]:
        candidates = []
        for tran in self._by_state_input.get((sid, symbol), []):
            if tran.guard is not None:
                try:
                    ok = tran.guard(vars, memory)
                except TypeError:
                    ok = tran.guard(vars)
                if not ok:
                    continue
            candidates.append(tran)

        if not candidates:
            return None, None, None

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            weights = [max(0.0, t.prob if t.prob is not None else 0.0) for t in candidates]
            if sum(weights) <= 0:
                chosen = max(candidates, key=lambda t: (t.traverse_count, -t.id))
            else:
                generator = rng if rng is not None else random
                chosen = generator.choices(candidates, weights=weights, k=1)[0]

        if chosen.action is None:
            return chosen.dst, vars.copy(), chosen

        try:
            new_vars = chosen.action(vars.copy(), memory)
        except TypeError:
            new_vars = chosen.action(vars.copy())
        return chosen.dst, new_vars, chosen

    def __str__(self) -> str:
        lines = [
            "==== PEFSM Summary ====",
            f"States: {len(self.states)}",
            f"Transitions: {len(self.transitions)}",
            f"Start state: {self.start_state}",
            f"End states: {[sid for sid, s in self.states.items() if s.is_end]}",
            "",
            "---- States ----",
        ]
        for sid, state in self.states.items():
            flags = []
            if state.is_start:
                flags.append("START")
            if state.is_end:
                flags.append("END")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            lines.append(f"[{sid}] {state.name}{flag_str}, visits={state.visit_count}")
            for tran in state.transitions:
                prob = tran.prob if tran.prob is not None else 0.0
                conf = tran.confidence if tran.confidence is not None else "unknown"
                lines.append(
                    f"    --[{tran.symbol}]--> {tran.dst} "
                    f"(count={tran.traverse_count}, prob={prob:.4f}, conf={conf})"
                )
        return "\n".join(lines)
