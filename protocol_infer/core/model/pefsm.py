from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import math
import heapq
import random

from protocol_infer.core.model.efsm import EFSM, MemoryContext
from protocol_infer.core.model.fsm import Transition


class PEFSM(EFSM):
    """
    概率扩展有限状态机。
    在 EFSM 结构不变的基础上，为转移补充：
        - traverse_count: 观测次数
        - prob: 状态出边概率 P(transition | src)
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
        grouped: Dict[int, List[Transition]] = defaultdict(list)
        for tran in self.transitions:
            grouped[tran.src].append(tran)

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

    def prune_transitions(
        self,
        min_count: Optional[int] = None,
        min_prob: Optional[float] = None,
        combine: str = "or",
        preserve_end_reachability: bool = True,
    ) -> Dict[str, int]:
        """
        根据统计置信度修剪低质量转移边
        """
        if self.start_state is None or not self.transitions:
            return {"before_transitions": len(self.transitions), "after_transitions": len(self.transitions)}

        combine = (combine or "or").strip().lower()
        if combine not in {"or", "and"}:
            combine = "or"

        all_transitions = list(self.transitions)
        by_src: Dict[int, List[Transition]] = defaultdict(list)
        out_srcs = set()
        for t in all_transitions:
            by_src[t.src].append(t)
            out_srcs.add(t.src)

        end_states = {sid for sid, st in self.states.items() if getattr(st, "is_end", False)}
        if not end_states:
            end_states = {sid for sid in self.states.keys() if sid not in out_srcs}
        if not end_states:
            end_states = {self.start_state}

        def keep_by_rule(t: Transition) -> bool:
            conds = []
            if min_count is not None:
                conds.append(int(t.traverse_count) >= int(min_count))
            if min_prob is not None:
                conds.append(float(t.prob or 0.0) >= float(min_prob))
            if not conds:
                return True
            return all(conds) if combine == "and" else any(conds)

        keep_ids = {t.id for t in all_transitions if keep_by_rule(t)}

        if preserve_end_reachability:
            def reachable_ids(kept: set) -> set:
                seen = set()
                q = [self.start_state]
                kept_by_src: Dict[int, List[Transition]] = defaultdict(list)
                for t in all_transitions:
                    if t.id in kept:
                        kept_by_src[t.src].append(t)
                while q:
                    cur = q.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    for t in kept_by_src.get(cur, []):
                        if t.dst not in seen:
                            q.append(t.dst)
                return seen

            def best_path_edges(start: int, goal: int) -> List[Transition]:
                dist: Dict[int, float] = {start: 0.0}
                prev: Dict[int, Tuple[int, Transition]] = {}
                heap = [(0.0, start)]
                while heap:
                    d, u = heapq.heappop(heap)
                    if d != dist.get(u, 0.0):
                        continue
                    if u == goal:
                        break
                    for t in by_src.get(u, []):
                        p = float(t.prob or 0.0)
                        w = (-math.log(max(p, 1e-12))) + (1.0 / (float(t.traverse_count) + 1.0))
                        nd = d + w
                        if nd < dist.get(t.dst, float("inf")):
                            dist[t.dst] = nd
                            prev[t.dst] = (u, t)
                            heapq.heappush(heap, (nd, t.dst))
                if goal not in prev and goal != start:
                    return []
                path: List[Transition] = []
                cur = goal
                while cur != start:
                    item = prev.get(cur)
                    if item is None:
                        return []
                    u, t = item
                    path.append(t)
                    cur = u
                path.reverse()
                return path

            reachable = reachable_ids(keep_ids)
            for end in sorted(end_states):
                if end in reachable:
                    continue
                path = best_path_edges(self.start_state, end)
                if not path:
                    continue
                for t in path:
                    keep_ids.add(t.id)
                reachable = reachable_ids(keep_ids)

            if end_states and not (end_states & reachable):
                fallback_end = max(end_states, key=lambda sid: getattr(self.states.get(sid), "visit_count", 0))
                path = best_path_edges(self.start_state, fallback_end)
                for t in path:
                    keep_ids.add(t.id)

        kept_transitions = [t for t in all_transitions if t.id in keep_ids]

        for st in self.states.values():
            st.transitions = []
        self.transitions = []
        self._by_state_input = {}
        for t in kept_transitions:
            self._register_transition(t)

        keep_tid = {t.id for t in kept_transitions}
        self._transition_guards = {k: v for k, v in self._transition_guards.items() if k in keep_tid}
        self._transition_actions = {k: v for k, v in self._transition_actions.items() if k in keep_tid}

        self.compute_probabilities()
        if self.start_state is not None:
            adj: Dict[int, List[int]] = defaultdict(list)
            radj: Dict[int, List[int]] = defaultdict(list)
            incident = set()
            for t in self.transitions:
                adj[t.src].append(t.dst)
                radj[t.dst].append(t.src)
                incident.add(t.src)
                incident.add(t.dst)

            reachable = set()
            stack = [self.start_state]
            while stack:
                cur = stack.pop()
                if cur in reachable:
                    continue
                reachable.add(cur)
                for nxt in adj.get(cur, []):
                    if nxt not in reachable:
                        stack.append(nxt)

            explicit_ends = {sid for sid, st in self.states.items() if getattr(st, "is_end", False)}
            can_reach_end = set()
            if explicit_ends:
                q = [sid for sid in explicit_ends if sid in reachable]
                while q:
                    cur = q.pop()
                    if cur in can_reach_end:
                        continue
                    can_reach_end.add(cur)
                    for pre in radj.get(cur, []):
                        if pre in reachable and pre not in can_reach_end:
                            q.append(pre)

            keep_states = set()
            if explicit_ends:
                keep_states = reachable & can_reach_end
            else:
                for sid, st in self.states.items():
                    if sid not in reachable:
                        continue
                    if getattr(st, "is_start", False) or getattr(st, "is_end", False) or sid in incident:
                        keep_states.add(sid)
                if self.start_state in reachable:
                    keep_states.add(self.start_state)

            if keep_states:
                self.states = {sid: st for sid, st in self.states.items() if sid in keep_states}
                kept2 = [t for t in self.transitions if t.src in keep_states and t.dst in keep_states]

                for st in self.states.values():
                    st.transitions = []
                self.transitions = []
                self._by_state_input = {}
                for t in kept2:
                    self._register_transition(t)

                keep_tid2 = {t.id for t in kept2}
                self._transition_guards = {k: v for k, v in self._transition_guards.items() if k in keep_tid2}
                self._transition_actions = {k: v for k, v in self._transition_actions.items() if k in keep_tid2}
                self.compute_probabilities()

        return {"before_transitions": len(all_transitions), "after_transitions": len(self.transitions)}

    def get_transition_probability(self, src: int, dst: int, symbol: Optional[str] = None) -> float:
        for tran in self.transitions:
            if tran.src != src or tran.dst != dst:
                continue
            if symbol is not None and tran.symbol != symbol:
                continue
            return tran.prob if tran.prob is not None else 0.0
        return 0.0

    def get_probabilistic_transitions(self, src: int) -> List[Transition]:
        transitions = [tran for tran in self.transitions if tran.src == src]
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
                    f"(count={tran.traverse_count}, state_prob={prob:.4f}, conf={conf})"
                )
        return "\n".join(lines)
