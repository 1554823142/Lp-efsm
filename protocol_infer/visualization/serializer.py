from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from protocol_infer.core.model.pefsm import PEFSM
from protocol_infer.evaluation.inspect_efsm import (
    extract_action_desc,
    extract_cross_message_desc,
    extract_guard_desc,
    _fmt_action_desc,
    _fmt_cross_message_desc,
    _fmt_guard_desc,
)


@dataclass
class ReplayStep:
    index: int
    timestamp: float
    session: str
    symbol: str
    src: str
    dst: Optional[str]
    transition_id: Optional[int]
    matched: bool
    is_strict: bool = False
    probability: Optional[float] = None
    confidence: Optional[str] = None
    payload_hex: str = ""
    direction: str = ""
    vars: Dict[str, float] = None
    note: Optional[str] = None


class PEFSMSerializer:
    def serialize_model(self, pefsm: PEFSM, protocol: str = "UNKNOWN") -> Dict:
        proto_upper = protocol.upper()
        state_visits = self._calc_state_visits(pefsm)
        nodes = []
        for sid, state in sorted(pefsm.states.items()):
            nodes.append(
                {
                    "data": {
                        "id": f"s{sid}",
                        "label": state.name,
                        "state_id": sid,
                        "is_start": state.is_start,
                        "is_end": state.is_end,
                        "visit_count": state_visits.get(sid, 0),
                        "variables": sorted(getattr(state, "variables", {}).keys()),
                        "out_degree": len(state.transitions),
                    }
                }
            )

        edges = []
        for tran in sorted(pefsm.transitions, key=lambda t: (t.src, t.symbol, t.dst, t.id)):
            guard_desc = extract_guard_desc(tran.guard)
            cross_desc = extract_cross_message_desc(tran.guard)
            action_desc = extract_action_desc(tran.action)
            guard_lines = self._normalize_lines(_fmt_guard_desc(guard_desc, proto_upper, indent=""))
            cross_lines = self._normalize_lines(_fmt_cross_message_desc(cross_desc, proto_upper, indent=""))
            action_lines = self._normalize_lines(_fmt_action_desc(action_desc, proto_upper, indent=""))

            edges.append(
                {
                    "data": {
                        "id": f"t{tran.id}",
                        "transition_id": tran.id,
                        "source": f"s{tran.src}",
                        "target": f"s{tran.dst}",
                        "label": tran.symbol,
                        "symbol": tran.symbol,
                        "probability": float(tran.prob or 0.0),
                        "count": int(tran.traverse_count),
                        "confidence": tran.confidence or "low",
                        "is_low_probability": (tran.prob or 0.0) < 0.1,
                        "has_guard": tran.guard is not None,
                        "has_action": tran.action is not None,
                        "guard_desc": guard_desc,
                        "cross_message_desc": cross_desc,
                        "action_desc": action_desc,
                        "guard_text": "\n".join(guard_lines),
                        "cross_message_text": "\n".join(cross_lines),
                        "action_text": "\n".join(action_lines),
                    }
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "summary": self._build_summary(pefsm),
        }

    def serialize_replay(self, steps: List[ReplayStep]) -> Dict:
        matched = sum(1 for step in steps if step.matched)
        total = len(steps)
        return {
            "steps": [asdict(step) for step in steps],
            "summary": {
                "total_steps": total,
                "matched_steps": matched,
                "unmatched_steps": total - matched,
                "match_rate": (matched / total) if total else 0.0,
            },
        }

    def _normalize_lines(self, lines: List[str]) -> List[str]:
        return [line.strip() for line in lines if line and line.strip()]

    def _calc_state_visits(self, pefsm: PEFSM) -> Dict[int, int]:
        visits: Dict[int, int] = {sid: 0 for sid in pefsm.states}
        if pefsm.start_state is not None:
            visits[pefsm.start_state] = 1
        for tran in pefsm.transitions:
            visits[tran.src] = max(visits.get(tran.src, 0), tran.traverse_count)
            visits[tran.dst] = visits.get(tran.dst, 0) + tran.traverse_count
        return visits

    def _build_summary(self, pefsm: PEFSM) -> Dict:
        transitions = pefsm.transitions
        branching = sum(1 for sid in pefsm.states if len(pefsm.get_probabilistic_transitions(sid)) > 1)
        return {
            "states": len(pefsm.states),
            "transitions": len(transitions),
            "start_state": f"s{pefsm.start_state}" if pefsm.start_state is not None else None,
            "high_confidence": sum(1 for tran in transitions if tran.confidence == "high"),
            "medium_confidence": sum(1 for tran in transitions if tran.confidence == "medium"),
            "low_confidence": sum(1 for tran in transitions if tran.confidence == "low"),
            "branching_states": branching,
        }
