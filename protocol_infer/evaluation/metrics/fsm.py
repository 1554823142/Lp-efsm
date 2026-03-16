from typing import Any, Dict, List, Set, Tuple
from protocol_infer.core.model.fsm import FSM
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.evaluation.base import Metric, Evaluator, _safe_div


class StateMatchRate(Metric):
    def compute(self, gt_states: Set[str], pred_states: Set[str]) -> Dict[str, float]:
        inter = len(gt_states.intersection(pred_states))
        return {"state_match_rate": _safe_div(inter, len(gt_states)) if gt_states else 0.0}


class TransitionPRF(Metric):
    def compute(self, gt_edges: Set[Tuple[str, str, str]], pred_edges: Set[Tuple[str, str, str]]) -> Dict[str, float]:
        tp = len(gt_edges.intersection(pred_edges))
        p = _safe_div(tp, len(pred_edges)) if pred_edges else 0.0
        r = _safe_div(tp, len(gt_edges)) if gt_edges else 0.0
        f1 = _safe_div(2 * p * r, p + r) if p + r > 0 else 0.0
        return {"transition_precision": p, "transition_recall": r, "transition_f1": f1}


class SimpleGED(Metric):
    def compute(self, gt_states: Set[str], gt_edges: Set[Tuple[str, str, str]], pred_states: Set[str], pred_edges: Set[Tuple[str, str, str]]) -> Dict[str, float]:
        ns = len(gt_states.symmetric_difference(pred_states))
        ne = len(gt_edges.symmetric_difference(pred_edges))
        denom = max(len(gt_states) + len(gt_edges), 1)
        return {"ged_norm": _safe_div(ns + ne, denom)}


class TraceCoverage(Metric):
    def compute(self, fsm: FSM, sequences: Dict[SessionKey, List[str]]) -> Dict[str, float]:
        total = len(sequences)
        ok = 0
        for seq in sequences.values():
            cur = fsm.start_state
            good = True
            for sym in seq:
                cands = fsm.get_transitions(cur, sym)
                if not cands:
                    good = False
                    break
                cur = cands[0].dst
            if good:
                ok += 1
        return {"trace_coverage": _safe_div(ok, total) if total > 0 else 0.0}


class FSMEvaluator(Evaluator):
    def __init__(self):
        super().__init__([StateMatchRate(), TransitionPRF(), SimpleGED(), TraceCoverage()])

    def evaluate(
        self,
        gt_states: Set[str],
        gt_edges: Set[Tuple[str, str, str]],
        pred_states: Set[str],
        pred_edges: Set[Tuple[str, str, str]],
        fsm: FSM,
        sequences: Dict[SessionKey, List[str]],
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        res.update(StateMatchRate().compute(gt_states, pred_states))
        res.update(TransitionPRF().compute(gt_edges, pred_edges))
        res.update(SimpleGED().compute(gt_states, gt_edges, pred_states, pred_edges))
        res.update(TraceCoverage().compute(fsm, sequences))
        return res
