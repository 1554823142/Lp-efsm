from typing import Any, Dict, List, Optional, Set, Tuple
from protocol_infer.core.model.efsm import EFSM, MemoryContext
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.evaluation.base import Metric, Evaluator, _safe_div, _mean


class GuardPRF(Metric):
    def compute(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
        negative_sequences: Optional[Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]] = None,
    ) -> Dict[str, float]:
        tp = fp = fn = 0
        for pairs in sequences.values():
            cur = efsm.start_state
            mem = MemoryContext()
            for sym, vars_dict in pairs:
                cands = efsm._by_state_input.get((cur, sym), [])
                if not cands:
                    continue
                nxt, _ = efsm.step_with_memory(cur, sym, vars_dict, mem)
                if nxt is None:
                    fn += 1
                    break
                tp += 1
                cur = nxt
        if negative_sequences:
            for pairs in negative_sequences.values():
                cur = efsm.start_state
                mem = MemoryContext()
                for sym, vars_dict in pairs:
                    cands = efsm._by_state_input.get((cur, sym), [])
                    if not cands:
                        continue
                    nxt, _ = efsm.step_with_memory(cur, sym, vars_dict, mem)
                    if nxt is not None:
                        fp += 1
                        cur = nxt
                    else:
                        break
        p = _safe_div(tp, tp + fp) if tp + fp > 0 else 0.0
        r = _safe_div(tp, tp + fn) if tp + fn > 0 else 0.0
        f1 = _safe_div(2 * p * r, p + r) if p + r > 0 else 0.0
        return {"guard_precision": p, "guard_recall": r, "guard_f1": f1}


class ActionMetrics(Metric):
    def compute(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
        numeric_vars: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        maes: List[float] = []
        exact_total = exact_ok = 0
        step_total = step_ok = 0
        for pairs in sequences.values():
            cur = efsm.start_state
            for i in range(len(pairs) - 1):
                sym, vars_in = pairs[i]
                sym_next, vars_next = pairs[i + 1]
                cands = efsm._by_state_input.get((cur, sym), [])
                if not cands:
                    continue
                t = cands[0]
                pred_vars = t.action(vars_in.copy()) if t.action else vars_in.copy()
                if numeric_vars:
                    vals = []
                    for k in numeric_vars:
                        if k in pred_vars and k in vars_next:
                            vals.append(abs(pred_vars[k] - vars_next[k]))
                    if vals:
                        maes.append(_mean(vals))
                common_keys = set(pred_vars.keys()).intersection(vars_next.keys())
                for k in common_keys:
                    exact_total += 1
                    if pred_vars[k] == vars_next[k]:
                        exact_ok += 1
                for k in common_keys:
                    if k in numeric_vars if numeric_vars else isinstance(vars_in.get(k, 0.0), (int, float)):
                        step_total += 1
                        d_true = vars_next[k] - vars_in.get(k, 0.0)
                        d_pred = pred_vars[k] - vars_in.get(k, 0.0)
                        if d_true == d_pred:
                            step_ok += 1
                cur = t.dst
        mae = _mean(maes) if maes else 0.0
        emr = _safe_div(exact_ok, exact_total) if exact_total > 0 else 0.0
        step_acc = _safe_div(step_ok, step_total) if step_total > 0 else 0.0
        return {"action_mae": mae, "exact_match_rate": emr, "step_accuracy": step_acc}


class TraceReplay(Metric):
    def __init__(self, tol: float = 0.0):
        self.tol = tol

    def compute(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
    ) -> Dict[str, float]:
        total = len(sequences)
        ok = 0
        for pairs in sequences.values():
            cur = efsm.start_state
            mem = MemoryContext()
            good = True
            for sym, vars_in in pairs:
                nxt, _ = efsm.step_with_memory(cur, sym, vars_in, mem)
                if nxt is None:
                    good = False
                    break
                cur = nxt
            if good:
                ok += 1
        return {"trace_replay_acc": _safe_div(ok, total) if total > 0 else 0.0}


class FAR_FRR(Metric):
    def compute(
        self,
        efsm: EFSM,
        positive_sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
        negative_sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
    ) -> Dict[str, float]:
        fn = 0
        for pairs in positive_sequences.values():
            cur = efsm.start_state
            mem = MemoryContext()
            good = True
            for sym, vars_in in pairs:
                nxt, _ = efsm.step_with_memory(cur, sym, vars_in, mem)
                if nxt is None:
                    good = False
                    break
                cur = nxt
            if not good:
                fn += 1
        fp = 0
        for pairs in negative_sequences.values():
            cur = efsm.start_state
            mem = MemoryContext()
            accepted = True
            for sym, vars_in in pairs:
                nxt, _ = efsm.step_with_memory(cur, sym, vars_in, mem)
                if nxt is None:
                    accepted = False
                    break
                cur = nxt
            if accepted:
                fp += 1
        fr = _safe_div(fn, len(positive_sequences)) if positive_sequences else 0.0
        fa = _safe_div(fp, len(negative_sequences)) if negative_sequences else 0.0
        return {"frr": fr, "far": fa}


class EFSMevaluator(Evaluator):
    def __init__(self):
        super().__init__([GuardPRF(), ActionMetrics(), TraceReplay(), FAR_FRR()])

    def evaluate(
        self,
        efsm: EFSM,
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
        negative_sequences: Optional[Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]] = None,
        numeric_vars: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        res.update(GuardPRF().compute(efsm, sequences, negative_sequences))
        res.update(ActionMetrics().compute(efsm, sequences, numeric_vars))
        res.update(TraceReplay().compute(efsm, sequences))
        if negative_sequences:
            res.update(FAR_FRR().compute(efsm, sequences, negative_sequences))
        return res
