from typing import Any, Dict, Set, Tuple
from protocol_infer.evaluation.base import Metric, Evaluator, _safe_div


class BoundaryPRF(Metric):
    def compute(self, gt: Set[Tuple[int, int]], pred: Set[Tuple[int, int]]) -> Dict[str, float]:
        tp = len(gt.intersection(pred))
        p = _safe_div(tp, len(pred)) if pred else 0.0
        r = _safe_div(tp, len(gt)) if gt else 0.0
        f1 = _safe_div(2 * p * r, p + r) if p + r > 0 else 0.0
        return {"boundary_precision": p, "boundary_recall": r, "boundary_f1": f1}


class BoundaryEvaluator(Evaluator):
    def __init__(self):
        super().__init__([BoundaryPRF()])

    def evaluate(self, gt: Set[Tuple[int, int]], pred: Set[Tuple[int, int]]) -> Dict[str, Any]:
        return super().evaluate(gt, pred)
