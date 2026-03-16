from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence
import math


class Metric(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        ...


class Evaluator(ABC):
    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = list(metrics)

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for m in self.metrics:
            r = m.compute(*args, **kwargs)
            res.update(r)
        return res


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _entropy(counts: Dict[int, int]) -> float:
    s = sum(counts.values())
    if s == 0:
        return 0.0
    e = 0.0
    for c in counts.values():
        if c > 0:
            p = c / s
            e -= p * math.log(p + 1e-12)
    return e


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
