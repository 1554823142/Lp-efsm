from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence
import math


# ---------------------------------------------------------------------------
# 通用工具函数（供各 Metric 子模块导入）
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _entropy(counter) -> float:
    """香农熵，counter 为 {label: count} 字典。"""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        if v > 0:
            p = v / total
            ent -= p * math.log(p)
    return ent


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class Metric(ABC):
    @abstractmethod
    def compute(self, *args, **kwargs) -> Dict[str, float]:
        pass


class Evaluator(ABC):
    def __init__(self, metrics: List[Metric] = None):
        self._metrics = metrics or []

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        pass
