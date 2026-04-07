from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple
from collections import defaultdict
from protocol_infer.core.datamodel.event import MessageEvent
from .core import AprioriCore
from .interfaces import ResultInterpreter, TransactionBuilder


class BytePositionTransactionBuilder(TransactionBuilder[MessageEvent]):
    '''
        未知协议
    '''
    def __init__(self, max_positions: int = 16, positions: Optional[Sequence[int]] = None):
        self.max_positions = max_positions
        self.positions = list(positions) if positions is not None else None

    def build(self, items: List[MessageEvent]) -> List[FrozenSet[Any]]:
        '''
        原始字节位置场景
        每个事件的 transaction 是其 payload 中
        候选字节偏移的(字节偏移, 字节值)集合。
        '''
        transactions = []
        for ev in items:
            payload = ev.payload or b""
            if self.positions is not None:
                positions = [i for i in self.positions if i < len(payload)]
            else:
                limit = min(len(payload), self.max_positions)
                positions = range(limit)
            transactions.append(frozenset((i, payload[i]) for i in positions))
        return transactions


class DictTransactionBuilder(TransactionBuilder[MessageEvent]):
    def __init__(
        self,
        extractor: Optional[Callable[[MessageEvent], Dict[str, Any]]] = None,
        discrete_threshold: int = 20,
        n_buckets: int = 5,
        exclude_fields: Optional[List[str]] = None,
    ):
        self.extractor = extractor or (lambda ev: getattr(ev, "parsed_fields", {}) or {})
        self.discrete_threshold = discrete_threshold
        self.n_buckets = n_buckets
        self.exclude_fields = set(exclude_fields or [])
        self._field_stats: Dict[str, Any] = {}

    def fit(self, items: List[MessageEvent]) -> "DictTransactionBuilder":
        field_values: Dict[str, list] = defaultdict(list)
        for ev in items:
            fields = self.extractor(ev)
            for k, v in fields.items():
                if k not in self.exclude_fields:
                    field_values[k].append(v)

        for field, values in field_values.items():
            unique = set(values)
            if len(unique) <= self.discrete_threshold:
                self._field_stats[field] = {"type": "discrete"}
            else:
                try:
                    sorted_vals = sorted(values)
                    n = len(sorted_vals)
                    boundaries = [sorted_vals[int(i * n / self.n_buckets)] for i in range(1, self.n_buckets)]
                    self._field_stats[field] = {
                        "type": "continuous",
                        "boundaries": boundaries,
                    }
                except (TypeError, ValueError):
                    self._field_stats[field] = {"type": "discrete"}
        return self

    def build(self, items: List[MessageEvent]) -> List[FrozenSet[Any]]:
        if not self._field_stats:
            self.fit(items)
        transactions = []
        for ev in items:
            fields = self.extractor(ev)
            items_set = set()
            for k, v in fields.items():
                if k in self.exclude_fields or k not in self._field_stats:
                    continue
                stat = self._field_stats[k]
                if stat["type"] == "discrete":
                    items_set.add((k, v))
                else:
                    bucket = self._discretize(v, stat["boundaries"])
                    items_set.add((k, bucket))
            transactions.append(frozenset(items_set))
        return transactions

    def _discretize(self, value, boundaries: list) -> int:
        for i, b in enumerate(boundaries):
            try:
                if value < b:
                    return i
            except (TypeError, ValueError):
                continue
        return len(boundaries)


class StaticFieldInterpreter(ResultInterpreter[List[FrozenSet]]):
    def __init__(self, global_static_threshold: float = 0.95):
        self.global_static_threshold = global_static_threshold
        self.last_global_static_items: FrozenSet[Any] = frozenset()

    def interpret(self, frequent_itemsets: List[Tuple[FrozenSet, float]]) -> List[Tuple[FrozenSet[Any], float]]:
        core = AprioriCore()

        global_static_items = set(self.last_global_static_items)
        global_static_items.update({
            next(iter(fs))
            for fs, sup in frequent_itemsets
            if len(fs) == 1 and sup >= self.global_static_threshold
        })
        self.last_global_static_items = frozenset(global_static_items)

        filtered = [
            (fs, sup) for fs, sup in frequent_itemsets
            if not any(item in global_static_items for item in fs)
        ]

        return core.maximal_itemsets(filtered)


class StaticFieldMiner:
    def __init__(
        self,
        builder: TransactionBuilder[MessageEvent] = None,
        interpreter: ResultInterpreter = None,
        min_support: float = 0.6,
    ):
        self.builder = builder or BytePositionTransactionBuilder()
        self.interpreter = interpreter or StaticFieldInterpreter()
        self.core = AprioriCore()
        self.min_support = min_support

    def _prune_global_static_items(self, transactions: List[FrozenSet[Any]]) -> List[FrozenSet[Any]]:
        if not transactions:
            return transactions
        threshold = getattr(self.interpreter, "global_static_threshold", None)
        if threshold is None:
            return transactions

        counts: Dict[Any, int] = defaultdict(int)
        total = len(transactions)
        for t in transactions:
            for item in t:
                counts[item] += 1

        global_static_items = {
            item for item, cnt in counts.items()
            if (cnt / total) >= float(threshold)
        }
        if hasattr(self.interpreter, "last_global_static_items"):
            self.interpreter.last_global_static_items = frozenset(global_static_items)
        if not global_static_items:
            return transactions
        return [frozenset(item for item in t if item not in global_static_items) for t in transactions]

    def mine(self, events: List[MessageEvent]) -> List[Tuple[FrozenSet[Any], float]]:
        if hasattr(self.builder, "fit"):
            self.builder.fit(events)

        transactions = self.builder.build(events)
        transactions = self._prune_global_static_items(transactions)
        fis = self.core.frequent_itemsets(transactions, self.min_support)
        return self.interpreter.interpret(fis)

    def get_global_static_items(self) -> FrozenSet[Any]:
        items = getattr(self.interpreter, "last_global_static_items", frozenset())
        if isinstance(items, frozenset):
            return items
        try:
            return frozenset(items)
        except TypeError:
            return frozenset()
