from typing import List, Tuple, FrozenSet, Any, Optional, Dict, Set
from protocol_infer.core.interface.feature_extractor import FeatureExtractor
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.apriori.miners import StaticFieldMiner, BytePositionTransactionBuilder, StaticFieldInterpreter
from protocol_infer.apriori.core import AprioriCore

class AprioriFeatureExtraction(FeatureExtractor):
    def __init__(
        self,
        positions: List[int],
        itemsets: List[FrozenSet[Tuple[int, int]]],
        field_groups: Optional[List[List[int]]] = None,
        static_items: Optional[Dict[int, int]] = None,
        max_payload_len: float = 1.0,
        onehot_weight: float = 2.0
    ):
        self.positions = positions
        self.itemsets = itemsets
        self.field_groups = field_groups or []
        self.static_items = static_items or {}
        self.max_payload_len = max_payload_len
        self.onehot_weight = onehot_weight
        self._miner: Optional[StaticFieldMiner] = None  # 由 from_events 填充
        self._byte_bins: Dict[int, Tuple[int, int]] = {}

    @classmethod                    # 可能不需要预先创建类实例
    def from_events(
        cls,                        # 类似self,接收类本身 
        events_all: List[MessageEvent],
        max_positions: int = 16,
        max_itemsets: int = 8,
        min_support: float = 0.3,
        onehot_weight: float = 2.0,
        global_static_threshold: float = 0.95,
        enable_field_groups: bool = True,
        field_group_min_support: float = 0.6,
        field_group_min_confidence: float = 0.9,
        field_group_bins: int = 3,
        field_group_max_groups: int = 8,
        enable_range_confirmation: bool = True,
        range_ratio_threshold: float = 0.1,
    ) -> "AprioriFeatureExtraction":
    
        # 1. 用 BytePositionTransactionBuilder + Apriori 发现伪字段
        miner = StaticFieldMiner(
            builder=BytePositionTransactionBuilder(max_positions=max_positions),
            interpreter=StaticFieldInterpreter(global_static_threshold=global_static_threshold),
            min_support=min_support
        )
        maximal_sets = miner.mine(events_all)

        # 2. 从结果中提取有价值的偏移位置 (伪字段)

        # 出现次数多的偏移, 说明它在多个不同的消息类型模式里都起作用, 区分价值更高，应该优先保留
        pos_count = {}
        for fs, _ in maximal_sets:
            for (pos, _val) in fs:
                pos_count[pos] = pos_count.get(pos, 0) + 1

        # 按出现次数降序，截断后再按偏移大小排序（保持特征维度顺序稳定）
        positions = sorted(
            sorted(pos_count, key=lambda p: -pos_count[p])[:max_positions]
        )

        # 按支持度降序排序后截断，保留最有代表性的项集
        top_itemsets = sorted(maximal_sets, key=lambda x: -x[1])[:max_itemsets]
        itemsets = [fs for fs, _ in top_itemsets]   # 只保留项集，丢弃支持度

        # 计算最大负载长度，用于归一化
        max_payload_len = max((len(e.payload or b"") for e in events_all), default=1)

        field_groups: List[List[int]] = []
        byte_bins: Dict[int, Tuple[int, int]] = {}
        static_items: Dict[int, int] = {}
        if enable_field_groups:
            field_groups, byte_bins = cls._discover_field_groups(
                events_all=events_all,
                max_positions=max_positions,
                min_support=field_group_min_support,
                min_confidence=field_group_min_confidence,
                n_bins=field_group_bins,
                max_groups=field_group_max_groups,
                enable_range_confirmation=enable_range_confirmation,
                range_ratio_threshold=range_ratio_threshold,
            )

        instance = cls(
            positions=positions,
            itemsets=itemsets,
            field_groups=field_groups,
            static_items=static_items,
            max_payload_len=float(max_payload_len),
            onehot_weight=onehot_weight,
        )
        instance._miner = miner
        instance._byte_bins = byte_bins
        instance.static_items = cls._extract_static_items(miner.get_global_static_items())
        return instance

    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        # 3. 构造特征向量
        features: List[List[float]] = []
        for event in trace:
            payload = event.payload or b""

            # 3.1. 静态偏移字节值 (归一化到 [0,1])
            pos_vals = [
                float(payload[pos]) / 255.0 if pos < len(payload) else 0.0
                for pos in self.positions
            ]
 
            # 3.2. 项集匹配 one-hot (加权)
            onehot = []
            for fs in self.itemsets:
                matched = all(p < len(payload) and payload[p] == v for (p, v) in fs)
                onehot.append(self.onehot_weight if matched else 0.0)

            # 3.3. 归一化长度
            length_val = len(payload) / max(self.max_payload_len, 1.0)      # 防止除以0

            # 3.4. 方向特征
            direction_val = float(event.direction.to_feature())

            group_vals = [
                self._field_group_feature(payload, group)
                for group in self.field_groups
            ]

            vec = pos_vals + onehot + group_vals + [length_val, direction_val]
            features.append(vec)
        return features

    @staticmethod
    def _extract_static_items(items: FrozenSet[Any]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for it in items:
            if isinstance(it, tuple) and len(it) == 2:
                pos, val = it
                if isinstance(pos, int):
                    try:
                        out[int(pos)] = int(val)
                    except (TypeError, ValueError):
                        continue
        return out

    @staticmethod
    def _percentiles(sorted_vals: List[int], ps: List[float]) -> List[int]:
        n = len(sorted_vals)
        if n == 0:
            return [0 for _ in ps]
        out = []
        for p in ps:
            idx = int(p * (n - 1))
            idx = max(0, min(idx, n - 1))
            out.append(sorted_vals[idx])
        return out

    @staticmethod
    def _discretize(v: int, t1: int, t2: int) -> int:
        if v < t1:
            return 0
        if v < t2:
            return 1
        return 2

    @classmethod
    def _discover_field_groups(
        cls,
        events_all: List[MessageEvent],
        max_positions: int,
        min_support: float,
        min_confidence: float,
        n_bins: int,
        max_groups: int,
        enable_range_confirmation: bool,
        range_ratio_threshold: float,
    ) -> Tuple[List[List[int]], Dict[int, Tuple[int, int]]]:
        values_by_pos: Dict[int, List[int]] = {i: [] for i in range(max_positions)}
        for ev in events_all:
            payload = ev.payload or b""
            limit = min(len(payload), max_positions)
            for i in range(limit):
                values_by_pos[i].append(int(payload[i]))

        byte_bins: Dict[int, Tuple[int, int]] = {}
        ranges: Dict[int, int] = {}
        for pos, vals in values_by_pos.items():
            if not vals:
                byte_bins[pos] = (0, 0)
                ranges[pos] = 0
                continue
            sv = sorted(vals)
            if n_bins <= 2:
                t1 = cls._percentiles(sv, [0.5])[0]
                t2 = t1
            else:
                t1, t2 = cls._percentiles(sv, [1 / 3, 2 / 3])
                if t2 < t1:
                    t1, t2 = t2, t1
            byte_bins[pos] = (t1, t2)
            ranges[pos] = max(vals) - min(vals)

        candidate_positions = [p for p in range(max_positions) if values_by_pos[p] and ranges.get(p, 0) > 0]
        candidate_set = set(candidate_positions)
        if not candidate_positions:
            return [], byte_bins

        transactions: List[FrozenSet[Any]] = []
        for ev in events_all:
            payload = ev.payload or b""
            limit = min(len(payload), max_positions)
            items = []
            for i in candidate_positions:
                if i >= limit:
                    break
                t1, t2 = byte_bins[i]
                b = cls._discretize(int(payload[i]), t1, t2)
                items.append((i, b))
            transactions.append(frozenset(items))

        core = AprioriCore()
        fis = core.frequent_itemsets(transactions, min_support=min_support)
        rules = core.association_rules(fis, min_confidence=min_confidence)

        parent: Dict[int, int] = {p: p for p in candidate_positions}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def ok_pair(i: int, j: int) -> bool:
            if not enable_range_confirmation:
                return True
            ri = ranges.get(i, 0)
            rj = ranges.get(j, 0)
            overall = max(ri, rj)
            if overall == 0:
                return True
            return min(ri, rj) <= overall * range_ratio_threshold

        for ante, cons, _sup, _conf in rules:
            ante_pos = {p for p, _v in ante}
            cons_pos = {p for p, _v in cons}
            for p in ante_pos:
                for q in cons_pos:
                    if p in candidate_set and q in candidate_set and abs(p - q) == 1 and ok_pair(p, q):
                        union(p, q)

        groups: Dict[int, List[int]] = {}
        for i in candidate_positions:
            r = find(i)
            groups.setdefault(r, []).append(i)

        field_groups = [sorted(g) for g in groups.values() if len(g) >= 2]
        field_groups.sort(key=lambda g: (-len(g), g[0]))
        field_groups = field_groups[:max_groups]
        return field_groups, byte_bins

    @staticmethod
    def _field_group_feature(payload: bytes, group: List[int]) -> float:
        if not payload:
            return 0.0
        acc = 0
        for i in group:
            if i < len(payload):
                acc = (acc + (i + 1) * payload[i]) % 256
        return float(acc) / 255.0
