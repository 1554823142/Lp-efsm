import random
from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from protocol_infer.apriori.core import AprioriCore
from protocol_infer.apriori.miners import BytePositionTransactionBuilder, StaticFieldInterpreter, StaticFieldMiner
from protocol_infer.control_flow_layer.features.protocol_semantics import extract_protocol_semantic_features
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.core.interface.feature_extractor import FeatureExtractor


class AprioriFeatureExtraction(FeatureExtractor):
    def __init__(
        self,
        positions: List[int],
        itemsets: List[FrozenSet[Tuple[int, int]]],
        field_groups: Optional[List[List[int]]] = None,
        static_items: Optional[Dict[int, int]] = None,
        max_payload_len: float = 1.0,
        onehot_weight: float = 2.0,
    ):
        self.positions = positions
        self.itemsets = itemsets
        self.field_groups = field_groups or []
        self.static_items = static_items or {}
        self.max_payload_len = max_payload_len
        self.onehot_weight = onehot_weight
        self._miner: Optional[StaticFieldMiner] = None
        self._byte_bins: Dict[int, Tuple[int, int]] = {}

    @classmethod
    def from_events(
        cls,
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
        max_mining_events: int = 2048,
        mining_seed: int = 42,
        scan_positions: int = 24,
    ) -> "AprioriFeatureExtraction":
        mining_events = cls._sample_events(events_all, limit=max_mining_events, seed=mining_seed)
        seed_positions = [p for p, _ in cls._select_informative_positions(mining_events, scan_positions)[:max_positions]]
        if not seed_positions:
            seed_positions = list(range(min(max_positions, scan_positions)))

        miner = StaticFieldMiner(
            builder=BytePositionTransactionBuilder(max_positions=scan_positions, positions=seed_positions),
            interpreter=StaticFieldInterpreter(global_static_threshold=global_static_threshold),
            min_support=min_support,
        )
        mined_itemsets = cls._mine_multiview_itemsets(
            miner=miner,
            events_all=mining_events,
            min_subset_size=max(24, min(128, len(mining_events) // 10 if mining_events else 24)),
        )
        static_items = cls._extract_static_items(miner.get_global_static_items())
        static_positions = set(static_items.keys())

        positions = cls._rank_positions(
            events_all=mining_events,
            mined_itemsets=mined_itemsets,
            max_positions=max_positions,
            scan_positions=scan_positions,
            seed_positions=seed_positions,
            excluded_positions=static_positions,
        )

        top_itemsets = sorted(
            mined_itemsets,
            key=lambda x: (-x[1], -len(x[0]), tuple(sorted(x[0]))),
        )[:max_itemsets]
        itemsets = [fs for fs, _ in top_itemsets]
        max_payload_len = max((len(e.payload or b"") for e in events_all), default=1)

        field_groups: List[List[int]] = []
        byte_bins: Dict[int, Tuple[int, int]] = {}
        if enable_field_groups:
            field_groups, byte_bins = cls._discover_field_groups(
                events_all=mining_events,
                max_positions=scan_positions,
                min_support=field_group_min_support,
                min_confidence=field_group_min_confidence,
                n_bins=field_group_bins,
                max_groups=field_group_max_groups,
                enable_range_confirmation=enable_range_confirmation,
                range_ratio_threshold=range_ratio_threshold,
                candidate_positions=positions,
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
        instance.static_items = static_items
        return instance

    @staticmethod
    def _sample_events(events_all: List[MessageEvent], limit: int, seed: int) -> List[MessageEvent]:
        if limit <= 0 or len(events_all) <= limit:
            return list(events_all)
        rng = random.Random(seed)
        sample_idx = sorted(rng.sample(range(len(events_all)), limit))
        return [events_all[i] for i in sample_idx]

    @classmethod
    def _mine_multiview_itemsets(
        cls,
        miner: StaticFieldMiner,
        events_all: List[MessageEvent],
        min_subset_size: int,
    ) -> List[Tuple[FrozenSet[Tuple[int, int]], float]]:
        merged: Dict[FrozenSet[Tuple[int, int]], float] = {}
        total = max(len(events_all), 1)

        def absorb(events_subset: List[MessageEvent]) -> None:
            if not events_subset:
                return
            for fs, support in miner.mine(events_subset):
                weighted = support * (len(events_subset) / total)
                prev = merged.get(fs, 0.0)
                if weighted > prev:
                    merged[fs] = weighted

        absorb(events_all)

        by_dir: Dict[object, List[MessageEvent]] = defaultdict(list)
        for ev in events_all:
            by_dir[ev.direction].append(ev)
        for subset in by_dir.values():
            if len(subset) >= min_subset_size:
                absorb(subset)

        return [(fs, sup) for fs, sup in merged.items()]

    @classmethod
    def _rank_positions(
        cls,
        events_all: List[MessageEvent],
        mined_itemsets: List[Tuple[FrozenSet[Tuple[int, int]], float]],
        max_positions: int,
        scan_positions: int,
        seed_positions: Optional[List[int]] = None,
        excluded_positions: Optional[set[int]] = None,
    ) -> List[int]:
        excluded_positions = excluded_positions or set()
        scores: Dict[int, float] = defaultdict(float)
        for fs, support in mined_itemsets:
            weight = support * max(len(fs), 1)
            for pos, _val in fs:
                if pos not in excluded_positions:
                    scores[pos] += weight

        if seed_positions:
            for pos in seed_positions:
                if pos not in excluded_positions:
                    scores[pos] += 0.05

        for pos, score in cls._select_informative_positions(events_all, scan_positions):
            if pos not in excluded_positions:
                scores[pos] += score

        if not scores:
            fallback = [p for p in range(min(max_positions, scan_positions)) if p not in excluded_positions]
            return fallback[:max_positions]

        ranked = sorted(scores, key=lambda p: (-scores[p], p))[:max_positions]
        return sorted(ranked)

    @staticmethod
    def _select_informative_positions(
        events_all: List[MessageEvent],
        scan_positions: int,
    ) -> List[Tuple[int, float]]:
        ranked: List[Tuple[int, float]] = []
        total = max(len(events_all), 1)
        for pos in range(scan_positions):
            counts: Dict[int, int] = defaultdict(int)
            present = 0
            for ev in events_all:
                payload = ev.payload or b""
                if pos >= len(payload):
                    continue
                present += 1
                counts[int(payload[pos])] += 1
            if present < max(8, int(total * 0.2)):
                continue
            unique = len(counts)
            if unique <= 1:
                continue
            unique_ratio = unique / present
            if unique_ratio > 0.35:
                continue
            max_prob = max(counts.values()) / present
            coverage = present / total
            score = coverage * (1.0 - max_prob) * (1.0 - unique_ratio)
            if score > 0:
                ranked.append((pos, score))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        features: List[List[float]] = []
        for event in trace:
            # 内部复用 extract_vars 的逻辑并转换为 flat list，保持与聚类层兼容
            vars_dict = self.extract_vars(event)
            # 注意：此处必须保证顺序一致，与之前 extract() 的 vec 结构对应
            # 但聚类层其实不关心 key，只要同一批次的顺序一致即可
            vec = list(vars_dict.values())
            features.append(vec)
        return features

    def extract_vars(self, event: MessageEvent) -> Dict[str, float]:
        """
        提取具名变量。
        b{offset} -> 原始字节位
        onehot_{idx} -> 项集匹配
        group_{idx} -> 字段组
        pkt_len, direction -> 基础特征
        b0..b7 -> 头部语义
        sem_{idx} -> 协议特定语义
        """
        payload = event.payload or b""
        vars_dict = {}

        # 1. 核心字节位置 (b{offset})
        for pos in self.positions:
            vars_dict[f"b{pos}"] = float(payload[pos]) / 255.0 if pos < len(payload) else 0.0

        # 2. 项集 One-hot
        for i, fs in enumerate(self.itemsets):
            matched = all(p < len(payload) and payload[p] == v for p, v in fs)
            vars_dict[f"onehot_{i}"] = self.onehot_weight if matched else 0.0

        # 3. 字段组
        for i, group in enumerate(self.field_groups):
            vars_dict[f"group_{i}"] = self._field_group_feature(payload, group)

        # 4. 基础特征
        vars_dict["pkt_len"] = len(payload) / max(self.max_payload_len, 1.0)
        vars_dict["direction"] = float(event.direction.to_feature())

        # 5. 协议语义
        semantic = extract_protocol_semantic_features(event)
        # semantic 前 8 个强制映射为 b0..b7，确保能被 eval 对齐
        for i in range(min(8, len(semantic))):
            vars_dict[f"b{i}"] = semantic[i]

        for i in range(8, len(semantic)):
            vars_dict[f"sem_{i-8}"] = semantic[i]

        return vars_dict

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
        candidate_positions: Optional[List[int]] = None,
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

        if candidate_positions is None:
            candidate_positions = [p for p in range(max_positions) if values_by_pos[p] and ranges.get(p, 0) > 0]
        else:
            candidate_positions = [p for p in candidate_positions if values_by_pos.get(p) and ranges.get(p, 0) > 0]
        candidate_set = set(candidate_positions)
        if not candidate_positions:
            return [], byte_bins

        transactions: List[FrozenSet[Any]] = []
        for ev in events_all:
            payload = ev.payload or b""
            items = []
            for i in candidate_positions:
                if i >= len(payload):
                    continue
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
