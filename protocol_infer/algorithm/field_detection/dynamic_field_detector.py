"""
动态字段检测器

在 Apriori 已选字段（静态字段）的基础上，通过对同类消息进行字节序列差分分析，
发现被 Apriori 遗漏的结构化动态字段。支持单字节和多字节（大端/小端）字段。

核心思路
--------
Apriori 通过"某字节位置上某个值出现频繁"来选位置，因此天然遗漏两类字段：
  1. 序列字段：每条消息值都不同（如计数器），没有任何单个值频繁
  2. 低基数但无主导值：只有 2~3 种取值，但均匀分布，没有一个值的支持度超过阈值

本模块的策略
------------
对每个消息类型（symbol）组内的所有 payload，逐字节位置扫描：
  - 跳过 Apriori 已知位置（避免重复）
  - 单字节：提取值序列，判断是否"结构化"（序列/低基数/有界范围）
  - 多字节（2/4字节，大端+小端）：仅当至少有一个组成字节不在已知集合时才尝试，
    组合成整数值后再做结构化判断
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from protocol_infer.core.datamodel.event import MessageEvent


@dataclass
class DynamicField:
    """描述一个检测到的动态字段（单字节或多字节）。"""

    start_pos: int   # payload 中的起始字节偏移
    width: int       # 字节数（1 / 2 / 4）
    endian: str      # 'big' 或 'little'（width==1 时无实际意义，统一存 'big'）

    @property
    def var_name(self) -> str:
        """生成变量名，与 b{pos} / s{pos} 命名空间不冲突。"""
        if self.width == 1:
            return f"dyn_{self.start_pos}"
        suffix = "b" if self.endian == "big" else "l"
        return f"dyn_{self.start_pos}_{self.width}{suffix}"

    @property
    def covers(self) -> Set[int]:
        """该字段覆盖的字节偏移集合。"""
        return set(range(self.start_pos, self.start_pos + self.width))

    def extract_value(self, payload: bytes) -> Optional[float]:
        """从 payload 中提取该字段的数值，payload 不够长时返回 None。"""
        end = self.start_pos + self.width
        if end > len(payload):
            return None
        return float(int.from_bytes(payload[self.start_pos:end], self.endian)) # pyright: ignore[reportArgumentType]

    def __hash__(self):
        return hash((self.start_pos, self.width, self.endian))

    def __eq__(self, other):
        if not isinstance(other, DynamicField):
            return False
        return (self.start_pos, self.width, self.endian) == (
            other.start_pos, other.width, other.endian,
        )


class DynamicFieldDetector:
    """
    在 Apriori 已知字段位置的基础上，增量发现结构化动态字段。

    Parameters
    ----------
    max_scan_bytes : int
        扫描 payload 的最大字节数（防止超长 payload 爆炸）。
    max_field_width : int
        支持的最大字段宽度（字节），取值 1/2/4。
    min_samples : int
        触发某个符号/转移的最小样本数，样本不足时跳过检测。
    sequential_delta_tol : float
        判定等差序列时允许的 delta 误差阈值。
    low_cardinality_threshold : int
        唯一值数量不超过此值时视为"低基数"结构化字段。
    bounded_span_ratio : float
        span / max_possible < 此比例时视为"有界范围"结构化字段。
    """

    def __init__(
        self,
        max_scan_bytes: int = 20,
        max_field_width: int = 4,
        min_samples: int = 4,
        sequential_delta_tol: float = 1e-6,
        low_cardinality_threshold: int = 16,
        bounded_span_ratio: float = 0.25,
    ):
        self.max_scan_bytes = max_scan_bytes
        self.max_field_width = max_field_width
        self.min_samples = min_samples
        self.sequential_delta_tol = sequential_delta_tol
        self.low_cardinality_threshold = low_cardinality_threshold
        self.bounded_span_ratio = bounded_span_ratio

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def detect_from_symbol_groups(
        self,
        symbol_events: Dict[str, List[MessageEvent]],
        known_positions: Set[int],
    ) -> List[DynamicField]:
        """
        对按消息类型（symbol）分组的事件集合做动态字段检测。

        Parameters
        ----------
        symbol_events : {symbol -> [MessageEvent, ...]}
            每个消息类型对应的所有事件。
        known_positions : Set[int]
            Apriori 已选位置，单字节扫描时跳过这些位置。
            （多字节检测时，若组成字节不全已知，仍会尝试。）

        Returns
        -------
        List[DynamicField]
            检测到的所有动态字段（跨 symbol 去重）。
        """
        # 用 dict 去重：key=(start_pos, width, endian)
        detected: Dict[Tuple[int, int, str], DynamicField] = {}

        for symbol, events in symbol_events.items():
            payloads = [ev.payload or b"" for ev in events]
            if len(payloads) < self.min_samples:
                continue

            max_len = max((len(p) for p in payloads), default=0)
            scan_limit = min(max_len, self.max_scan_bytes)
            if scan_limit == 0:
                continue

            # ── 第一趟：单字节扫描 ──────────────────────────────────────
            # 记录本 symbol 内找到的单字节动态位置，用于后续多字节剪枝
            single_found: Set[int] = set()

            for pos in range(scan_limit):
                if pos in known_positions:
                    continue  # Apriori 已处理，跳过

                values = self._extract_single(payloads, pos)
                if len(values) < self.min_samples:
                    continue

                if self._is_structured(values, bits=8):
                    key = (pos, 1, "big")
                    if key not in detected:
                        detected[key] = DynamicField(start_pos=pos, width=1, endian="big")
                    single_found.add(pos)

            # ── 第二趟：多字节扫描（width=2, 4）────────────────────────
            if self.max_field_width < 2:
                continue

            explained = known_positions | single_found  # 已充分解释的位置

            for width in [w for w in [2, 4] if w <= self.max_field_width]:
                for pos in range(scan_limit - width + 1):
                    component_positions = set(range(pos, pos + width))

                    # 若所有组成字节均已被解释，无需添加多字节字段
                    if component_positions.issubset(explained):
                        continue

                    for endian in ("big",):
                        values = self._extract_multi(payloads, pos, width, endian)
                        if len(values) < self.min_samples:
                            continue

                        if self._is_structured(values, bits=width * 8):
                            key = (pos, width, endian)
                            if key not in detected:
                                detected[key] = DynamicField(
                                    start_pos=pos, width=width, endian=endian
                                )

        # 后处理去重：若多字节字段的所有组成字节均已被单字节字段或 known_positions 覆盖，
        # 则该多字节字段是冗余的（不提供额外信息），予以移除。
        # 例如：dyn_1（1字节）已覆盖 b1，dyn_0_2b（2字节 b0+b1）中 b0 已在 known_positions，
        # b1 已由 dyn_1 覆盖，故 dyn_0_2b 冗余，移除。
        single_byte_positions = {pos for (pos, width, _) in detected if width == 1}
        fully_explained = known_positions | single_byte_positions
        detected = {
            key: field
            for key, field in detected.items()
            if field.width == 1 or not field.covers.issubset(fully_explained)
        }

        # 端序去重：同一 (pos,width) 同时检测到 big/little 时，仅保留一个。
        # 为了展示/推断一致性，这里全局只保留 big-endian（网络字节序）。
        by_window: Dict[Tuple[int, int], List[Tuple[Tuple[int, int, str], DynamicField]]] = {}
        for key, field in detected.items():
            by_window.setdefault((field.start_pos, field.width), []).append((key, field))

        pruned: Dict[Tuple[int, int, str], DynamicField] = {}
        for (pos, width), items in by_window.items():
            if width == 1 or len(items) == 1:
                for k, f in items:
                    pruned[k] = f
                continue

            best_key = None
            for k, f in items:
                if f.endian == "big":
                    best_key = k
                    break
            if best_key is None:
                best_key = items[0][0]

            pruned[best_key] = detected[best_key]

        detected = pruned

        return list(detected.values())

    # ------------------------------------------------------------------
    # 值提取辅助
    # ------------------------------------------------------------------

    def _extract_single(self, payloads: List[bytes], pos: int) -> List[float]:
        return [
            float(p[pos]) for p in payloads if pos < len(p)
        ]

    def _extract_multi(
        self, payloads: List[bytes], pos: int, width: int, endian: str
    ) -> List[float]:
        end = pos + width
        return [
            float(int.from_bytes(p[pos:end], endian))
            for p in payloads
            if end <= len(p)
        ]

    # ------------------------------------------------------------------
    # 结构化判断
    # ------------------------------------------------------------------

    def _is_structured(self, values: List[float], bits: int) -> bool:
        """
        判断一组值是否具有"结构化动态"特征，即对 EFSM 守卫学习有价值。

        满足以下任一条件即认为结构化：
          1. 等差序列（固定 delta）—— 序列号、计数器
          2. 低基数（唯一值数量少）—— Apriori 因无支配值而遗漏的离散字段
          3. 有界范围（span 远小于字段最大值域）—— 受限的动态数值
        """
        n = len(values)
        if n < self.min_samples:
            return False

        unique = set(values)

        # 纯常量由 Apriori 已处理，这里跳过，避免重复引入
        if len(unique) == 1:
            return False

        # ── 条件1：等差序列 ──────────────────────────────────────────
        if n >= 2:
            deltas = [values[i + 1] - values[i] for i in range(n - 1)]
            if max(abs(d - deltas[0]) for d in deltas) < self.sequential_delta_tol:
                return True

        # ── 条件2：低基数 ───────────────────────────────────────────
        if len(unique) <= self.low_cardinality_threshold:
            return True

        # ── 条件3：有界范围 ─────────────────────────────────────────
        span = max(values) - min(values)
        max_possible = float((1 << bits) - 1)
        if span < max_possible * self.bounded_span_ratio:
            return True

        return False
