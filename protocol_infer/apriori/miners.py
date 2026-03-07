from typing import List, Tuple, FrozenSet, Any, Dict, Callable, Optional
from collections import defaultdict
from protocol_infer.core.datamodel.event import MessageEvent
from .core import AprioriCore
from .interfaces import TransactionBuilder, ResultInterpreter


class BytePositionTransactionBuilder(TransactionBuilder[MessageEvent]):
    '''
        未知协议
    '''
    def __init__(self, max_positions: int = 16):
        self.max_positions = max_positions

    def build(self, items: List[MessageEvent]) -> List[FrozenSet[Any]]:
        '''
        原始字节位置场景
        每个事件的 transaction 是其 payload 中
        前 max_positions 个(字节偏移, 字节值)的集合。
        '''
        transactions = []
        for ev in items:
            payload = ev.payload or b""
            limit = min(len(payload), self.max_positions)
            transactions.append(frozenset((i, payload[i]) for i in range(limit)))
        return transactions


class DictTransactionBuilder(TransactionBuilder[MessageEvent]):
    """
    已解析字段场景
    extractor: 从 MessageEvent 提取已解析字段 dict 的函数。
               如果为 None，尝试读取 ev.parsed_fields。
    """
    def __init__(
        self,
        extractor: Optional[Callable[[MessageEvent], Dict[str, Any]]] = None,   # 字段提取函数, 返回一个dict
        discrete_threshold: int = 20,               # 离散字段的唯一值数量阈值
        n_buckets: int = 5,                         # 连续字段的分桶数量
        exclude_fields: Optional[List[str]] = None, # 排除的字段, 变化频率高的字段直接排除在外
    ):  
        self.extractor = extractor or (lambda ev: getattr(ev, "parsed_fields", {}) or {})
        self.discrete_threshold = discrete_threshold
        self.n_buckets = n_buckets
        self.exclude_fields = set(exclude_fields or [])         # 防止传入None
        self._field_stats: Dict[str, Any] = {}                       # 调用fit()进行填充

    def fit(self, items: List[MessageEvent]) -> "DictTransactionBuilder":
        """扫描全量数据，统计字段分布，决定离散化策略"""
        field_values: Dict[str, list] = defaultdict(list)
        # 1. 收集所有字段值
        for ev in items:
            fields = self.extractor(ev)
            for k, v in fields.items():
                if k not in self.exclude_fields:
                    field_values[k].append(v)       # 收集字段值

        # 2. 统计字段分布, 决定字段的类型(离散/连续)
        for field, values in field_values.items():
            unique = set(values)
            if len(unique) <= self.discrete_threshold:      # 唯一值数量小, 则视为离散字段
                self._field_stats[field] = {"type": "discrete"}
            else:                                           # 唯一值数量大, 则视为连续字段      
                try:            # 等频分桶为了每个桶里的样本量大致相同
                    sorted_vals = sorted(values)
                    n = len(sorted_vals)
                    boundaries = [                      # 计算分桶边界
                        sorted_vals[int(i * n / self.n_buckets)]
                        for i in range(1, self.n_buckets)
                    ]
                    self._field_stats[field] = {
                        "type": "continuous",
                        "boundaries": boundaries,
                    }
                except (TypeError, ValueError):
                    # 如果不可排序，作为离散处理
                    self._field_stats[field] = {"type": "discrete"}
        return self

    def build(self, items: List[MessageEvent]) -> List[FrozenSet[Any]]:
        if not self._field_stats:
            # 如果没有 fit，尝试现场扫描, 按顺序调用则不会进入此分支
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
                    items_set.add((k, v))           # 直接用原始值
                else:
                    bucket = self._discretize(v, stat["boundaries"])
                    items_set.add((k, bucket))      # 用分桶索引替代原始值
            transactions.append(frozenset(items_set))
        return transactions

    def _discretize(self, value, boundaries: list) -> int:
        for i, b in enumerate(boundaries):
            try:
                if value < b:           # 就是找到第一个小于 value 的分桶边界
                    return i
            except (TypeError, ValueError):
                continue
        return len(boundaries)          # 如果都不满足, 则放到最后一个桶


class StaticFieldInterpreter(ResultInterpreter[List[FrozenSet]]):
    def __init__(self, global_static_threshold: float = 0.95):
        """
        global_static_threshold: support 超过此值的单项集视为全局静态字段，过滤掉
        即去除基本每次都会出现的字段, 对区分消息无作用
        """
        self.global_static_threshold = global_static_threshold

    def interpret(
        self, frequent_itemsets: List[Tuple[FrozenSet, float]]
    ) -> List[Tuple[FrozenSet[Any], float]]:
        core = AprioriCore()

        # 找出全局静态字段（所有消息都有的，无区分能力）
        global_static_items = {
            next(iter(fs))
            for fs, sup in frequent_itemsets
            if len(fs) == 1 and sup >= self.global_static_threshold         # sup : 支持率
        }

        # 过滤掉包含全局静态字段的项集
        filtered = [
            (fs, sup) for fs, sup in frequent_itemsets
            if not any(item in global_static_items for item in fs)      # 只要项集中有一个全局则过滤
        ]

        return core.maximal_itemsets(filtered)      # 取最大频繁项集(不被其他频繁项集包含的项集)


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

    def mine(self, events: List[MessageEvent]) -> List[Tuple[FrozenSet[Any], float]]:
        # DictTransactionBuilder 需要先 fit, 即先构建字段统计信息
        if hasattr(self.builder, "fit"):
            self.builder.fit(events)

        # 1. MessageEvent 转换为 项集
        transactions = self.builder.build(events)    
        # 2. 挖掘频繁项集   
        fis = self.core.frequent_itemsets(transactions, self.min_support)  
        # 3. 解释频繁项集, 过滤全局静态字段    
        return self.interpreter.interpret(fis)          
