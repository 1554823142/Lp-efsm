from typing import List, Tuple, FrozenSet, Any
from protocol_infer.core.interface.feature_extractor import FeatureExtractor
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.apriori.miners import StaticFieldMiner, BytePositionTransactionBuilder, StaticFieldInterpreter

class AprioriFeatureExtraction(FeatureExtractor):
    def __init__(
        self,
        positions: List[int],
        itemsets: List[FrozenSet[Tuple[int, int]]],
        max_payload_len: float = 1.0,
        onehot_weight: float = 2.0
    ):
        self.positions = positions
        self.itemsets = itemsets
        self.max_payload_len = max_payload_len
        self.onehot_weight = onehot_weight
        self._miner: Optional[StaticFieldMiner] = None  # 由 from_events 填充

    @classmethod                    # 可能不需要预先创建类实例
    def from_events(
        cls,                        # 类似self,接收类本身 
        events_all: List[MessageEvent],
        max_positions: int = 16,
        max_itemsets: int = 8,
        min_support: float = 0.3,
        onehot_weight: float = 2.0,
        global_static_threshold: float = 0.95
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

        instance = cls(
            positions=positions,
            itemsets=itemsets,
            max_payload_len=float(max_payload_len),
            onehot_weight=onehot_weight,
        )
        instance._miner = miner
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

            vec = pos_vals + onehot + [length_val, direction_val]
            features.append(vec)
        return features
