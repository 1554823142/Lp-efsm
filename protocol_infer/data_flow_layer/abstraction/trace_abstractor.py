from typing import List, Dict, Optional
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.interface.message_abstraction import MessageAbstractor
from protocol_infer.core.datamodel.abstract_message import AbstractMessage
from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor

class AbstractionProcessor:
    def __init__(self, abstractor: Optional[MessageAbstractor] = None):
        self.abstractor = abstractor

    def fit_and_abstract(self, trace: Trace, sessions: Dict, feature_processor: FeatureProcessor):
        # 默认抽象器
        abstractor = self.abstractor
        if abstractor is None:
            from protocol_infer.control_flow_layer.abstraction.clustering_abstraction import ClusterMessageAbstractor
            from protocol_infer.algorithm.clustering.kmeans import KMeansClustering
            abstractor = ClusterMessageAbstractor(KMeansClustering(n_clusters=8))

        all_vars = []
        cached_vars = {}                # 缓存每个事件的变量

        for events in sessions.values():
            for ev in events:
                vars_dict = feature_processor.extract_vars(ev)
                cached_vars[ev] = vars_dict
                # 收集所有变量, 用于训练抽象器
                all_vars.append(feature_processor.var_list(vars_dict))

        if all_vars:
            abstractor.fit(all_vars)       # 训练抽象器

        # 生成抽象消息, 将变量转换为符号
        abstract_msgs: List[AbstractMessage] = []
        for events in sessions.values():
            for ev in events:
                vars_dict = cached_vars[ev]
                symbol = abstractor.abstract(feature_processor.var_list(vars_dict))
                abstract_msgs.append(
                    AbstractMessage(
                        session_key=ev.session_key,
                        timestamp=ev.timestamp,
                        symbol=symbol,
                        vars=vars_dict,
                        direction=ev.direction
                    )
                )

        trace.abstract_messages = abstract_msgs
        return trace