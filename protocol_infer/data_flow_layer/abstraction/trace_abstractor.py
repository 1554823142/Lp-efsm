from typing import List, Dict, Optional, Tuple, Any
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.interface.message_abstraction import MessageAbstractor
from protocol_infer.core.datamodel.abstract_message import AbstractMessage
from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor

class AbstractionProcessor:
    def __init__(
        self,
        abstractor: Optional[MessageAbstractor] = None,
        default_n_clusters: int = 8,
    ):
        self.abstractor = abstractor
        self.default_n_clusters = default_n_clusters

    def fit_and_abstract(self, trace: Trace, sessions: Dict, feature_processor: FeatureProcessor):
        abstractor = self.abstractor
        if abstractor is None:
            from protocol_infer.control_flow_layer.abstraction.clustering_abstraction import ClusterMessageAbstractor
            from protocol_infer.algorithm.clustering.dbscan import DBSCANClustering
            abstractor = ClusterMessageAbstractor(DBSCANClustering())

        all_vars: List[List[float]] = []
        ev_vars_pairs: List[Tuple[Any, Dict[str, float], List[float]]] = []

        for events in sessions.values():
            for ev in events:
                vars_dict = feature_processor.extract_vars(ev)
                var_list = feature_processor.var_list(vars_dict)
                all_vars.append(var_list)
                ev_vars_pairs.append((ev, vars_dict, var_list))

        if not all_vars:
            trace.abstract_messages = []
            return trace

        abstractor.fit(all_vars)

        abstract_msgs: List[AbstractMessage] = []
        for ev, vars_dict, var_list in ev_vars_pairs:
            symbol = abstractor.abstract(var_list)
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
