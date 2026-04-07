from typing import List, Dict

from protocol_infer.control_flow_layer.features.protocol_semantics import (
    extract_protocol_semantic_features,
)
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.core.interface.feature_extractor import FeatureExtractor


class ControlFeatureExtraction(FeatureExtractor):
    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        features = []

        for event in trace:
            payload = event.payload or b""
            payload_len = min(len(payload), 4096) / 4096.0
            direction_val = float(event.direction.to_feature())
            vec = [payload_len, direction_val] + extract_protocol_semantic_features(event)
            features.append(vec)
        return features

    def extract_vars(self, event: MessageEvent) -> Dict[str, float]:
        """
        覆盖父类，提供更有意义的变量名。
        """
        payload = event.payload or b""
        payload_len = min(len(payload), 4096) / 4096.0
        direction_val = float(event.direction.to_feature())
        semantic = extract_protocol_semantic_features(event)

        vars_dict = {
            "pkt_len": payload_len,
            "direction": direction_val,
        }

        # semantic 前 8 个是 b0..b7
        for i in range(min(8, len(semantic))):
            vars_dict[f"b{i}"] = semantic[i]

        # 后面是协议特定的 flags 和 features
        for i in range(8, len(semantic)):
            vars_dict[f"sem_{i-8}"] = semantic[i]

        return vars_dict
