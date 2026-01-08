from typing import List
from protocol_infer.core.interface.feature_extractor import FeatureExtractor
from protocol_infer.core.datamodel.event import MessageEvent

class ControlFeatureExtraction(FeatureExtractor):
    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        features = []

        for event in trace:
            # 暂时考虑: 负载量, 端口号, 方向


            # 处理 payload
            if event.payload and len(event.payload) > 0:
                payload_value = float(len(event.payload))  # 使用长度而不是内容
            else:
                payload_value = 0.0  # 空负载设为0
           
            vec = [
                float(payload_value),
                float(event.session_key.port1),
                float(event.session_key.port2),
                float(event.direction.to_feature()),
            ]
            features.append(vec)
        return features