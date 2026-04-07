# protocol_infer/core/interface/feature_extractor.py

from abc import ABC, abstractmethod
from typing import List, Dict
from protocol_infer.core.datamodel.event import MessageEvent

class FeatureExtractor(ABC):
    """
    特征提取, 根据traces构建出特征向量
    """

    @abstractmethod
    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        pass

    def extract_vars(self, event: MessageEvent) -> Dict[str, float]:
        """
        提取具名变量字典，用于 EFSM 学习。
        子类应重写此方法以提供语义化的变量名。
        """
        features = self.extract([event])[0]
        return {f"b{i}": v for i, v in enumerate(features)}
