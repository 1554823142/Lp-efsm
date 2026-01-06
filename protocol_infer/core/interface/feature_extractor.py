# protocol_infer/core/interface/feature_extractor.py

from abc import ABC, abstractmethod
from typing import List
from protocol_infer.core.datamodel.event import MessageEvent

class FeatureExtractor(ABC):
    """
    特征提取, 根据traces构建出特征向量
    """

    @abstractmethod
    def extract(self, trace: List[MessageEvent]) -> List[List[float]]:
        
        pass
