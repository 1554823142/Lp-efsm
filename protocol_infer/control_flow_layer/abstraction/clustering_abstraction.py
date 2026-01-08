from protocol_infer.core.interface.message_abstraction import MessageAbstractor
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm
from typing import List

class ClusterMessageAbstractor(MessageAbstractor):

    def __init__(self, algorithm: ClusteringAlgorithm):
        self.algorithm = algorithm
        self._trained = False

    def fit(self, features: List[List[float]]) -> None:
        self.algorithm.fit(features)
        self._trained = True

    def abstract(self, feature: List[float]) -> str:
        if not self._trained:
            raise RuntimeError("Abstractor not fitted")

        label = self.algorithm.predict([feature])[0]
        return f"C{label}"
