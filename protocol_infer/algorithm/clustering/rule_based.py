from typing import List, Tuple
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm

class RuleBasedClustering(ClusteringAlgorithm):
    """
    基于规则的聚类
    
    待改进:
    目前仅实现每个唯一的数据向量视为一个独立的簇
    """

    def __init__(self):
        self.cluster_map = {}
        self.next_id = 0

    def fit(self, X: List[List[float]]) -> None:
        for vec in X:
            key = tuple(vec)
            if key not in self.cluster_map:
                self.cluster_map[key] = self.next_id
                self.next_id += 1

    def predict(self, X: List[List[float]]) -> List[int]:
        labels = []
        for vec in X:
            key = tuple(vec)
            labels.append(self.cluster_map[key])
        return labels
