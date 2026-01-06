from typing import List
from sklearn.cluster import AgglomerativeClustering
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm

class HierarchicalClustering(ClusteringAlgorithm):

    def __init__(self, distance_threshold: float):
        self.model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold
        )

    def fit(self, X: List[List[float]]) -> None:
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X: List[List[float]]) -> List[int]:
        return self.labels_.tolist()
