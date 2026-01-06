from typing import List
from sklearn.cluster import KMeans
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm

class KMeansClustering(ClusteringAlgorithm):

    def __init__(self, n_clusters: int):
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, X: List[List[float]]) -> None:
        self.model.fit(X)

    def predict(self, X: List[List[float]]) -> List[int]:
        return self.model.predict(X).tolist()
