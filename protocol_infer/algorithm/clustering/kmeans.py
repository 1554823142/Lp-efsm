from typing import List, Optional
from sklearn.cluster import KMeans
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm

class KMeansClustering(ClusteringAlgorithm):

    def __init__(self, n_clusters: int, random_state: Optional[int] = 42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")

    def fit(self, X: List[List[float]]) -> None:
        self.model.fit(X)

    def predict(self, X: List[List[float]]) -> List[int]:
        return self.model.predict(X).tolist()
