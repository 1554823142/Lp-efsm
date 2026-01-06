from abc import ABC, abstractmethod
from typing import List, Any

class ClusteringAlgorithm(ABC):
    """
    Abstract base class for all clustering algorithms.
    Input:  feature vectors
    Output: cluster labels
    """

    @abstractmethod
    def fit(self, X: List[List[float]]) -> None:
        """
        Train / prepare the clustering model.

        Parameters:
            X: List of feature vectors
        """
        pass

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Assign each sample to a cluster.

        Returns:
            List of cluster ids (same length as X)
        """
        pass

    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """
        Default implementation: fit then predict.
        Override if algorithm supports direct fit_predict.
        """
        self.fit(X)
        return self.predict(X)
