from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm

class DBSCANClustering(ClusteringAlgorithm):
    """
    DBSCAN 聚类算法实现。
    适合未知协议场景，不需要预先指定簇数量，能识别噪声。
    """
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"):
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        self._labels: Optional[np.ndarray] = None

    def fit(self, X: List[List[float]]) -> None:
        X_arr = np.array(X)
        self.model.fit(X_arr)
        self._labels = self.model.labels_

    def predict(self, X: List[List[float]]) -> List[int]:
        # DBSCAN sklearn 实现没有直接的 predict 方法处理新数据
        # 这里在 fit 时已经保存了标签。
        # 如果 X 与 fit 时的输入一致，直接返回；否则需要 fit_predict
        if self._labels is not None and len(X) == len(self._labels):
            return self._labels.tolist()
        
        X_arr = np.array(X)
        return self.model.fit_predict(X_arr).tolist()
