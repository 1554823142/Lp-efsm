import warnings
from typing import List, Optional, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
from protocol_infer.core.algorithm.clustering import ClusteringAlgorithm


class KMeansClustering(ClusteringAlgorithm):
    """
    KMeans 聚类算法。

    n_clusters 可以是：
    - 整数：固定簇数（原有行为）
    - "auto"：用轮廓系数（silhouette score）在 [k_min, k_max] 范围内
      自动搜索最优 K，无需预先指定
    """

    def __init__(
        self,
        n_clusters: Union[int, str] = "auto",
        random_state: Optional[int] = 42,
        k_min: int = 2,
        k_max: int = 20,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.k_min = k_min
        self.k_max = k_max
        self.model: Optional[KMeans] = None
        self.best_k_: Optional[int] = None   # fit 后可查看实际选定的 K

    # ------------------------------------------------------------------
    # 自动 K 选择
    # ------------------------------------------------------------------

    def _select_k(self, X: np.ndarray) -> int:
        """用轮廓系数在 [k_min, k_max] 中选最优 K。"""
        n = len(X)
        # 样本数不足时直接返回能支持的最大 K
        k_lo = self.k_min
        k_hi = min(self.k_max, n - 1)   # silhouette 要求 K < n_samples
        if k_hi < k_lo:
            return max(k_lo, 2)

        best_k = k_lo
        best_score = -2.0

        for k in range(k_lo, k_hi + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = km.fit_predict(X)
            # 所有点落入同一簇时轮廓系数无意义，跳过
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels, sample_size=min(2000, n))
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    # ------------------------------------------------------------------
    # ClusteringAlgorithm 接口
    # ------------------------------------------------------------------

    def fit(self, X: List[List[float]]) -> None:
        X_arr = np.array(X, dtype=float)

        if self.n_clusters == "auto":
            self.best_k_ = self._select_k(X_arr)
        else:
            self.best_k_ = int(self.n_clusters)

        self.model = KMeans(
            n_clusters=self.best_k_,
            random_state=self.random_state,
            n_init="auto",
        )
        self.model.fit(X_arr)

    def predict(self, X: List[List[float]]) -> List[int]:
        if self.model is None:
            raise RuntimeError("KMeansClustering not fitted yet")
        return self.model.predict(np.array(X, dtype=float)).tolist()
