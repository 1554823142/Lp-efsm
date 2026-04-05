
from collections import defaultdict
from typing import Dict, List, Optional, Union
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.features.control_feature_extraction import ControlFeatureExtraction
from protocol_infer.control_flow_layer.abstraction.clustering_abstraction import ClusterMessageAbstractor
from protocol_infer.algorithm.clustering.kmeans import KMeansClustering
from protocol_infer.algorithm.clustering.dbscan import DBSCANClustering
from protocol_infer.control_flow_layer.inference.pta_infer import PTAInfer
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.fsm import FSM
from protocol_infer.algorithm.states_merging.K_tails import KTailStateMerger

class ControlFlowPipeline:
    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: Union[int, str] = "auto",
        k: int = 4,
        eps: float = 0.5,
        min_samples: int = 5,
        use_apriori: bool = True
    ):
        self.use_apriori = use_apriori

        # 1. 选择聚类算法
        if algorithm == "kmeans":
            self.clusterer = KMeansClustering(n_clusters=n_clusters, random_state=42)
        elif algorithm == "dbscan":
            self.clusterer = DBSCANClustering(eps=eps, min_samples=min_samples)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.abstractor = ClusterMessageAbstractor(self.clusterer)
        
        # 默认特征提取器，若开启 Apriori 会在 run 时被动态替换或增强
        self.featureer = ControlFeatureExtraction()
        
        self.inferer = PTAInfer()
        self.merger = KTailStateMerger(k)
        
        # 缓存sessions，供其他层使用
        self._sessions: Dict[SessionKey, List] = None
        # 缓存每会话的(events, features)
        self._sess_features: Dict[SessionKey, tuple] = None
        # 缓存 apriori 发现的位置
        self._apriori_positions: Optional[List[int]] = None
        self._apriori_static_items: Optional[Dict[int, int]] = None

    def run_from_pcap(self, pcap_path: str) -> FSM:
        trace = PCAPPipeline().run(pcap_path)
        return self.run(trace)

    def run(self, trace: Trace) -> FSM:
        # 1. 按照 session_key 分桶
        sessions = defaultdict(list)
        for ev in trace.events:
            sessions[ev.session_key].append(ev)
        
        # 为每个会话排序事件
        for sk, events in sessions.items():
            events.sort(key=lambda e: e.timestamp)
        
        # 缓存sessions供其他层使用
        self._sessions = dict(sessions)

        # 2. 如果开启了 Apriori，先进行伪字段发现
        if self.use_apriori:
            from protocol_infer.control_flow_layer.features.apriori_feature_extraction import AprioriFeatureExtraction
            # 使用全量事件发现静态字段组合 (伪字段)
            self.featureer = AprioriFeatureExtraction.from_events(trace.events)
            self._apriori_positions = self.featureer.positions
            self._apriori_static_items = getattr(self.featureer, "static_items", None)

        # 3. 提取特征
        all_features = []
        sess_features = {}
        for sk, events in sessions.items():
            features = self.featureer.extract(events)
            sess_features[sk] = (events, features)
            all_features.extend(features)

        # 缓存(sess_features)供数据流层复用
        self._sess_features = sess_features

        # 4. 训练聚类模型 (Message Abstraction)
        if len(all_features) == 0:
            raise RuntimeError("no events found")
        self.abstractor.fit(all_features)
        if isinstance(self.clusterer, KMeansClustering) and self.clusterer.best_k_ is not None:
            mode = "auto" if self.clusterer.n_clusters == "auto" else "fixed"
            print(f"[ControlFlow] KMeans K={self.clusterer.best_k_} "
                  f"(mode={mode}, samples={len(all_features)})")

        # 5. 生成符号序列
        sequences = {}
        for sk, (events, features) in sess_features.items():
            symbols = [self.abstractor.abstract(f) for f in features]
            sequences[sk] = symbols

        # 6. 推断初始 FSM (PTA)
        fsm = self.inferer.infer(sequences)

        # 7. 状态合并 (K-tails)
        fsm = self.merger.merge(fsm)
        
        return fsm
    
    def get_sessions(self) -> Dict[SessionKey, List]:
        """
        获取处理后的sessions，供其他层使用
        
        Returns:
            按session_key分组的events字典，如果还没有运行过run方法则返回None
        """
        return self._sessions

    def get_sess_features(self) -> Dict[SessionKey, tuple]:
        """
        获取每个会话的(events, features)，供其他层复用符号特征
        """
        return self._sess_features

    def get_apriori_positions(self) -> Optional[List[int]]:
        """
        获取 Apriori 发现的有效载荷偏移位置，供数据流层复用
        """
        return self._apriori_positions

    def get_apriori_static_items(self) -> Optional[Dict[int, int]]:
        return self._apriori_static_items
