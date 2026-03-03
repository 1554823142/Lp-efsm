from collections import defaultdict
from typing import Dict, List
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.features.control_feature_extraction import ControlFeatureExtraction
from protocol_infer.control_flow_layer.abstraction.clustering_abstraction import ClusterMessageAbstractor
from protocol_infer.algorithm.clustering.kmeans import KMeansClustering
from protocol_infer.control_flow_layer.inference.pta_infer import PTAInfer
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.fsm import FSM
from protocol_infer.algorithm.states_merging.K_tails import KTailStateMerger

class ControlFlowPipeline:
    def __init__(self, n_clusters: int = 8, k: int = 4):
        self.featureer = ControlFeatureExtraction()
        self.abstractor = ClusterMessageAbstractor(KMeansClustering(n_clusters=n_clusters, random_state=42))
        self.inferer = PTAInfer()
        self.merger = KTailStateMerger(k)
        
        # 缓存sessions，供其他层使用
        self._sessions: Dict[SessionKey, List] = None
        # 缓存每会话的(events, features)
        self._sess_features: Dict[SessionKey, tuple] = None

    def run_from_pcap(self, pcap_path: str) -> FSM:
        trace = PCAPPipeline().run(pcap_path)
        return self.run(trace)

    def run(self, trace: Trace) -> FSM:
        # group events by session
        sessions = defaultdict(list)
        for ev in trace.events:
            sessions[ev.session_key].append(ev)     # 将事件按照session_key分桶  session_key -> [事件]
        
        # 为每个会话排序事件
        for sk, events in sessions.items():
            events.sort(key=lambda e: e.timestamp)
        
        # 缓存sessions供其他层使用
        self._sessions = dict(sessions)

        # sort and extract features per event and collect all features
        all_features = []
        sess_features = {}
        for sk, events in sessions.items():

            features = self.featureer.extract(events)       # 提取特征
            sess_features[sk] = (events, features)
            all_features.extend(features)

        # 缓存(sess_features)供数据流层复用
        self._sess_features = sess_features

        # 训练聚类模型
        if len(all_features) == 0:
            raise RuntimeError("no events found")
        self.abstractor.fit(all_features)

        # build sequences
        sequences = {}
        for sk, (events, features) in sess_features.items():
            symbols = [self.abstractor.abstract(f) for f in features]   # 生成符号序列
            sequences[sk] = symbols

        # infer FSM
        fsm = self.inferer.infer(sequences)

        # merge states
        fsm = self.merger.merge(fsm)
        # print(fsm)
        
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
