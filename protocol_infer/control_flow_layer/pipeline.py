from collections import defaultdict
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.features.control_feature_extraction import ControlFeatureExtraction
from protocol_infer.control_flow_layer.abstraction.clustering_abstraction import ClusterMessageAbstractor
from protocol_infer.algorithm.clustering.kmeans import KMeansClustering
from protocol_infer.control_flow_layer.inference.pta_infer import PTAInfer
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.model.fsm import FSM

class ControlFlowPipeline:
    def __init__(self, n_clusters: int = 8):
        self.featureer = ControlFeatureExtraction()
        self.abstractor = ClusterMessageAbstractor(KMeansClustering(n_clusters=n_clusters))
        self.inferer = PTAInfer()

    def run_from_pcap(self, pcap_path: str) -> FSM:
        trace = PCAPPipeline().run(pcap_path)
        return self.run(trace)

    def run(self, trace: Trace) -> FSM:
        # group events by session
        sessions = defaultdict(list)
        for ev in trace.events:
            sessions[ev.session_key].append(ev)     # 将事件按照session_key分桶  session_key -> [事件]

        # sort and extract features per event and collect all features
        all_features = []
        sess_features = {}
        for sk, events in sessions.items():

            features = self.featureer.extract(events)       # 提取特征
            sess_features[sk] = (events, features)
            all_features.extend(features)

        # 训练聚类模型
        if len(all_features) == 0:
            raise RuntimeError("no events found")
        self.abstractor.fit(all_features)

        # build sequences
        sequences = {}
        for sk, (events, features) in sess_features.items():
            symbols = [self.abstractor.abstract(f) for f in features]
            print(symbols)
            sequences[sk] = symbols

        # infer FSM
        fsm = self.inferer.infer(sequences)
        return fsm
