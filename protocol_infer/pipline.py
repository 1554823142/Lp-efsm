from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.probabilistic_layer.pipeline import ProbabilisticPipeline


class ProtocolInferPipeline:
    def __init__(self):
        self.pcap_pipeline = PCAPPipeline()
        self.control_flow_pipeline = ControlFlowPipeline()
        self.data_flow_pipeline = DataFlowPipeline()
        self.probabilistic_pipeline = ProbabilisticPipeline()

    def run(self, pcap_path: str):
        # 1. Pcap -> Trace
        trace = self.pcap_pipeline.run(pcap_path)

        # 2. Trace -> FSM (Control Flow)
        fsm = self.control_flow_pipeline.run(trace)

        # 3. FSM -> EFSM (Data Flow)
        # 将 Apriori 发现的偏移位置、复用的抽象器和特征提取器都传递给 DataFlowPipeline
        self.data_flow_pipeline.abstraction_processor.abstractor = self.control_flow_pipeline.abstractor
        self.data_flow_pipeline.symbol_featureer = self.control_flow_pipeline.featureer

        efsm = self.data_flow_pipeline.run(
            trace=trace,
            fsm=fsm,
            sessions=self.control_flow_pipeline.get_sessions(),
            precomputed_sess_features=self.control_flow_pipeline.get_sess_features(),
            apriori_positions=self.control_flow_pipeline.get_apriori_positions(),
            apriori_static_items=self.control_flow_pipeline.get_apriori_static_items(),
        )
        return efsm

    def run_pefsm(self, pcap_path: str):
        trace = self.pcap_pipeline.run(pcap_path)
        fsm = self.control_flow_pipeline.run(trace)

        self.data_flow_pipeline.abstraction_processor.abstractor = self.control_flow_pipeline.abstractor
        self.data_flow_pipeline.symbol_featureer = self.control_flow_pipeline.featureer

        efsm = self.data_flow_pipeline.run(
            trace=trace,
            fsm=fsm,
            sessions=self.control_flow_pipeline.get_sessions(),
            precomputed_sess_features=self.control_flow_pipeline.get_sess_features(),
            apriori_positions=self.control_flow_pipeline.get_apriori_positions(),
            apriori_static_items=self.control_flow_pipeline.get_apriori_static_items(),
        )
        return self.probabilistic_pipeline.run(efsm=efsm, trace=trace)
