from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline


class ProtocolInferPipeline:
    def __init__(self):
        self.pcap_pipeline = PCAPPipeline()
        self.control_flow_pipeline = ControlFlowPipeline()
        self.data_flow_pipeline = DataFlowPipeline()

    def run(self, pcap_path: str):
        trace = self.pcap_pipeline.run(pcap_path)
        return self.control_flow_pipeline.run(trace)