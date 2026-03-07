from protocol_infer.core.datamodel.event import MessageEvent, Direction
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.control_flow_layer.features.apriori_feature_extraction import AprioriFeatureExtraction

# 模拟 Modbus TCP 流量
sk = SessionKey('1.1.1.1', 1000, '2.2.2.2', 502, 'TCP')
# 读线圈请求 (Func 0x01)
payload1 = b'\x00\x01\x00\x00\x00\x06\x01\x01\x00\x00\x00\x08'
# 写寄存器请求 (Func 0x06)
payload2 = b'\x00\x02\x00\x00\x00\x06\x01\x06\x00\x01\x00\x01'

events = []
for i in range(10):
    events.append(MessageEvent(sk, 0.1*i, payload1 if i%2==0 else payload2, Direction.C2S))

trace = Trace(events=events)

# 测试 DBSCAN + Apriori
print("Starting pipeline...")
pipeline = ControlFlowPipeline(algorithm="dbscan", use_apriori=True, eps=0.5, min_samples=2)
fsm = pipeline.run(trace)

# 检查特征提取器结果
extractor = pipeline.featureer
print(f"Positions discovered: {extractor.positions}")
print(f"Itemsets discovered: {len(extractor.itemsets)}")

features = extractor.extract(events)
print(f"First 2 features:")
print(features[0])
print(features[1])

labels = pipeline.clusterer.predict(features)
print(f"Labels: {labels}")

print(f"FSM States: {len(fsm.states)}")
print(f"FSM Transitions: {len(fsm.transitions)}")
for t in fsm.transitions:
    print(f"Transition: {t.src} --({t.symbol})--> {t.dst}")
