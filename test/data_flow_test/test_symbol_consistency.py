import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root / "protocol_infer"))
sys.path.insert(0, str(project_root))

from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.event import MessageEvent, Direction
from protocol_infer.core.datamodel.trace import Trace

from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline

def build_synthetic_trace():
    sk1 = SessionKey("1.1.1.1", 1000, "2.2.2.2", 2000, "TCP")
    sk2 = SessionKey("3.3.3.3", 1001, "4.4.4.4", 2001, "TCP")
    events = []
    base_ts = 0.0
    payloads = [b"\x01\x02\x03", b"\x10\x20", b"\x05\x06\x07\x08", b"\xAA"]
    dirs = [Direction.C2S, Direction.S2C, Direction.C2S, Direction.S2C]
    for i, (p, d) in enumerate(zip(payloads, dirs)):
        events.append(MessageEvent(session_key=sk1, timestamp=base_ts + i * 0.1, payload=p, direction=d))
    for i, (p, d) in enumerate(zip(payloads, dirs)):
        events.append(MessageEvent(session_key=sk2, timestamp=base_ts + i * 0.1, payload=p, direction=d))
    events.sort(key=lambda e: e.timestamp)
    return Trace(events=events)

def compute_symbols_from_sess_features(sess_features, abstractor):
    symbols_per_session = {}
    for sk, (events, features) in sess_features.items():
        syms = [abstractor.abstract(f) for f in features]
        symbols_per_session[sk] = syms
    return symbols_per_session

def test_symbol_consistency_with_precomputed_features():
    trace = build_synthetic_trace()
    cf = ControlFlowPipeline(n_clusters=2, k=2)
    fsm = cf.run(trace)
    sessions = cf.get_sessions()
    sess_features = cf.get_sess_features()
    df = DataFlowPipeline(abstractor=cf.abstractor)
    efsm = df.run(trace, fsm, sessions=sessions, precomputed_sess_features=sess_features)
    symbols_ref = compute_symbols_from_sess_features(sess_features, cf.abstractor)
    symbols_actual = {}
    for sk in sessions:
        syms = [am.symbol for am in trace.abstract_messages if am.session_key == sk]
        symbols_actual[sk] = syms
    assert symbols_actual == symbols_ref
    num_trans = len(efsm.transitions)
    num_guard = sum(1 for t in efsm.transitions if t.guard is not None)
    num_action = sum(1 for t in efsm.transitions if t.action is not None)
    assert num_trans >= 1
    assert num_guard >= 1
    assert num_action >= 1

def test_symbol_consistency_with_feature_extractor():
    trace = build_synthetic_trace()
    cf = ControlFlowPipeline(n_clusters=2, k=2)
    fsm = cf.run(trace)
    sessions = cf.get_sessions()
    df = DataFlowPipeline(abstractor=cf.abstractor, symbol_featureer=cf.featureer)
    efsm = df.run(trace, fsm, sessions=sessions)
    # 重新用控制流层特征提取器计算特征并对比
    sess_features_local = {}
    for sk, evs in sessions.items():
        feats = cf.featureer.extract(evs)
        sess_features_local[sk] = (evs, feats)
    symbols_ref = compute_symbols_from_sess_features(sess_features_local, cf.abstractor)
    symbols_actual = {}
    for sk in sessions:
        syms = [am.symbol for am in trace.abstract_messages if am.session_key == sk]
        symbols_actual[sk] = syms
    assert symbols_actual == symbols_ref
    num_trans = len(efsm.transitions)
    num_guard = sum(1 for t in efsm.transitions if t.guard is not None)
    num_action = sum(1 for t in efsm.transitions if t.action is not None)
    assert num_trans >= 1
    assert num_guard >= 1
    assert num_action >= 1
