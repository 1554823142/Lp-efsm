from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import os
import random

from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.event import MessageEvent, Direction
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.control_flow_layer.inference.pta_infer import PTAInfer
from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.evaluation.metrics import ClusteringEvaluator, FSMEvaluator, EFSMevaluator


class ProtocolLabeler:
    name: str

    def label(self, ev: MessageEvent) -> str:
        raise NotImplementedError()


def _dir_tag(d: Direction) -> str:
    return "c2s" if d == Direction.C2S else "s2c"


class ModbusTCPLabeler(ProtocolLabeler):
    name = "MODBUS"

    def label(self, ev: MessageEvent) -> str:
        payload = ev.payload or b""
        if len(payload) < 8:
            return f"unknown_{_dir_tag(ev.direction)}"
        proto_id = int.from_bytes(payload[2:4], byteorder="big", signed=False)
        if proto_id != 0:
            return f"unknown_{_dir_tag(ev.direction)}"
        fc = payload[7]
        return f"fc_{fc:02x}_{_dir_tag(ev.direction)}"


class IEC104Labeler(ProtocolLabeler):
    name = "IEC60870-104"

    def label(self, ev: MessageEvent) -> str:
        payload = ev.payload or b""
        if len(payload) < 7 or payload[0] != 0x68:
            return f"unknown_{_dir_tag(ev.direction)}"
        tid = payload[6]
        return f"type_{tid:02x}_{_dir_tag(ev.direction)}"


class DNP3Labeler(ProtocolLabeler):
    name = "DNP3"

    def label(self, ev: MessageEvent) -> str:
        payload = ev.payload or b""
        if len(payload) < 4 or payload[0] != 0x05 or payload[1] != 0x64:
            return f"unknown_{_dir_tag(ev.direction)}"
        ctrl = payload[3]
        func = ctrl & 0x0F
        prm = 1 if (ctrl & 0x40) else 0
        return f"func_{func:01x}_prm{prm}_{_dir_tag(ev.direction)}"


def _looks_like_capture(path: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 16:
            return False
        with open(path, "rb") as f:
            head = f.read(4)
        if len(head) != 4:
            return False
        magic = int.from_bytes(head, byteorder="little", signed=False)
        if magic in (0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1):
            return True
        if head == b"\x0a\x0d\x0d\x0a":
            return True
        return False
    except OSError:
        return False


@dataclass
class SupervisedEvalResult:
    protocol: str
    pcap_paths: List[str]
    train_sessions: int
    test_sessions: int
    clustering: Dict[str, float]
    fsm: Dict[str, float]
    efsm: Dict[str, float]


def _group_sessions(trace: Trace) -> Dict[SessionKey, List[MessageEvent]]:
    sessions: Dict[SessionKey, List[MessageEvent]] = {}
    for ev in trace.events:
        sessions.setdefault(ev.session_key, []).append(ev)
    for evs in sessions.values():
        evs.sort(key=lambda e: e.timestamp)
    return sessions


def _split_keys(keys: List[SessionKey], test_ratio: float, seed: int) -> Tuple[List[SessionKey], List[SessionKey]]:
    rng = random.Random(seed)
    keys = list(keys)
    rng.shuffle(keys)
    n_test = max(1, int(len(keys) * test_ratio)) if keys else 0
    test = keys[:n_test]
    train = keys[n_test:]
    if not train and test:
        train, test = test, []
    return train, test


def _encode_strs(xs: List[str]) -> List[int]:
    table: Dict[str, int] = {}
    out: List[int] = []
    for s in xs:
        if s not in table:
            table[s] = len(table)
        out.append(table[s])
    return out


def _majority_map(pairs: List[Tuple[str, str]]) -> Dict[str, str]:
    counts: Dict[str, Dict[str, int]] = {}
    for sym, lab in pairs:
        counts.setdefault(sym, {})
        counts[sym][lab] = counts[sym].get(lab, 0) + 1
    mapping: Dict[str, str] = {}
    for sym, c in counts.items():
        mapping[sym] = max(c.items(), key=lambda kv: kv[1])[0]
    return mapping


def _build_label_sequences(
    sessions: Dict[SessionKey, List[MessageEvent]],
    labeler: ProtocolLabeler,
) -> Dict[SessionKey, List[str]]:
    return {sk: [labeler.label(ev) for ev in evs] for sk, evs in sessions.items()}


def _edges_from_sequences(seqs: Dict[SessionKey, List[str]]) -> Tuple[set, set]:
    states = set(["START"])
    edges = set()
    for seq in seqs.values():
        if not seq:
            continue
        states.add(seq[0])
        edges.add(("START", seq[0], seq[0]))
        for a, b in zip(seq, seq[1:]):
            states.add(a)
            states.add(b)
            edges.add((a, b, b))
    return states, edges


def _build_symbol_sequences(
    cf: ControlFlowPipeline,
    sessions: Dict[SessionKey, List[MessageEvent]],
) -> Tuple[Dict[SessionKey, List[str]], List[Tuple[str, str]]]:
    seqs: Dict[SessionKey, List[str]] = {}
    flat_pairs: List[Tuple[str, str]] = []
    for sk, evs in sessions.items():
        feats = cf.featureer.extract(evs)
        syms = [cf.abstractor.abstract(f) for f in feats]
        seqs[sk] = syms
        for ev, sym in zip(evs, syms):
            flat_pairs.append((sk, sym))
    return seqs, flat_pairs


def _build_symbol_var_sequences(
    cf: ControlFlowPipeline,
    df_feature_processor: FeatureProcessor,
    sessions: Dict[SessionKey, List[MessageEvent]],
) -> Tuple[Dict[SessionKey, List[Tuple[str, Dict[str, float]]]], List[str], List[str]]:
    sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]] = {}
    flat_syms: List[str] = []
    flat_labs: List[str] = []
    for sk, evs in sessions.items():
        feats = cf.featureer.extract(evs)
        pairs = []
        for ev, feat in zip(evs, feats):
            sym = cf.abstractor.abstract(feat)
            vars_dict = df_feature_processor.extract_vars(ev)
            pairs.append((sym, vars_dict))
            flat_syms.append(sym)
        sequences[sk] = pairs
    return sequences, flat_syms, flat_labs


def _mutate_negative(
    sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]],
    seed: int,
) -> Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]:
    rng = random.Random(seed)
    neg: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]] = {}
    for sk, pairs in sequences.items():
        out = []
        for sym, vars_dict in pairs:
            v = dict(vars_dict)
            keys = [k for k in v.keys() if k.startswith("b")]
            if keys:
                k = rng.choice(keys)
                v[k] = float((int(v[k]) + rng.randint(1, 7)) % 256)
            else:
                if "len" in v:
                    v["len"] = float(max(0.0, v["len"] + rng.randint(1, 10)))
            out.append((sym, v))
        neg[sk] = out
    return neg


def evaluate_protocol_pcaps(
    labeler: ProtocolLabeler,
    pcap_paths: Sequence[str],
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
) -> SupervisedEvalResult:
    pipeline = PCAPPipeline()
    events: List[MessageEvent] = []
    for p in pcap_paths:
        if not _looks_like_capture(p):
            continue
        try:
            t = pipeline.run(p)
        except Exception:
            continue
        events.extend(t.events)
    if not events:
        raise RuntimeError(f"no parsable capture files for {labeler.name}")
    events.sort(key=lambda e: e.timestamp)
    trace = Trace(events=events)

    sessions_all = _group_sessions(trace)
    keys = list(sessions_all.keys())
    if len(keys) > max_sessions:
        rng = random.Random(seed)
        rng.shuffle(keys)
        keys = keys[:max_sessions]
        sessions_all = {k: sessions_all[k] for k in keys}

    if len(sessions_all) < 2:
        only_key = next(iter(sessions_all.keys()))
        evs = sessions_all[only_key]
        split_at = max(1, int(len(evs) * (1.0 - test_ratio)))
        split_at = min(split_at, len(evs) - 1) if len(evs) > 1 else split_at
        train_evs = evs[:split_at]
        test_evs = evs[split_at:]
        test_key = SessionKey(
            ip1=only_key.ip1,
            port1=only_key.port1,
            ip2=f"{only_key.ip2}#test",
            port2=only_key.port2,
            protocol=only_key.protocol,
        )
        test_evs = [
            MessageEvent(session_key=test_key, timestamp=e.timestamp, payload=e.payload, direction=e.direction)
            for e in test_evs
        ]
        train_sessions = {only_key: train_evs}
        test_sessions = {test_key: test_evs} if test_evs else {}
        train_keys = [only_key]
        test_keys = [test_key] if test_evs else []
    else:
        train_keys, test_keys = _split_keys(list(sessions_all.keys()), test_ratio=test_ratio, seed=seed)
        train_sessions = {k: sessions_all[k] for k in train_keys}
        test_sessions = {k: sessions_all[k] for k in test_keys}

    train_trace = Trace(events=[ev for k in train_keys for ev in train_sessions[k]])
    cf = ControlFlowPipeline(use_apriori=True)
    fsm = cf.run(train_trace)

    df = DataFlowPipeline(abstractor=cf.abstractor, symbol_featureer=cf.featureer)
    efsm = df.run(
        trace=train_trace,
        fsm=fsm,
        sessions=train_sessions,
        precomputed_sess_features=cf.get_sess_features(),
        apriori_positions=cf.get_apriori_positions(),
        apriori_static_items=cf.get_apriori_static_items(),
    )

    fp = FeatureProcessor(
        apriori_positions=cf.get_apriori_positions(),
        apriori_static_items=cf.get_apriori_static_items(),
    )
    test_label_seqs = _build_label_sequences(test_sessions, labeler)
    test_pairs, test_syms, _ = _build_symbol_var_sequences(cf, fp, test_sessions)
    test_true_labels_flat = [labeler.label(ev) for k in test_keys for ev in test_sessions[k]]

    y_true = _encode_strs(test_true_labels_flat)
    y_pred = _encode_strs(test_syms)
    clustering_res = ClusteringEvaluator(supervised=True, unsupervised=False).evaluate(
        labels_true=y_true, labels_pred=y_pred
    )

    train_syms: List[str] = []
    train_labs: List[str] = []
    for k in train_keys:
        evs = train_sessions[k]
        feats = cf.featureer.extract(evs)
        for ev, feat in zip(evs, feats):
            train_syms.append(cf.abstractor.abstract(feat))
            train_labs.append(labeler.label(ev))
    sym2lab = _majority_map(list(zip(train_syms, train_labs)))

    pred_label_seqs: Dict[SessionKey, List[str]] = {}
    for sk, pairs in test_pairs.items():
        pred_label_seqs[sk] = [sym2lab.get(sym, "UNK") for sym, _ in pairs]

    gt_states, gt_edges = _edges_from_sequences(test_label_seqs)
    pred_states, pred_edges = _edges_from_sequences(pred_label_seqs)

    pred_fsm = PTAInfer().infer(pred_label_seqs).determinize()
    fsm_res = FSMEvaluator().evaluate(
        gt_states=gt_states,
        gt_edges=gt_edges,
        pred_states=pred_states,
        pred_edges=pred_edges,
        fsm=pred_fsm,
        sequences=test_label_seqs,
    )

    neg_pairs = _mutate_negative(test_pairs, seed=seed + 1)
    numeric_vars = {k for k in efsm.variable_defs if k.startswith("b") or k == "len"}
    efsm_res = EFSMevaluator().evaluate(
        efsm=efsm,
        sequences=test_pairs,
        negative_sequences=neg_pairs,
        numeric_vars=numeric_vars,
    )

    return SupervisedEvalResult(
        protocol=labeler.name,
        pcap_paths=list(pcap_paths),
        train_sessions=len(train_sessions),
        test_sessions=len(test_sessions),
        clustering=clustering_res,
        fsm=fsm_res,
        efsm=efsm_res,
    )


def pick_industrial_protocols(data_root: str) -> List[Tuple[ProtocolLabeler, List[str]]]:
    candidates: List[Tuple[ProtocolLabeler, List[str]]] = []

    modbus = os.path.join(data_root, "MODBUS")
    if os.path.isdir(modbus):
        paths = [
            os.path.join(modbus, fn)
            for fn in sorted(os.listdir(modbus))
            if fn.lower().endswith(".pcap") and _looks_like_capture(os.path.join(modbus, fn))
        ]
        paths = paths[:6]
        if paths:
            candidates.append((ModbusTCPLabeler(), paths))

    iec104 = os.path.join(data_root, "IEC60870-104")
    if os.path.isdir(iec104):
        paths = [
            os.path.join(iec104, fn)
            for fn in sorted(os.listdir(iec104))
            if fn.lower().endswith(".pcap") and _looks_like_capture(os.path.join(iec104, fn))
        ]
        paths = paths[:6]
        if paths:
            candidates.append((IEC104Labeler(), paths))

    dnp3 = os.path.join(data_root, "DNP3")
    if os.path.isdir(dnp3):
        paths = [
            os.path.join(dnp3, fn)
            for fn in sorted(os.listdir(dnp3))
            if fn.lower().endswith(".pcap") and _looks_like_capture(os.path.join(dnp3, fn))
        ]
        paths = paths[:6]
        if paths:
            candidates.append((DNP3Labeler(), paths))

    return candidates


def run_supervised_benchmark(
    data_root: str,
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
) -> List[SupervisedEvalResult]:
    results: List[SupervisedEvalResult] = []
    for labeler, pcaps in pick_industrial_protocols(data_root):
        results.append(
            evaluate_protocol_pcaps(
                labeler=labeler,
                pcap_paths=pcaps,
                seed=seed,
                test_ratio=test_ratio,
                max_sessions=max_sessions,
            )
        )
    return results
