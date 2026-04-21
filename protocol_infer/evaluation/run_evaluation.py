"""
入口脚本：把三个模块串起来，一条命令完成完整评估。

流程
----
1. 从 PCAP 文件加载事件（复用 PCAPPipeline）
2. 构建 ControlFlow + DataFlow，推断 EFSM
3. 用 FieldExtractor（模块一）对测试会话做字段级提取，生成 SessionTrace
4. 从 GT 规范库（模块二）获取对应协议的 ProtocolGT
5. 用 EnhancedEFSMEvaluator（模块三）计算四个维度指标
6. 同时运行原有 EFSMevaluator 以保留历史指标
7. 端到端 Trace 重放：在测试集（无则训练集）上用 PEFSM + ReplayBuilder 统计会话级/步级重放成功率（与 Web 一致）

命令行用法
----------
python -m protocol_infer.evaluation.run_evaluation \
    --protocol MODBUS \
    --data-dir /path/to/Data/MODBUS \
    [--max-pcaps 6] [--max-sessions 200] [--test-ratio 0.2] [--seed 42] \
    [--output results.json]

Python API 用法
---------------
from protocol_infer.evaluation.run_evaluation import run_full_evaluation
results = run_full_evaluation(protocol="MODBUS", data_dir="/path/to/Data/MODBUS")
print(results)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from protocol_infer.core.datamodel.event import MessageEvent, Direction
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.data_flow_layer.feature.data_feature_extraction import FeatureProcessor
from protocol_infer.evaluation.supervised_eval import (
    _looks_like_capture,
    _group_sessions,
    _split_long_sessions,
    _split_keys,
    _mutate_negative,
    _build_symbol_var_sequences,
    _normalize_protocol_name,
    ModbusTCPLabeler,
    IEC104Labeler,
    DNP3Labeler,
    EtherNetIPLabeler,
    MQTTLabeler,
    S7CommLabeler,
    ProtocolLabeler,
    filter_events_for_protocol,
)
from protocol_infer.evaluation.metrics import EFSMevaluator
from protocol_infer.evaluation.field_extractor import FieldExtractor, SessionTrace
from protocol_infer.evaluation.gt_guard_action import get_gt, ProtocolGT
from protocol_infer.evaluation.efsm_evaluator import EnhancedEFSMEvaluator, EFSMEvalResult


# ---------------------------------------------------------------------------
# Labeler 注册表
# ---------------------------------------------------------------------------

_LABELERS: Dict[str, ProtocolLabeler] = {
    "MODBUS": ModbusTCPLabeler(),
    "MODBUSTCP": ModbusTCPLabeler(),
    "IEC104": IEC104Labeler(),
    "IEC60870-104": IEC104Labeler(),
    "DNP3": DNP3Labeler(),
    "ETHERNET_IP": EtherNetIPLabeler(),
    "ETHERNETIP": EtherNetIPLabeler(),
    "S7COMM": S7CommLabeler(),
    "S7": S7CommLabeler(),
    "MQTT": MQTTLabeler(),
}

_PROTOCOL_DIRS: Dict[str, str] = {
    "MODBUS": "MODBUS",
    "MODBUSTCP": "MODBUS",
    "IEC104": "IEC60870-104",
    "IEC60870-104": "IEC60870-104",
    "DNP3": "DNP3",
    "ETHERNET_IP": "Ethernet_IP",
    "ETHERNETIP": "Ethernet_IP",
    "MQTT": "MQTT",
}


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class FullEvalResult:
    protocol: str
    train_sessions: int
    test_sessions: int
    # 划分 train/test 前（且 max_sessions 截断后）的 TCP 会话条数
    total_sessions: int
    # 原有历史指标（与 EFSMevaluator 兼容）
    legacy_efsm: Dict[str, float]
    # 新四维指标
    enhanced_efsm: Dict[str, float]
    # 端到端重放：PEFSM + ReplayBuilder，与 Web 一致；replay_on 为 test / train / none / skipped
    replay_on: str = "none"
    replay_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "train_sessions": self.train_sessions,
            "test_sessions": self.test_sessions,
            "total_sessions": self.total_sessions,
            "legacy_efsm": self.legacy_efsm,
            "enhanced_efsm": self.enhanced_efsm,
            "replay_on": self.replay_on,
            "replay_metrics": dict(self.replay_metrics),
        }


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

_MAX_SINGLE_PCAP_BYTES = 12 * 1024 * 1024  # 单文件限制 12MB，避免巨型文件卡死训练

def _collect_pcaps(data_dir: str, max_pcaps: int) -> List[str]:
    """从目录中收集有效的 PCAP 文件路径。"""
    paths: List[str] = []
    if not os.path.isdir(data_dir):
        return paths
    candidates: List[Tuple[int, str, str]] = []
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".pcap"):
            continue
        full = os.path.join(data_dir, fn)
        if not _looks_like_capture(full):
            continue
        try:
            size = os.path.getsize(full)
            if size > _MAX_SINGLE_PCAP_BYTES:
                continue
        except OSError:
            continue
        candidates.append((size, fn, full))

    candidates.sort(key=lambda item: (-item[0], item[1].lower()))
    for _size, _name, full in candidates:
        paths.append(full)
        if len(paths) >= max_pcaps:
            break
    return paths


def run_full_evaluation_from_sessions(
    protocol: str,
    sessions_all: Dict[SessionKey, List[MessageEvent]],
    labeler: ProtocolLabeler,
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
) -> FullEvalResult:
    proto_upper = protocol.upper()

    desired_min_sessions = int(max_sessions)
    if len(sessions_all) < desired_min_sessions:
        sessions_all = _split_long_sessions(sessions_all, target_sessions=desired_min_sessions)

    keys = list(sessions_all.keys())
    if len(keys) > max_sessions:
        rng = random.Random(seed)
        rng.shuffle(keys)
        keys = keys[:max_sessions]
        sessions_all = {k: sessions_all[k] for k in keys}

    total_sessions = len(sessions_all)

    if len(sessions_all) < 2:
        only_key = next(iter(sessions_all.keys()))
        evs = sessions_all[only_key]
        split_at = max(1, min(int(len(evs) * (1.0 - test_ratio)), len(evs) - 1))
        train_evs, test_evs_raw = evs[:split_at], evs[split_at:]
        test_key = SessionKey(
            ip1=only_key.ip1,
            port1=only_key.port1,
            ip2=only_key.ip2,
            port2=only_key.port2,
            protocol=only_key.protocol,
            segment_id=int(getattr(only_key, "segment_id", 0)) + 1,
        )
        test_evs = [
            MessageEvent(session_key=test_key, timestamp=e.timestamp, payload=e.payload, direction=e.direction)
            for e in test_evs_raw
        ]
        train_sessions = {only_key: train_evs}
        test_sessions = {test_key: test_evs} if test_evs else {}
        train_keys = [only_key]
    else:
        train_keys, test_keys = _split_keys(list(sessions_all.keys()), test_ratio, seed)
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
    test_pairs, _test_syms, _ = _build_symbol_var_sequences(cf, fp, test_sessions)
    neg_pairs = _mutate_negative(test_pairs, seed=seed + 1)
    numeric_vars = {k for k in efsm.variable_defs if k.startswith("b") or k == "len"}
    legacy_res = EFSMevaluator().evaluate(
        efsm=efsm,
        sequences=test_pairs,
        negative_sequences=neg_pairs,
        numeric_vars=numeric_vars,
    )

    field_extractor = FieldExtractor(
        protocol=proto_upper,
        labeler=labeler.label,
    )
    session_traces: Dict[SessionKey, SessionTrace] = field_extractor.extract_sessions(test_sessions)

    for sk, st in session_traces.items():
        evs = test_sessions.get(sk, [])
        if not evs:
            continue
        try:
            feats = cf.featureer.extract(evs)
            for rec, (ev, feat) in zip(st.records, zip(evs, feats)):
                rec.abstract_sym = cf.abstractor.abstract(feat)
                rec.abstract_vars = fp.extract_vars(ev)
        except Exception:
            pass

    gt: Optional[ProtocolGT] = get_gt(proto_upper)
    if gt is None:
        print(
            f"[WARN] No GT spec for protocol '{protocol}'. Enhanced metrics will be skipped.",
            file=sys.stderr,
        )
        enhanced_res: Dict[str, float] = {}
    else:
        evaluator = EnhancedEFSMEvaluator()
        eval_result: EFSMEvalResult = evaluator.evaluate(
            efsm=efsm,
            gt=gt,
            session_traces=session_traces,
        )
        enhanced_res = eval_result.to_dict()

    replay_on = "none"
    replay_metrics: Dict[str, Any] = {}
    try:
        from protocol_infer.probabilistic_layer.pipeline import ProbabilisticPipeline
        from protocol_infer.visualization.replay import ReplayBuilder, summarize_replay_by_session

        if train_trace.abstract_messages:
            pefsm = ProbabilisticPipeline().run(efsm=efsm, trace=train_trace)
            if test_sessions:
                test_events: List[MessageEvent] = []
                for _sk, evs in test_sessions.items():
                    test_events.extend(evs)
                test_events.sort(key=lambda e: e.timestamp)
                replay_trace = Trace(events=test_events)
                df.materialize_abstract_messages(
                    replay_trace,
                    sessions=test_sessions,
                    precomputed_sess_features=None,
                    apriori_positions=cf.get_apriori_positions(),
                    skip_dynamic_detection=True,
                )
                replay_on = "test"
            else:
                replay_trace = train_trace
                replay_on = "train"
            steps = ReplayBuilder().build(pefsm, replay_trace)
            replay_metrics = summarize_replay_by_session(steps)
        else:
            replay_on = "skipped_no_abstract"
    except Exception as exc:
        print(f"[WARN] Trace replay metrics failed: {exc}", file=sys.stderr)
        replay_on = "error"
        replay_metrics = {}

    return FullEvalResult(
        protocol=proto_upper,
        train_sessions=len(train_sessions),
        test_sessions=len(test_sessions),
        total_sessions=total_sessions,
        legacy_efsm=legacy_res,
        enhanced_efsm=enhanced_res,
        replay_on=replay_on,
        replay_metrics=replay_metrics,
    )


def _encode_mqtt_remaining_length(n: int) -> bytes:
    out = bytearray()
    x = int(n)
    while True:
        digit = x % 128
        x //= 128
        if x > 0:
            digit |= 0x80
        out.append(digit)
        if x == 0:
            break
    return bytes(out)


def generate_synthetic_sessions(
    protocol: str,
    n_sessions: int,
    session_len: int,
    seed: int,
) -> Dict[SessionKey, List[MessageEvent]]:
    proto = _normalize_protocol_name(protocol)
    rng = random.Random(seed)
    sessions: Dict[SessionKey, List[MessageEvent]] = {}

    def rand_bytes(k: int) -> bytes:
        try:
            return rng.randbytes(k)
        except AttributeError:
            return bytes(rng.getrandbits(8) for _ in range(k))

    def mk_key(i: int) -> SessionKey:
        ip1 = f"10.0.{(i // 254) % 254 + 1}.{i % 254 + 1}"
        ip2 = f"10.1.{(i // 254) % 254 + 1}.{(i * 7) % 254 + 1}"
        if proto == "MODBUS":
            return SessionKey(ip1=ip1, port1=502, ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)
        if proto == "DNP3":
            return SessionKey(ip1=ip1, port1=20000, ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)
        if proto == "IEC60870-104":
            return SessionKey(ip1=ip1, port1=2404, ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)
        if proto == "ETHERNET_IP":
            return SessionKey(ip1=ip1, port1=44818, ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)
        if proto == "MQTT":
            return SessionKey(ip1=ip1, port1=1883, ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)
        return SessionKey(ip1=ip1, port1=10000 + (i % 20000), ip2=ip2, port2=20000 + (i % 20000), protocol="TCP", segment_id=i)

    def mk_event(sk: SessionKey, ts: float, payload: bytes, direction: Direction) -> MessageEvent:
        return MessageEvent(session_key=sk, timestamp=ts, payload=payload, direction=direction)

    known_enip_cmds = [0x0004, 0x0063, 0x0064, 0x0065, 0x0066, 0x006F, 0x0070, 0x0072]

    for i in range(max(0, int(n_sessions))):
        sk = mk_key(i)
        ts = float(i) * 10.0
        evs: List[MessageEvent] = []
        steps = max(2, int(session_len))
        for j in range(steps):
            direction = Direction.C2S if (j % 2 == 0) else Direction.S2C

            if proto == "MODBUS":
                tid = (i * 1000 + j) & 0xFFFF
                unit = 1
                fc = rng.choice([1, 2, 3, 4, 5, 6, 15, 16])
                pdu = bytes([unit, fc]) + rand_bytes(4)
                mbap = tid.to_bytes(2, "big") + b"\x00\x00" + len(pdu).to_bytes(2, "big")
                payload = mbap + pdu
            elif proto == "DNP3":
                seq = (i * 997 + j) & 0x0F
                ctrl = (0xC0 if direction == Direction.C2S else 0x80) | seq
                src = (100 + (i % 400)) & 0xFFFF
                dst = (200 + ((i * 3) % 400)) & 0xFFFF
                if direction == Direction.S2C:
                    src, dst = dst, src
                length = 0x05
                header = b"\x05\x64" + bytes([length]) + bytes([ctrl]) + dst.to_bytes(2, "little") + src.to_bytes(2, "little")
                payload = header + rand_bytes(4)
            elif proto == "IEC60870-104":
                typ = rng.choice([1, 3, 9, 13, 36, 45, 46, 47, 48, 100])
                payload = bytes([0x68, 0x04, 0x00, 0x00, 0x00, 0x00, typ & 0xFF]) + rand_bytes(2)
            elif proto == "ETHERNET_IP":
                command = rng.choice(known_enip_cmds)
                payload = command.to_bytes(2, "little") + (0).to_bytes(2, "little") + rand_bytes(20)
            elif proto == "MQTT":
                msg_type = rng.randint(1, 14)
                payload = bytes([(msg_type << 4) & 0xF0]) + _encode_mqtt_remaining_length(0)
            else:
                payload = rand_bytes(16)

            ts += 0.05 + rng.random() * 0.01
            evs.append(mk_event(sk, ts, payload, direction))
        sessions[sk] = evs

    return sessions


def run_full_evaluation(
    protocol: str,
    data_dir: str,
    labeler: Optional[ProtocolLabeler] = None,
    pcap_paths: Optional[List[str]] = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
    max_pcaps: int = 6,
) -> FullEvalResult:
    """
    完整评估流程：PCAP -> 推断 EFSM -> 字段提取 -> GT 对比 -> 指标输出。

    参数
    ----
    protocol:     协议名（MODBUS / IEC104 / DNP3 / ...）
    data_dir:     PCAP 文件目录（与 protocol 对应）
    labeler:      可选，覆盖默认 labeler
    pcap_paths:   可选，直接指定 PCAP 路径列表（忽略 data_dir）
    seed, test_ratio, max_sessions, max_pcaps: 同 evaluate_protocol_pcaps
    """
    proto_upper = protocol.upper()

    # ---- 1. 确定 labeler ----
    if labeler is None:
        labeler = _LABELERS.get(proto_upper)
        if labeler is None:
            raise ValueError(
                f"Unknown protocol '{protocol}'. "
                f"Supported: {list(_LABELERS.keys())}. "
                "Pass a custom labeler= argument for other protocols."
            )

    # ---- 2. 收集 PCAP ----
    if pcap_paths is None:
        pcap_paths = _collect_pcaps(data_dir, max_pcaps)
    if not pcap_paths:
        raise RuntimeError(f"No valid PCAP files found in '{data_dir}'")

    # ---- 3. 解析事件 ----
    pipeline = PCAPPipeline()
    events: List[MessageEvent] = []
    for p in pcap_paths:
        try:
            t = pipeline.run(p)
            events.extend(t.events)
        except Exception as exc:
            print(f"[WARN] Failed to parse {p}: {exc}", file=sys.stderr)
    if not events:
        raise RuntimeError("No events parsed from any PCAP file.")
    events = filter_events_for_protocol(proto_upper, events)
    events.sort(key=lambda e: e.timestamp)
    trace = Trace(events=events)
    sessions_all = _group_sessions(trace)
    return run_full_evaluation_from_sessions(
        protocol=protocol,
        sessions_all=sessions_all,
        labeler=labeler,
        seed=seed,
        test_ratio=test_ratio,
        max_sessions=max_sessions,
    )


def run_benchmark(
    data_root: str,
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
    max_pcaps: int = 6,
) -> List[FullEvalResult]:
    """
    对 data_root 下所有已知工业协议跑完整评估，返回结果列表。
    """
    results: List[FullEvalResult] = []
    for proto, subdir in _PROTOCOL_DIRS.items():
        # 去重（MODBUS 和 MODBUSTCP 指向同一目录）
        if proto != list(_LABELERS.keys())[list(_PROTOCOL_DIRS.values()).index(subdir)]:
            continue
        d = os.path.join(data_root, subdir)
        if not os.path.isdir(d):
            continue
        try:
            r = run_full_evaluation(
                protocol=proto,
                data_dir=d,
                seed=seed,
                test_ratio=test_ratio,
                max_sessions=max_sessions,
                max_pcaps=max_pcaps,
            )
            results.append(r)
            print(f"[OK] {proto}: {_format_result(r)}")
        except Exception as exc:
            print(f"[ERROR] {proto}: {exc}", file=sys.stderr)
    return results


def _format_result(r: FullEvalResult) -> str:
    e = r.enhanced_efsm
    if not e:
        return "(no enhanced metrics)"
    return (
        f"guard_f1={e.get('guard_f1', 0):.3f} "
        f"viol={e.get('guard_violation_rate', 0):.3f} "
        f"action_jaccard={e.get('action_coverage_jaccard', 0):.3f} "
        f"diff_acc={e.get('state_diff_accuracy', 0):.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run enhanced EFSM evaluation against known industrial protocol GT specs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--protocol", "-P",
        required=False,
        help="Protocol name (MODBUS, DNP3, IEC104). "
             "Omit to run benchmark on all protocols under --data-root.",
    )
    p.add_argument(
        "--data-dir", "-d",
        default=None,
        help="Directory containing PCAP files for the specified protocol.",
    )
    p.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "Data"),
        help="Root data directory for benchmark mode (used when --protocol is omitted).",
    )
    p.add_argument("--max-pcaps", type=int, default=6)
    p.add_argument("--max-sessions", type=int, default=200)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-sessions", type=int, default=0)
    p.add_argument("--synthetic-session-len", type=int, default=20)
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to write JSON results.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.protocol:
        # 单协议模式
        if args.synthetic_sessions and args.synthetic_sessions > 0:
            labeler = _LABELERS.get(args.protocol.upper())
            if labeler is None:
                raise ValueError(
                    f"Unknown protocol '{args.protocol}' for synthetic mode. "
                    f"Supported: {list(_LABELERS.keys())}"
                )
            sessions_all = generate_synthetic_sessions(
                protocol=args.protocol,
                n_sessions=args.synthetic_sessions,
                session_len=args.synthetic_session_len,
                seed=args.seed,
            )
            result = run_full_evaluation_from_sessions(
                protocol=args.protocol,
                sessions_all=sessions_all,
                labeler=labeler,
                seed=args.seed,
                test_ratio=args.test_ratio,
                max_sessions=args.max_sessions,
            )
        else:
            if args.data_dir is None:
                subdir = _PROTOCOL_DIRS.get(args.protocol.upper(), args.protocol)
                args.data_dir = os.path.join(str(args.data_root), str(subdir))
            result = run_full_evaluation(
                protocol=args.protocol,
                data_dir=args.data_dir,
                seed=args.seed,
                test_ratio=args.test_ratio,
                max_sessions=args.max_sessions,
                max_pcaps=args.max_pcaps,
            )
        output = result.to_dict()
        _print_result(result)
    else:
        # Benchmark 模式
        results = run_benchmark(
            data_root=os.path.realpath(args.data_root),
            seed=args.seed,
            test_ratio=args.test_ratio,
            max_sessions=args.max_sessions,
            max_pcaps=args.max_pcaps,
        )
        output = [r.to_dict() for r in results]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {args.output}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


def _print_result(r: FullEvalResult) -> None:
    print(f"\n{'='*60}")
    print(f"Protocol : {r.protocol}")
    print(f"Total sessions (after cap): {r.total_sessions}")
    print(f"Train sessions : {r.train_sessions}")
    print(f"Test  sessions : {r.test_sessions}")
    print(f"\n--- Legacy EFSM Metrics ---")
    for k, v in sorted(r.legacy_efsm.items()):
        print(f"  {k:35s}: {v:.4f}")
    print(f"\n--- End-to-end Trace Replay (primary) ---")
    print(f"  replay_eval_on                      : {r.replay_on}")
    if r.replay_metrics:
        for k, v in sorted(r.replay_metrics.items()):
            if isinstance(v, float):
                print(f"  {k:35s}: {v:.4f}")
            else:
                print(f"  {k:35s}: {v}")
    else:
        print("  (no replay metrics)")
    print(f"\n--- Enhanced EFSM / GT (reference) ---")
    if r.enhanced_efsm:
        for k, v in sorted(r.enhanced_efsm.items()):
            print(f"  {k:35s}: {v:.4f}")
    else:
        print("  (no GT spec available for this protocol)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
