"""
P-EFSM 效果展示脚本
===================

使用 Data/ 中的真实数据集训练 P-EFSM，并打印：
  1. 核心 EFSM 指标（precision / recall / f1）
  2. P-EFSM 结构统计指标
  3. 同一状态多条出边的计数与转移概率

默认协议: MODBUS
默认数据目录: Data/MODBUS
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.probabilistic_layer.pipeline import ProbabilisticPipeline
from protocol_infer.core.model.pefsm import PEFSM
from protocol_infer.evaluation.run_evaluation import run_full_evaluation, FullEvalResult
from protocol_infer.evaluation.supervised_eval import _group_sessions, _split_keys, _looks_like_capture


_SEP = "=" * 76
_SUB = "-" * 76

_PROTO_SUBDIR = {
    "MODBUS": "MODBUS",
    "MODBUSTCP": "MODBUS",
    "IEC104": "IEC60870-104",
    "IEC60870-104": "IEC60870-104",
    "DNP3": "DNP3",
}


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _collect_pcaps(data_dir: str, max_pcaps: int) -> List[str]:
    paths: List[str] = []
    if not os.path.isdir(data_dir):
        return paths
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".pcap"):
            continue
        full = os.path.join(data_dir, fn)
        if _looks_like_capture(full):
            paths.append(full)
        if len(paths) >= max_pcaps:
            break
    return paths


def _load_trace(pcap_paths: List[str]) -> Trace:
    pipeline = PCAPPipeline()
    events: List[MessageEvent] = []
    for path in pcap_paths:
        trace = pipeline.run(path)
        events.extend(trace.events)
    events.sort(key=lambda e: e.timestamp)
    return Trace(events=events)


def _split_sessions(
    trace: Trace,
    seed: int,
    test_ratio: float,
    max_sessions: int,
) -> Tuple[Dict[SessionKey, List[MessageEvent]], Dict[SessionKey, List[MessageEvent]], List[SessionKey]]:
    sessions_all = _group_sessions(trace)
    keys = list(sessions_all.keys())
    if len(keys) > max_sessions:
        rng = random.Random(seed)
        rng.shuffle(keys)
        keys = keys[:max_sessions]
        sessions_all = {k: sessions_all[k] for k in keys}

    if len(sessions_all) < 2:
        only_key = next(iter(sessions_all.keys()))
        return {only_key: sessions_all[only_key]}, {}, [only_key]

    train_keys, test_keys = _split_keys(list(sessions_all.keys()), test_ratio, seed)
    train_sessions = {k: sessions_all[k] for k in train_keys}
    test_sessions = {k: sessions_all[k] for k in test_keys}
    return train_sessions, test_sessions, train_keys


def _build_pefsm(
    train_sessions: Dict[SessionKey, List[MessageEvent]],
    train_keys: List[SessionKey],
) -> Tuple[PEFSM, Trace]:
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

    pp = ProbabilisticPipeline()
    pefsm = pp.run(efsm=efsm, trace=train_trace)
    return pefsm, train_trace


def _state_groups(pefsm: PEFSM):
    grouped = defaultdict(list)
    for tran in pefsm.transitions:
        grouped[tran.src].append(tran)
    return grouped


def _print_eval_metrics(result: FullEvalResult) -> None:
    enhanced = result.enhanced_efsm or {}
    legacy = result.legacy_efsm or {}
    preferred = enhanced if {"guard_precision", "guard_recall", "guard_f1"}.issubset(enhanced.keys()) else legacy
    source = "enhanced_efsm" if preferred is enhanced else "legacy_efsm"

    print(_SEP)
    print("[1] 端到端重放（主指标）")
    print(_SUB)
    print(f"protocol        : {result.protocol}")
    print(f"replay_eval_on  : {result.replay_on}")
    rm = result.replay_metrics or {}
    print(f"sess_replay_acc : {_fmt(rm.get('session_replay_accuracy', 0.0))}")
    print(
        f"sess_ok / total : {rm.get('sessions_full_replay_ok', 0)} / "
        f"{rm.get('sessions_replay_evaluated', 0)}"
    )
    print(f"step_replay_acc : {_fmt(rm.get('step_replay_accuracy', 0.0))}")
    print(f"steps matched   : {rm.get('steps_matched', 0)} / {rm.get('steps_total', 0)}")
    print(_SUB)
    print("[1b] Guard 字段级（GT 参考）")
    print(_SUB)
    print(f"metric_source   : {source}")
    print(f"guard_precision : {_fmt(preferred.get('guard_precision', 0.0))}")
    print(f"guard_recall    : {_fmt(preferred.get('guard_recall', 0.0))}")
    print(f"guard_f1        : {_fmt(preferred.get('guard_f1', 0.0))}")


def _print_pefsm_metrics(pefsm: PEFSM, train_trace: Trace) -> None:
    grouped = _state_groups(pefsm)
    confidence_counter = Counter((t.confidence or "unknown") for t in pefsm.transitions)
    branching_states = {sid: trans for sid, trans in grouped.items() if len(trans) > 1}

    print("\n" + _SEP)
    print("[2] P-EFSM 结构指标")
    print(_SUB)
    print(f"states                         : {len(pefsm.states)}")
    print(f"transitions                    : {len(pefsm.transitions)}")
    print(f"abstract_messages(train)       : {len(train_trace.abstract_messages)}")
    print(f"branching_states(>1 edges)     : {len(branching_states)}")
    print(f"confidence_high                : {confidence_counter.get('high', 0)}")
    print(f"confidence_medium              : {confidence_counter.get('medium', 0)}")
    print(f"confidence_low                 : {confidence_counter.get('low', 0)}")


def _print_probability_demo(pefsm: PEFSM, top_k: int = 12) -> None:
    grouped = _state_groups(pefsm)

    print("\n" + _SEP)
    print("[3] 概率转移如何体现")
    print(_SUB)
    print("说明: 对同一源状态 src 的所有出边统计 count，再归一化得到 state-level transition probability。")
    print("也就是: P(transition | src) = count(src -> edge) / sum_count(all outgoing edges from src)\n")

    ranked_states = sorted(
        grouped.items(),
        key=lambda item: (-sum(t.traverse_count for t in item[1]), item[0]),
    )

    shown = 0
    for src, transitions in ranked_states:
        total = sum(t.traverse_count for t in transitions)
        if total <= 0 or len(transitions) <= 1:
            continue
        shown += 1
        print(f"[State {shown}] src=s{src}, outgoing_total={total}")
        ordered = sorted(transitions, key=lambda t: (-t.traverse_count, t.symbol, t.dst))
        for tran in ordered:
            print(
                f"  s{tran.src} --[{tran.symbol}]--> s{tran.dst}"
                f"   count={tran.traverse_count:<4d}"
                f" state_prob={(tran.prob or 0.0):.4f}"
                f" confidence={tran.confidence}"
            )
        print()
        if shown >= top_k:
            break


def run_demo(
    protocol: str,
    data_dir: str,
    seed: int,
    test_ratio: float,
    max_sessions: int,
    max_pcaps: int,
) -> None:
    pcap_paths = _collect_pcaps(data_dir, max_pcaps)
    if not pcap_paths:
        raise RuntimeError(f"在 {data_dir} 下没有找到可用 pcap")

    result = run_full_evaluation(
        protocol=protocol,
        data_dir=data_dir,
        seed=seed,
        test_ratio=test_ratio,
        max_sessions=max_sessions,
        max_pcaps=max_pcaps,
    )

    trace = _load_trace(pcap_paths)
    train_sessions, _, train_keys = _split_sessions(
        trace=trace,
        seed=seed,
        test_ratio=test_ratio,
        max_sessions=max_sessions,
    )
    pefsm, train_trace = _build_pefsm(train_sessions, train_keys)

    print(f"使用数据集: {data_dir}")
    print(f"参与训练的 pcap 数: {len(pcap_paths)}")
    for path in pcap_paths:
        print(f"  - {os.path.basename(path)}")

    _print_eval_metrics(result)
    _print_pefsm_metrics(pefsm, train_trace)
    _print_probability_demo(pefsm)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用真实数据集训练并展示 P-EFSM 效果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", "-P", default="MODBUS", help="协议名")
    parser.add_argument("--data-dir", "-d", default=None, help="数据目录")
    parser.add_argument("--data-root", default=os.path.join(_ROOT, "Data"), help="数据根目录")
    parser.add_argument("--max-pcaps", type=int, default=6)
    parser.add_argument("--max-sessions", type=int, default=200)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    protocol = args.protocol.upper()
    data_dir = args.data_dir
    if data_dir is None:
        subdir = _PROTO_SUBDIR.get(protocol, protocol)
        data_dir = os.path.join(args.data_root, subdir)

    run_demo(
        protocol=protocol,
        data_dir=data_dir,
        seed=args.seed,
        test_ratio=args.test_ratio,
        max_sessions=args.max_sessions,
        max_pcaps=args.max_pcaps,
    )


if __name__ == "__main__":
    main()
