"""
P-EFSM 效果展示脚本
===================

使用 Data/ 中的真实数据集训练 P-EFSM，并打印：
  1. 现有 EFSM 评估指标（复用项目已有评估器）
  2. P-EFSM 结构统计指标
  3. 概率转移分布，展示 probability / count / confidence 如何体现

默认协议: MODBUS
默认数据目录: Data/MODBUS

运行示例:
  python test/evaluation_test/eval_pefsm.py
  python test/evaluation_test/eval_pefsm.py --protocol MODBUS --data-dir Data/MODBUS --max-pcaps 6
"""

from __future__ import annotations

import argparse
import math
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


def _probability_groups(pefsm: PEFSM):
    grouped = defaultdict(list)
    for tran in pefsm.transitions:
        grouped[(tran.src, tran.symbol)].append(tran)
    return grouped


def _entropy(probs: List[float]) -> float:
    vals = [p for p in probs if p > 0]
    if not vals:
        return 0.0
    return -sum(p * math.log(p, 2) for p in vals)


def _print_eval_metrics(result: FullEvalResult) -> None:
    print(_SEP)
    print("[1] EFSM 评估指标（复用现有评估器）")
    print(_SUB)
    print(f"protocol        : {result.protocol}")
    print(f"train_sessions  : {result.train_sessions}")
    print(f"test_sessions   : {result.test_sessions}")

    print("\nLegacy EFSM Metrics:")
    for k, v in sorted(result.legacy_efsm.items()):
        print(f"  {k:<28} {_fmt(v)}")

    print("\nEnhanced EFSM Metrics:")
    if result.enhanced_efsm:
        for k, v in sorted(result.enhanced_efsm.items()):
            print(f"  {k:<28} {_fmt(v)}")
    else:
        print("  (该协议暂无 GT 指标)")


def _print_pefsm_metrics(pefsm: PEFSM, train_trace: Trace) -> None:
    grouped = _probability_groups(pefsm)
    confidence_counter = Counter((t.confidence or "unknown") for t in pefsm.transitions)
    branch_groups = {k: v for k, v in grouped.items() if len(v) > 1}
    entropies = [_entropy([t.prob or 0.0 for t in trans]) for trans in grouped.values()]

    print("\n" + _SEP)
    print("[2] P-EFSM 结构与概率指标")
    print(_SUB)
    print(f"states                         : {len(pefsm.states)}")
    print(f"transitions                    : {len(pefsm.transitions)}")
    print(f"abstract_messages(train)       : {len(train_trace.abstract_messages)}")
    print(f"probabilistic_groups           : {len(grouped)}")
    print(f"branching_groups(>1 edges)     : {len(branch_groups)}")
    print(f"avg_group_entropy(bits)        : {sum(entropies) / len(entropies) if entropies else 0.0:.4f}")
    print(f"confidence_high                : {confidence_counter.get('high', 0)}")
    print(f"confidence_medium              : {confidence_counter.get('medium', 0)}")
    print(f"confidence_low                 : {confidence_counter.get('low', 0)}")


def _print_probability_demo(pefsm: PEFSM, top_k: int = 12) -> None:
    grouped = _probability_groups(pefsm)
    by_src = defaultdict(list)
    for tran in pefsm.transitions:
        by_src[tran.src].append(tran)

    print("\n" + _SEP)
    print("[3] 概率转移如何体现")
    print(_SUB)
    print("说明1: 模型内部保存的是条件概率 P(dst | src, symbol)。")
    print("说明2: 如果当前 EFSM 已被确定性化，同一个 (src, symbol) 只有一条边，那么该条件概率会自然变成 1.0。")
    print("说明3: 为了更直观看到概率层效果，下面额外打印按源状态 src 归一化后的出边频率分布。\n")

    ranked_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(t.traverse_count for t in item[1]),
            item[0][0],
            item[0][1],
        ),
    )

    shown = 0
    print("[A] 条件概率 P(dst | src, symbol)")
    for (src, symbol), transitions in ranked_groups:
        transitions = sorted(
            transitions,
            key=lambda t: (-(t.prob or 0.0), -t.traverse_count, t.dst),
        )
        total = sum(t.traverse_count for t in transitions)
        if total <= 0:
            continue
        shown += 1
        print(f"  [Group {shown}] src=s{src}, symbol={symbol}, total_count={total}")
        for tran in transitions:
            prob = tran.prob or 0.0
            print(
                f"    s{tran.src} --[{tran.symbol}]--> s{tran.dst}"
                f"   count={tran.traverse_count:<4d}"
                f" prob={prob:.4f}"
                f" confidence={tran.confidence}"
            )
        if shown >= top_k:
            break

    print("\n[B] 按源状态归一化的出边频率分布")
    src_ranked = sorted(
        by_src.items(),
        key=lambda item: (-sum(t.traverse_count for t in item[1]), item[0]),
    )
    shown = 0
    for src, transitions in src_ranked:
        total = sum(t.traverse_count for t in transitions)
        if total <= 0 or len(transitions) <= 1:
            continue
        shown += 1
        print(f"  [State {shown}] src=s{src}, outgoing_total={total}")
        ordered = sorted(transitions, key=lambda t: (-t.traverse_count, t.symbol, t.dst))
        for tran in ordered:
            state_prob = tran.traverse_count / total if total > 0 else 0.0
            print(
                f"    s{tran.src} --[{tran.symbol}]--> s{tran.dst}"
                f"   count={tran.traverse_count:<4d}"
                f" state_prob={state_prob:.4f}"
                f" cond_prob={(tran.prob or 0.0):.4f}"
                f" confidence={tran.confidence}"
            )
        print()
        if shown >= top_k:
            break

    top_transitions = sorted(
        pefsm.transitions,
        key=lambda t: (-t.traverse_count, -(t.prob or 0.0), t.src, t.symbol, t.dst),
    )[:top_k]

    print(_SUB)
    print(f"Top-{top_k} transitions by traverse_count:")
    for tran in top_transitions:
        print(
            f"  s{tran.src} --[{tran.symbol}]--> s{tran.dst}"
            f"   count={tran.traverse_count}"
            f" prob={(tran.prob or 0.0):.4f}"
            f" confidence={tran.confidence}"
        )


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
