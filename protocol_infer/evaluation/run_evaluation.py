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
from dataclasses import dataclass
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
    _split_keys,
    _mutate_negative,
    _build_symbol_var_sequences,
    ModbusTCPLabeler,
    IEC104Labeler,
    DNP3Labeler,
    ProtocolLabeler,
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
}

_PROTOCOL_DIRS: Dict[str, str] = {
    "MODBUS": "MODBUS",
    "MODBUSTCP": "MODBUS",
    "IEC104": "IEC60870-104",
    "IEC60870-104": "IEC60870-104",
    "DNP3": "DNP3",
}


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class FullEvalResult:
    protocol: str
    train_sessions: int
    test_sessions: int
    # 原有历史指标（与 EFSMevaluator 兼容）
    legacy_efsm: Dict[str, float]
    # 新四维指标
    enhanced_efsm: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "train_sessions": self.train_sessions,
            "test_sessions": self.test_sessions,
            "legacy_efsm": self.legacy_efsm,
            "enhanced_efsm": self.enhanced_efsm,
        }


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def _collect_pcaps(data_dir: str, max_pcaps: int) -> List[str]:
    """从目录中收集有效的 PCAP 文件路径。"""
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
    events.sort(key=lambda e: e.timestamp)
    trace = Trace(events=events)

    # ---- 4. 会话分组 & 切分 ----
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
        split_at = max(1, min(int(len(evs) * (1.0 - test_ratio)), len(evs) - 1))
        train_evs, test_evs_raw = evs[:split_at], evs[split_at:]
        test_key = SessionKey(
            ip1=only_key.ip1, port1=only_key.port1,
            ip2=f"{only_key.ip2}#test", port2=only_key.port2,       # 区分test与train
            protocol=only_key.protocol,
        )
        test_evs = [
            MessageEvent(session_key=test_key, timestamp=e.timestamp,
                         payload=e.payload, direction=e.direction)
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

    # ---- 5. 推断 FSM + EFSM ----
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

    # ---- 6. 原有历史指标（EFSMevaluator）----
    fp = FeatureProcessor(
        apriori_positions=cf.get_apriori_positions(),
        apriori_static_items=cf.get_apriori_static_items(),
    )
    test_pairs, test_syms, _ = _build_symbol_var_sequences(cf, fp, test_sessions)
    neg_pairs = _mutate_negative(test_pairs, seed=seed + 1)     # 改变seed, 避免相关
    numeric_vars = {k for k in efsm.variable_defs if k.startswith("b") or k == "len"}
    legacy_res = EFSMevaluator().evaluate(
        efsm=efsm,
        sequences=test_pairs,
        negative_sequences=neg_pairs,
        numeric_vars=numeric_vars,
    )

    # ---- 7. 模块一：字段级提取 -> SessionTrace ----
    field_extractor = FieldExtractor(
        protocol=proto_upper,
        labeler=labeler.label,
    )
    session_traces: Dict[SessionKey, SessionTrace] = field_extractor.extract_sessions(test_sessions)

    # ---- 7b. 填充 abstract_sym 和 abstract_vars（供增强评估器使用）----
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

    # ---- 8. 模块二：获取 GT ----
    gt: Optional[ProtocolGT] = get_gt(proto_upper)
    if gt is None:
        print(
            f"[WARN] No GT spec for protocol '{protocol}'. "
            "Enhanced metrics will be skipped.",
            file=sys.stderr,
        )
        enhanced_res: Dict[str, float] = {}
    else:
        # ---- 9. 模块三：增强评估 ----
        evaluator = EnhancedEFSMEvaluator()
        eval_result: EFSMEvalResult = evaluator.evaluate(
            efsm=efsm,
            gt=gt,
            session_traces=session_traces,
        )
        enhanced_res = eval_result.to_dict()

    return FullEvalResult(
        protocol=proto_upper,
        train_sessions=len(train_sessions),
        test_sessions=len(test_sessions),
        legacy_efsm=legacy_res,
        enhanced_efsm=enhanced_res,
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
        if args.data_dir is None:
            subdir = _PROTOCOL_DIRS.get(args.protocol.upper(), args.protocol)
            args.data_dir = os.path.join(args.data_root, subdir)
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
    print(f"Train sessions : {r.train_sessions}")
    print(f"Test  sessions : {r.test_sessions}")
    print(f"\n--- Legacy EFSM Metrics ---")
    for k, v in sorted(r.legacy_efsm.items()):
        print(f"  {k:35s}: {v:.4f}")
    print(f"\n--- Enhanced EFSM Metrics ---")
    if r.enhanced_efsm:
        for k, v in sorted(r.enhanced_efsm.items()):
            print(f"  {k:35s}: {v:.4f}")
    else:
        print("  (no GT spec available for this protocol)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
