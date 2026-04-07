"""
EFSM 学习结果检视工具

对比推断出的自动机 guard/action/字段划分与协议真值规范（GT），
生成人类可读的逐项对比报告。

用法
----
python -m protocol_infer.evaluation.inspect_efsm \
    --protocol MODBUS \
    --data-dir /path/to/Data/MODBUS \
    [--max-pcaps 6] [--output report.txt]

也可作为 API 使用：
    from protocol_infer.evaluation.inspect_efsm import inspect_and_report
    report = inspect_and_report(protocol="MODBUS", data_dir="...")
    print(report)
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from protocol_infer.core.datamodel.event import MessageEvent
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
    ModbusTCPLabeler,
    IEC104Labeler,
    DNP3Labeler,
)
from protocol_infer.evaluation.gt_guard_action import (
    get_gt,
    ProtocolGT,
    TransitionSpec,
    PROTOCOL_VAR_ALIASES,
    normalize_var_names,
)
from protocol_infer.core.model.efsm import EFSM


# ---------------------------------------------------------------------------
# Labeler 注册
# ---------------------------------------------------------------------------

_LABELERS = {
    "MODBUS": ModbusTCPLabeler(),
    "MODBUSTCP": ModbusTCPLabeler(),
    "IEC104": IEC104Labeler(),
    "IEC60870-104": IEC104Labeler(),
    "DNP3": DNP3Labeler(),
}

_PROTOCOL_DIRS = {
    "MODBUS": "MODBUS",
    "IEC104": "IEC60870-104",
    "DNP3": "DNP3",
}

# 协议已知字段的字节范围（用于字段划分报告）
_PROTOCOL_FIELD_LAYOUT: Dict[str, List[Tuple[str, List[int], str]]] = {
    "MODBUS": [
        ("txn_id",    [0, 1],     "Transaction ID（每请求递增）"),
        ("proto_id",  [2, 3],     "Protocol ID（Modbus=0x0000，常量）"),
        ("length",    [4, 5],     "Remaining Bytes Count"),
        ("unit_id",   [6],        "Unit/Slave ID"),
        ("fc",        [7],        "Function Code（关键 guard 字段）"),
        ("start_addr",[8, 9],     "Starting Address（请求）"),
        ("byte_count",[8],        "Byte Count（响应）"),
        ("quantity",  [10, 11],   "Quantity（请求）"),
    ],
    "DNP3": [
        ("start",     [0, 1],     "Start Bytes（固定 0x0564）"),
        ("length",    [2],        "Frame Length"),
        ("ctrl",      [3],        "Control（DIR/PRM/FCB/FCV/FC）"),
        ("dest",      [4, 5],     "Destination Address"),
        ("src",       [6, 7],     "Source Address"),
        ("crc1",      [8, 9],     "Header CRC"),
    ],
    "IEC104": [
        ("start",     [0],        "Start Byte（固定 0x68）"),
        ("apdu_len",  [1],        "APDU Length"),
        ("ctrl1",     [2],        "Control Field 1（帧类型/I/S/U）"),
        ("ctrl2",     [3],        "Control Field 2"),
        ("ctrl3",     [4],        "Control Field 3（I 帧 NS 高字节）"),
        ("ctrl4",     [5],        "Control Field 4（I 帧 NR）"),
        ("type_id",   [6],        "Type Identification（I 帧）"),
        ("vsq",       [7],        "Variable Structure Qualifier（I 帧）"),
    ],
}


# ---------------------------------------------------------------------------
# 闭包解析工具
# ---------------------------------------------------------------------------

def _extract_closure_vars(fn) -> Dict[str, Any]:
    """
    从 Python 闭包中提取被捕获的局部变量（nonlocals）。
    若函数不是闭包，或提取失败，则返回空字典。
    """
    if fn is None:
        return {}
    try:
        cvars = inspect.getclosurevars(fn)
        return dict(cvars.nonlocals)
    except Exception:
        return {}


def _unwrap_fn(fn, target_param: str = "_base_guard") -> Any:
    """
    EFSMInferencer 会把 learner 产生的 guard/action 用默认参数值的方式包一层：
        def wrapped_guard(vars, memory=None, _base_guard=base_guard, ...)

    这些值存在 fn.__defaults__ 中，无法通过 getclosurevars() 读取。
    本函数使用 inspect.signature 按参数名提取默认值，返回内层函数。
    """
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        if target_param in sig.parameters:
            default = sig.parameters[target_param].default
            if default is not inspect.Parameter.empty:
                return default
    except (ValueError, TypeError):
        pass
    return fn


def extract_cross_message_desc(guard_fn) -> Dict[str, Any]:
    """
    从 wrapped_guard 中提取 _memory_guard（CrossMessageLearner 产生的跨消息规则）。
    CrossMessageLearner 的 guard 闭包捕获了 identity_rules、seq_rules、linear_rules。
    """
    if guard_fn is None:
        return {"_type": "none"}
    try:
        sig = inspect.signature(guard_fn)
        memory_guard = (
            sig.parameters["_memory_guard"].default
            if "_memory_guard" in sig.parameters
            else inspect.Parameter.empty
        )
        if memory_guard is inspect.Parameter.empty or memory_guard is None:
            return {"_type": "none"}
        cvars = inspect.getclosurevars(memory_guard)
        nl = cvars.nonlocals
        if not nl:
            return {"_type": "opaque"}
        return {
            "_type": "cross_message",
            "identity_rules": nl.get("identity_rules", []),
            "seq_rules":      nl.get("seq_rules", []),
            "linear_rules":   nl.get("linear_rules", []),
        }
    except Exception:
        return {"_type": "opaque"}


def extract_guard_desc(guard_fn) -> Dict[str, Any]:
    """
    从 AprioriGuardLearner 生成的 guard 闭包中提取可读描述。

    EFSMInferencer 会在外层套一个 wrapped_guard，需先解包得到 _base_guard，
    再从中提取 single_constraints、joint_rules 等。

    返回字典包含：
      single_constraints : Dict[str, tuple]  -- {变量名: (类型, 值)}
      joint_rules        : list              -- 关联规则列表
      linear_pairs       : list              -- 线性对列表
      triplet_sums       : list              -- 三元和列表
    """
    if guard_fn is None:
        return {"_type": "none"}

    # 解包 wrapped_guard → base_guard
    inner = _unwrap_fn(guard_fn)
    if inner is None:
        return {"_type": "none"}

    nonlocals = _extract_closure_vars(inner)
    if not nonlocals:
        return {"_type": "opaque"}

    # 检查是否是 Apriori guard 闭包（应含 single_constraints）
    if "single_constraints" not in nonlocals:
        return {"_type": "opaque"}

    return {
        "_type": "apriori",
        "single_constraints": nonlocals.get("single_constraints", {}),
        "joint_rules":        nonlocals.get("joint_rules", []),
        "linear_pairs":       nonlocals.get("linear_pairs", []),
        "triplet_sums":       nonlocals.get("triplet_sums", []),
    }


def extract_action_desc(action_fn) -> Dict[str, Any]:
    """从 action 闭包中提取 action_rules。"""
    if action_fn is None:
        return {"_type": "none"}

    # 解包 wrapped_action → base_action
    inner = _unwrap_fn(action_fn, "_base_action")
    if inner is None:
        return {"_type": "none"}

    nonlocals = _extract_closure_vars(inner)

    # action_rules 存在说明是 Apriori 学习到的完整 action
    if "action_rules" in nonlocals:
        return {
            "_type": "apriori",
            "action_rules": nonlocals.get("action_rules", {}),
        }

    # 否则尝试判断是否为恒等映射（样本不足时的 fallback: lambda vars: vars.copy()）
    try:
        test_in = {"_test": 1.0}
        test_out = inner(test_in)
        if test_out == test_in:
            return {"_type": "identity"}
    except Exception:
        pass

    if not nonlocals:
        return {"_type": "opaque"}
    return {"_type": "opaque"}


# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------

def _alias(var_name: str, protocol: str) -> str:
    """
    将字节位变量名（如 b7）转换为语义名（如 fc）。
    若无别名，返回原名。
    """
    aliases = PROTOCOL_VAR_ALIASES.get(protocol.upper(), {})
    mapped = aliases.get(var_name, var_name)
    if isinstance(mapped, list):
        return "/".join(mapped)
    return mapped


def _fmt_constraint(var_name: str, constraint: tuple, protocol: str) -> str:
    """将单变量约束格式化为可读字符串。"""
    sem = _alias(var_name, protocol)
    label = f"{var_name}" if sem == var_name else f"{var_name}({sem})"
    ctype = constraint[0]
    if ctype == "eq":
        val = constraint[1]
        return f"{label} == {val!r}"
    if ctype == "in":
        vals = sorted(constraint[1])
        vals_str = "{" + ", ".join(f"{v!r}" for v in vals) + "}"
        return f"{label} in {vals_str}"
    if ctype == "range":
        lo, hi = constraint[1]
        return f"{lo!r} <= {label} <= {hi!r}"
    if ctype == "delta":
        return f"{label} += {constraint[1]!r}  [序列]"
    return f"{label} ~ {constraint!r}"


def _fmt_guard_desc(guard_desc: Dict, protocol: str, indent: str = "    ") -> List[str]:
    """将 guard_desc 格式化为多行字符串列表。"""
    lines = []
    gtype = guard_desc.get("_type", "unknown")

    if gtype == "none":
        lines.append(f"{indent}(无 guard — 始终可触发)")
        return lines
    if gtype == "opaque":
        lines.append(f"{indent}(guard 不可解析)")
        return lines

    # 单变量约束（按变量名排序；过多时截断展示）
    single = guard_desc.get("single_constraints", {})
    max_single_show = 16
    if single:
        items = sorted(single.items())
        lines.append(f"{indent}[单变量约束]{' (其余已省略)' if len(items) > max_single_show else ''}")
        for name, cst in items[:max_single_show]:
            lines.append(f"{indent}  {_fmt_constraint(name, cst, protocol)}")
    else:
        lines.append(f"{indent}[单变量约束] (无)")

    # 关联规则（按置信度已排序；展示截断减轻冗余）
    joint = guard_desc.get("joint_rules", [])
    max_joint_show = 8
    if joint:
        lines.append(f"{indent}[关联规则]{' (其余已省略)' if len(joint) > max_joint_show else ''}")
        # 折叠“互为蕴含”的循环规则：A->B 与 B->A（通常由样本分布导致的等价共现）
        # 仅在 1->1 且 conf≈1 时折叠，避免错误合并一般规则。
        folded = []
        used = set()
        for i, (a_vars, a_vals, c_vars, c_vals, conf) in enumerate(joint):
            if i in used:
                continue
            if (
                conf >= 0.999
                and len(a_vars) == 1
                and len(c_vars) == 1
                and len(a_vals) == 1
                and len(c_vals) == 1
            ):
                a = (a_vars[0], a_vals[0])
                b = (c_vars[0], c_vals[0])
                j_idx = None
                for j in range(i + 1, len(joint)):
                    if j in used:
                        continue
                    aa, av, cc, cv, cconf = joint[j]
                    if (
                        cconf >= 0.999
                        and len(aa) == 1
                        and len(cc) == 1
                        and len(av) == 1
                        and len(cv) == 1
                        and (aa[0], av[0]) == b
                        and (cc[0], cv[0]) == a
                    ):
                        j_idx = j
                        break
                if j_idx is not None:
                    used.add(i)
                    used.add(j_idx)
                    folded.append(("equiv", a, b, conf))
                    continue
            used.add(i)
            folded.append(("rule", (a_vars, a_vals, c_vars, c_vals, conf)))

        shown = 0
        for item in folded:
            if shown >= max_joint_show:
                break
            if item[0] == "equiv":
                _tag, a, b, conf = item
                (a_name, a_val), (b_name, b_val) = a, b
                a_sem = _alias(a_name, protocol)
                b_sem = _alias(b_name, protocol)
                a_label = a_name if a_sem == a_name else f"{a_name}({a_sem})"
                b_label = b_name if b_sem == b_name else f"{b_name}({b_sem})"
                lines.append(
                    f"{indent}  {a_label}≈{a_val!r} & {b_label}≈{b_val!r}  (conf={conf:.2f}, 共现/等价)"
                )
                shown += 1
                continue

            _tag, (ante_vars, ante_vals, cons_vars, cons_vals, conf) = item
            ante_parts = []
            for n, v in zip(ante_vars, ante_vals):
                sem = _alias(n, protocol)
                tag = n if sem == n else f"{n}({sem})"
                ante_parts.append(f"{tag}≈{v!r}")
            cons_parts = []
            for n, v in zip(cons_vars, cons_vals):
                sem = _alias(n, protocol)
                tag = n if sem == n else f"{n}({sem})"
                cons_parts.append(f"{tag}≈{v!r}")
            rule_str = " & ".join(ante_parts) + " -> " + " & ".join(cons_parts)
            lines.append(f"{indent}  {rule_str}  (conf={conf:.2f})")
            shown += 1

    max_linear_show = 6
    max_triplet_show = 6
    # 线性关系
    linear = guard_desc.get("linear_pairs", [])
    if linear:
        lines.append(f"{indent}[线性关系]{' (其余已省略)' if len(linear) > max_linear_show else ''}")
        for a, b, k, c, r2 in linear[:max_linear_show]:
            sa, sb = _alias(a, protocol), _alias(b, protocol)
            la = a if sa == a else f"{a}({sa})"
            lb = b if sb == b else f"{b}({sb})"
            sign = "+" if c >= 0 else "-"
            lines.append(
                f"{indent}  {la} = {k:.4g}·{lb} {sign} {abs(c):.4g}  (R²={r2:.4f})"
            )

    # 三元和
    triplets = guard_desc.get("triplet_sums", [])
    if triplets:
        lines.append(f"{indent}[三元和关系]{' (其余已省略)' if len(triplets) > max_triplet_show else ''}")
        for a, b, c_var, res in triplets[:max_triplet_show]:
            sa, sb, sc = _alias(a, protocol), _alias(b, protocol), _alias(c_var, protocol)
            la = a if sa == a else f"{a}({sa})"
            lb = b if sb == b else f"{b}({sb})"
            lc = c_var if sc == c_var else f"{c_var}({sc})"
            lines.append(f"{indent}  {la} + {lb} = {lc}  (max_res={res:.2e})")

    return lines


def _fmt_action_desc(action_desc: Dict, protocol: str, indent: str = "    ") -> List[str]:
    """将 action_desc 格式化为多行字符串列表。"""
    lines = []
    atype = action_desc.get("_type", "unknown")

    if atype == "none":
        lines.append(f"{indent}(无 action)")
        return lines
    if atype == "identity":
        lines.append(f"{indent}(样本不足 — 所有变量保持不变)")
        return lines
    if atype == "opaque":
        lines.append(f"{indent}(action 不可解析)")
        return lines

    rules = action_desc.get("action_rules", {})
    delta_vars = {n: p for n, (t, p) in rules.items() if t == "delta"}
    keep_vars  = [n for n, (t, _) in rules.items() if t == "keep"]

    max_delta_show = 12
    if delta_vars:
        lines.append(f"{indent}[变化字段]{' (其余已省略)' if len(delta_vars) > max_delta_show else ''}")
        for name, delta in sorted(delta_vars.items())[:max_delta_show]:
            sem = _alias(name, protocol)
            label = name if sem == name else f"{name}({sem})"
            lines.append(f"{indent}  {label} += {delta!r}")

    if keep_vars:
        keep_labels = []
        for n in sorted(keep_vars):
            sem = _alias(n, protocol)
            keep_labels.append(n if sem == n else f"{n}({sem})")
        max_keep_show = 14
        if len(keep_labels) > max_keep_show:
            shown = ", ".join(keep_labels[:max_keep_show])
            lines.append(f"{indent}[不变字段] {shown}, …（共{len(keep_labels)}项）")
        else:
            lines.append(f"{indent}[不变字段] {', '.join(keep_labels)}")

    if not delta_vars and not keep_vars:
        lines.append(f"{indent}(action_rules 为空 — 直接返回原 vars)")

    return lines


def _fmt_cross_message_desc(cross_desc: Dict, protocol: str, indent: str = "    ") -> List[str]:
    """将 CrossMessageLearner 学到的跨消息规则格式化为可读字符串。"""
    lines = []
    aliases = PROTOCOL_VAR_ALIASES.get(protocol.upper(), {})

    def _sem(n):
        m = aliases.get(n, n)
        return f"{n}({m})" if isinstance(m, str) and m != n else n

    identity = cross_desc.get("identity_rules", [])
    seq      = cross_desc.get("seq_rules", [])
    linear   = cross_desc.get("linear_rules", [])

    if identity:
        lines.append(f"{indent}[恒等关系] curr.x == prev.y")
        for dst_var, src_var in identity:
            lines.append(f"{indent}  curr.{_sem(dst_var)} == prev.{_sem(src_var)}")

    if seq:
        lines.append(f"{indent}[序列递增]")
        for var, delta in seq:
            lines.append(f"{indent}  {_sem(var)} += {delta!r}  (逐消息递增)")

    if linear:
        lines.append(f"{indent}[线性关系] curr.dst = k * prev.src + c")
        for dst_var, src_var, k, c, r2 in linear:
            sign = "+" if c >= 0 else "-"
            lines.append(
                f"{indent}  curr.{_sem(dst_var)} = {k:.4g} * prev.{_sem(src_var)} {sign} {abs(c):.4g}"
                f"  (R²={r2:.4f})"
            )

    if not identity and not seq and not linear:
        lines.append(f"{indent}(无跨消息规则)")

    return lines


# ---------------------------------------------------------------------------
# 字段划分报告
# ---------------------------------------------------------------------------

def build_field_report(
    feature_processor: FeatureProcessor,
    protocol: str,
    gt: Optional[ProtocolGT],
) -> str:
    """
    生成字段划分对比报告：
      - 协议真值字段布局
      - Apriori 选出的字节位置（静态/动态）
      - DynamicFieldDetector 补充检测到的字段
      - 覆盖率分析（TP/FP/FN）
    """
    lines: List[str] = []
    proto_upper = protocol.upper()
    aliases = PROTOCOL_VAR_ALIASES.get(proto_upper, {})

    lines.append("=" * 70)
    lines.append("  字段划分对比")
    lines.append("=" * 70)

    # ── 1. 协议真值字段布局 ─────────────────────────────────────────────────
    layout = _PROTOCOL_FIELD_LAYOUT.get(proto_upper, [])
    lines.append("\n【协议真值字段布局】")
    if layout:
        for field_name, byte_positions, desc in layout:
            pos_str = ", ".join(f"b{p}" for p in byte_positions)
            lines.append(f"  {field_name:15s}  [{pos_str}]  {desc}")
    else:
        lines.append("  (未内置该协议的字段布局，请参考协议规范)")

    # ── 2. Apriori 静态字段（常量位置）───────────────────────────────────────
    static_items = feature_processor.apriori_static_items  # {pos: value}
    lines.append("\n【Apriori 检测 — 静态字段（常量字节位置）】")
    if static_items:
        for pos in sorted(static_items.keys()):
            val = static_items[pos]
            sem = _alias(f"s{pos}", proto_upper)
            tag = f"s{pos}" if sem == f"s{pos}" else f"s{pos} -> {sem}"
            lines.append(f"  字节 b{pos:2d}: 常量值 0x{val:02X} ({val})  [{tag}]")
    else:
        lines.append("  (无静态字节位置)")

    # ── 3. Apriori 动态字段（变量位置）───────────────────────────────────────
    dyn_positions = feature_processor.apriori_positions  # List[int]
    lines.append("\n【Apriori 检测 — 动态字段（变量字节位置）】")
    if dyn_positions:
        for pos in sorted(dyn_positions):
            sem = _alias(f"b{pos}", proto_upper)
            tag = f"b{pos}" if sem == f"b{pos}" else f"b{pos} -> {sem}"
            lines.append(f"  字节 b{pos:2d}: 动态变量  [{tag}]")
    else:
        lines.append("  (无动态字节位置)")

    # ── 4. DynamicFieldDetector 补充字段 ─────────────────────────────────────
    dyn_fields = feature_processor.dynamic_fields
    lines.append("\n【DynamicFieldDetector 检测 — 多字节结构化字段】")
    if dyn_fields:
        for f in sorted(dyn_fields, key=lambda x: x.start_pos):
            covered = sorted(f.covers)
            pos_str = ", ".join(f"b{p}" for p in covered)
            sem_parts = set()
            for p in covered:
                s = _alias(f"b{p}", proto_upper)
                if s != f"b{p}":
                    sem_parts.add(s)
            sem_tag = f"→ {'/'.join(sorted(sem_parts))}" if sem_parts else ""
            lines.append(
                f"  {f.var_name:18s}: {f.width}字节 {f.endian}-endian  [{pos_str}]  {sem_tag}"
            )
    else:
        lines.append("  (无补充检测字段)")

    # ── 5. 覆盖率分析 ─────────────────────────────────────────────────────────
    lines.append("\n【字段覆盖率分析】")

    # 学习到的所有字节位置（Apriori 静态 + 动态）
    learned_positions: Set[int] = (
        set(dyn_positions)
        | set(static_items.keys())
        | {p for f in dyn_fields for p in f.covers}
    )

    # GT 中有别名的字节位置（即协议已定义的字段位置）
    gt_positions: Set[int] = set()
    for var_str in aliases:
        if var_str.startswith("b") and var_str[1:].isdigit():
            gt_positions.add(int(var_str[1:]))
        elif var_str.startswith("s") and var_str[1:].isdigit():
            gt_positions.add(int(var_str[1:]))

    tp = learned_positions & gt_positions
    fp = learned_positions - gt_positions
    fn = gt_positions - learned_positions

    lines.append(f"  已知协议字节位置 (GT):  {sorted(gt_positions)}")
    lines.append(f"  学习到的字节位置:        {sorted(learned_positions)}")
    lines.append(
        f"  TP (正确识别): {sorted(tp)}  "
        f"FP (误识别): {sorted(fp)}  "
        f"FN (漏检): {sorted(fn)}"
    )
    if gt_positions:
        recall = len(tp) / len(gt_positions)
        prec   = len(tp) / len(learned_positions) if learned_positions else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        lines.append(f"  Precision={prec:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guard/Action 对比报告
# ---------------------------------------------------------------------------

def _build_sym_to_msgtype_map(
    efsm: EFSM,
    sessions: Dict[SessionKey, List[MessageEvent]],
    labeler,
) -> Dict[str, Set[str]]:
    """
    通过对训练会话做消息类型标注，构建 symbol → msg_type 的映射。
    labeler.label(event) 返回该 event 的协议消息类型字符串。
    """
    sym_to_types: Dict[str, Set[str]] = defaultdict(set)
    return dict(sym_to_types)   # 简单实现：留空，由调用方填充


def build_guard_action_report(
    efsm: EFSM,
    protocol: str,
    gt: Optional[ProtocolGT],
    sym_to_types: Optional[Dict[str, Set[str]]] = None,
) -> str:
    """
    生成 EFSM 每条转移的 guard/action vs GT 对比报告。
    """
    lines: List[str] = []
    proto_upper = protocol.upper()

    lines.append("=" * 70)
    lines.append("  Guard / Action 条件对比（逐转移）")
    lines.append("=" * 70)

    if not efsm.transitions:
        lines.append("  (EFSM 无转移)")
        return "\n".join(lines)

    # 按 (src, dst, symbol) 排序以保证稳定输出
    sorted_trans = sorted(
        efsm.transitions,
        key=lambda t: (t.src, t.symbol, t.dst),
    )

    for tran in sorted_trans:
        src_name = efsm.states[tran.src].name if tran.src in efsm.states else f"s{tran.src}"
        dst_name = efsm.states[tran.dst].name if tran.dst in efsm.states else f"s{tran.dst}"
        sym = tran.symbol
        cnt = tran.traverse_count

        lines.append(f"\n{'─'*60}")
        lines.append(f"  转移: {src_name} ──[{sym}]──> {dst_name}   (触发次数={cnt})")

        # ── 学习到的 Guard（单消息内部约束） ─────────────────────────────
        lines.append("  [学习 Guard - 单消息]")
        guard_desc = extract_guard_desc(tran.guard)
        lines.extend(_fmt_guard_desc(guard_desc, proto_upper, indent="    "))

        # ── 跨消息 Guard（CrossMessageLearner 规则） ──────────────────────
        cross_desc = extract_cross_message_desc(tran.guard)
        if cross_desc.get("_type") == "cross_message":
            lines.append("  [学习 Guard - 跨消息]")
            lines.extend(_fmt_cross_message_desc(cross_desc, proto_upper, indent="    "))

        # ── 学习到的 Action ───────────────────────────────────────────────
        lines.append("  [学习 Action]")
        action_desc = extract_action_desc(tran.action)
        lines.extend(_fmt_action_desc(action_desc, proto_upper, indent="    "))

        # ── GT 参考（尝试匹配） ────────────────────────────────────────────
        if gt is not None:
            matched_gt = _try_match_gt(tran, guard_desc, gt, proto_upper, sym_to_types)
            if matched_gt:
                lines.append("  [GT 参考]")
                for ts in matched_gt:
                    lines.append(f"    规则 [{ts.label}]: {ts.guard.describe()}")
                    if ts.action.description:
                        lines.append(f"    Action: {ts.action.description}")
                # 字段覆盖对比
                lines.append("  [Guard 字段对比]")
                _append_field_comparison(lines, guard_desc, matched_gt, proto_upper)
            else:
                lines.append("  [GT 参考] (未找到匹配的 GT 转移规范)")
        else:
            lines.append("  [GT 参考] (无 GT 规范)")

    return "\n".join(lines)


def _try_match_gt(
    tran,
    guard_desc: Dict,
    gt: ProtocolGT,
    protocol: str,
    sym_to_types: Optional[Dict[str, Set[str]]],
) -> List[TransitionSpec]:
    """
    尝试根据学习到的 guard 约束推断匹配的 GT 转移规范。

    策略：
      1. 若有 sym_to_types 映射，直接用消息类型查找 GT
      2. 否则用 guard 中的关键字段值（如 fc == 3）尝试匹配 GT guard
    """
    sym = tran.symbol

    # 策略 1：symbol → msg_type 映射
    if sym_to_types:
        types = sym_to_types.get(sym, set())
        results = []
        for ts in gt.transitions:
            if types & ts.src_types or types & ts.dst_types:
                results.append(ts)
        if results:
            return results

    # 策略 2：从 guard 的 eq 约束中提取关键字段值匹配 GT
    single = guard_desc.get("single_constraints", {})
    aliases = PROTOCOL_VAR_ALIASES.get(protocol, {})

    # 找出 guard 中所有等值约束对应的语义字段名和值
    semantic_vals: Dict[str, Any] = {}
    for var_name, cst in single.items():
        if cst[0] == "eq":
            sem = aliases.get(var_name, var_name)
            if isinstance(sem, list):
                for s in sem:
                    semantic_vals[s] = cst[1]
            else:
                semantic_vals[sem] = cst[1]

    if not semantic_vals:
        return []

    # 与每条 GT 转移的 guard 做简单相似度打分
    scored: List[Tuple[float, TransitionSpec]] = []
    for ts in gt.transitions:
        score = 0.0
        for c in ts.guard.constraints:
            if c.field_name in semantic_vals:
                actual_val = semantic_vals[c.field_name]
                if c.eq is not None and actual_val == c.eq:
                    score += 2.0
                elif c.in_set is not None and actual_val in c.in_set:
                    score += 1.5
                elif c.in_range is not None:
                    lo, hi = c.in_range
                    if lo <= actual_val <= hi:
                        score += 1.0
        if score > 0:
            scored.append((score, ts))

    scored.sort(key=lambda x: -x[0])
    # 返回得分最高的（最多 2 条）
    return [ts for _, ts in scored[:2]]


def _append_field_comparison(
    lines: List[str],
    guard_desc: Dict,
    gt_transitions: List[TransitionSpec],
    protocol: str,
) -> None:
    """在 lines 中追加 guard 字段覆盖分析。"""
    # 学习到的 guard 字段（语义化）
    learned_fields: Set[str] = set()
    single = guard_desc.get("single_constraints", {})
    aliases = PROTOCOL_VAR_ALIASES.get(protocol, {})
    for var_name in single:
        sem = aliases.get(var_name, var_name)
        if isinstance(sem, list):
            learned_fields.update(sem)
        else:
            learned_fields.add(sem)
    for _, _, cons_vars, _, _ in guard_desc.get("joint_rules", []):
        for n in cons_vars:
            sem = aliases.get(n, n)
            if isinstance(sem, list):
                learned_fields.update(sem)
            else:
                learned_fields.add(sem)
    for a, b, _, _, _ in guard_desc.get("linear_pairs", []):
        for n in (a, b):
            sem = aliases.get(n, n)
            if isinstance(sem, list):
                learned_fields.update(sem)
            else:
                learned_fields.add(sem)

    # GT guard 字段
    gt_fields: Set[str] = set()
    for ts in gt_transitions:
        gt_fields.update(ts.guard.field_names)

    tp = learned_fields & gt_fields
    fp = learned_fields - gt_fields
    fn = gt_fields - learned_fields

    def _fmt_set(s: Set[str]) -> str:
        return "{" + ", ".join(sorted(s)) + "}" if s else "(none)"

    lines.append(f"    GT guard 字段: {_fmt_set(gt_fields)}")
    lines.append(f"    学习字段:      {_fmt_set(learned_fields)}")
    if gt_fields:
        prec = len(tp) / len(learned_fields) if learned_fields else 0.0
        rec  = len(tp) / len(gt_fields)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(
            f"    TP={_fmt_set(tp)}  FP={_fmt_set(fp)}  FN={_fmt_set(fn)}"
        )
        lines.append(f"    Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


# ---------------------------------------------------------------------------
# GT 总览
# ---------------------------------------------------------------------------

def build_gt_overview(gt: Optional[ProtocolGT]) -> str:
    """生成 GT 规范总览。"""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("  协议真值规范（GT）总览")
    lines.append("=" * 70)

    if gt is None:
        lines.append("  (无 GT 规范)")
        return "\n".join(lines)

    lines.append(f"  协议: {gt.protocol}")
    lines.append(f"  转移规则数: {len(gt.transitions)}")
    lines.append(f"  Guard 字段集合: {sorted(gt.guard_fields)}")
    lines.append(f"  Action 变量集合: {sorted(gt.action_vars)}")

    lines.append("\n  转移规则列表:")
    for i, ts in enumerate(gt.transitions, 1):
        src_str = "{" + ", ".join(sorted(ts.src_types)[:3]) + ("..." if len(ts.src_types) > 3 else "") + "}"
        dst_str = "{" + ", ".join(sorted(ts.dst_types)[:3]) + ("..." if len(ts.dst_types) > 3 else "") + "}"
        lines.append(f"  [{i:2d}] {ts.label}")
        lines.append(f"       src_types: {src_str}")
        lines.append(f"       dst_types: {dst_str}")
        lines.append(f"       guard: {ts.guard.describe()}")
        if ts.action.description:
            lines.append(f"       action: {ts.action.description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def inspect_and_report(
    protocol: str,
    data_dir: str,
    pcap_paths: Optional[List[str]] = None,
    seed: int = 42,
    test_ratio: float = 0.2,
    max_sessions: int = 200,
    max_pcaps: int = 6,
) -> str:
    """
    完整检视流程：
      1. 加载 PCAP → 推断 EFSM
      2. 获取 GT 规范
      3. 生成字段划分 + guard/action 对比报告
    """
    proto_upper = protocol.upper()

    # ---- 1. 收集 PCAP ----
    if pcap_paths is None:
        pcap_paths = []
        if os.path.isdir(data_dir):
            for fn in sorted(os.listdir(data_dir)):
                if fn.lower().endswith(".pcap"):
                    full = os.path.join(data_dir, fn)
                    if _looks_like_capture(full):
                        pcap_paths.append(full)
                    if len(pcap_paths) >= max_pcaps:
                        break
    if not pcap_paths:
        return f"[ERROR] 在 '{data_dir}' 下未找到有效 PCAP 文件。"

    # ---- 2. 解析事件 ----
    pipe = PCAPPipeline()
    events: List[MessageEvent] = []
    for p in pcap_paths:
        try:
            t = pipe.run(p)
            events.extend(t.events)
        except Exception as exc:
            print(f"[WARN] 解析 {p} 失败: {exc}", file=sys.stderr)
    if not events:
        return "[ERROR] 所有 PCAP 文件解析失败，无可用事件。"
    events.sort(key=lambda e: e.timestamp)
    trace = Trace(events=events)

    # ---- 3. 会话分组 & 切分 ----
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
        train_sessions = {only_key: evs[:split_at]}
    else:
        train_keys, _ = _split_keys(list(sessions_all.keys()), test_ratio, seed)
        train_sessions = {k: sessions_all[k] for k in train_keys}

    train_trace = Trace(events=[ev for evs in train_sessions.values() for ev in evs])

    # ---- 4. 推断 FSM + EFSM ----
    print(f"[INFO] 正在推断 FSM/EFSM（训练会话数={len(train_sessions)}）...", file=sys.stderr)
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

    # ---- 5. 获取 GT ----
    gt = get_gt(proto_upper)

    # ---- 6. 组装报告 ----
    sections: List[str] = []

    # 摘要
    sections.append("=" * 70)
    sections.append(f"  EFSM 检视报告 — 协议: {proto_upper}")
    sections.append("=" * 70)
    sections.append(
        f"  训练会话: {len(train_sessions)}  "
        f"| PCAP 文件: {len(pcap_paths)}  "
        f"| FSM 状态数: {len(efsm.states)}  "
        f"| 转移数: {len(efsm.transitions)}"
    )
    sections.append(
        f"  学习到的变量: {sorted(efsm.variable_defs)}"
    )

    # GT 总览
    sections.append("\n" + build_gt_overview(gt))

    # 字段划分对比
    fp = df.feature_processor
    sections.append("\n" + build_field_report(fp, proto_upper, gt))

    # Guard/Action 对比
    sections.append("\n" + build_guard_action_report(efsm, proto_upper, gt))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="检视 EFSM 学习结果并与协议 GT 对比。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--protocol", "-P", required=True,
                   help="协议名 (MODBUS / IEC104 / DNP3)")
    p.add_argument("--data-dir", "-d", required=True,
                   help="PCAP 文件目录")
    p.add_argument("--max-pcaps",    type=int,   default=6)
    p.add_argument("--max-sessions", type=int,   default=200)
    p.add_argument("--test-ratio",   type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output", "-o", default=None,
                   help="可选：报告输出路径（默认打印到 stdout）")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    report = inspect_and_report(
        protocol=args.protocol,
        data_dir=args.data_dir,
        seed=args.seed,
        test_ratio=args.test_ratio,
        max_sessions=args.max_sessions,
        max_pcaps=args.max_pcaps,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已写入: {args.output}", file=sys.stderr)
    else:
        # 在 Windows 控制台下强制使用 UTF-8 输出，避免 GBK 编码错误
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)


if __name__ == "__main__":
    main()
