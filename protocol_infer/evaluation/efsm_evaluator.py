"""
模块三：评估器

将推断的 EFSM 与 GT（ProtocolGT）对比，输出四个维度的量化指标：

1. guard_prf        Guard 精确率/召回率/F1（字段级对齐匹配）
2. guard_violation  Guard 违规率（用实际流量验证推断约束是否被实际数据满足）
3. action_coverage  Action 副作用覆盖率（推断变量 vs 规范变量的交并比）
4. state_diff_acc   State diff 准确率（推断 action 预测的变量变化 vs 实际观测变化）

依赖
----
- protocol_infer.core.model.efsm.EFSM
- protocol_infer.evaluation.gt_guard_action.ProtocolGT, GuardSpec
- protocol_infer.evaluation.field_extractor.SessionTrace, TraceRecord
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from protocol_infer.core.model.efsm import EFSM, MemoryContext
from protocol_infer.core.model.fsm import Transition
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.evaluation.base import _safe_div, _mean
from protocol_infer.evaluation.gt_guard_action import (
    ProtocolGT,
    GuardSpec,
    ActionSpec,
    TransitionSpec,
    FieldConstraint,
    normalize_var_names,
)
from protocol_infer.evaluation.field_extractor import SessionTrace, TraceRecord


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class GuardPRFResult:
    """Guard 精确率/召回率/F1。"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "guard_precision": self.precision,
            "guard_recall": self.recall,
            "guard_f1": self.f1,
            "guard_tp": float(self.tp),
            "guard_fp": float(self.fp),
            "guard_fn": float(self.fn),
        }


@dataclass
class GuardViolationResult:
    """Guard 违规率：推断 guard 约束在实际流量中被违反的比例。"""
    total_checks: int = 0
    violations: int = 0

    @property
    def violation_rate(self) -> float:
        return _safe_div(self.violations, self.total_checks)

    def to_dict(self) -> Dict[str, float]:
        return {
            "guard_violation_rate": self.violation_rate,
            "guard_violation_count": float(self.violations),
            "guard_check_count": float(self.total_checks),
        }


@dataclass
class ActionCoverageResult:
    """Action 副作用覆盖率：推断变量集与 GT 变量集的 Jaccard 相似度。"""
    inferred_vars: Set[str] = field(default_factory=set)
    gt_vars: Set[str] = field(default_factory=set)

    @property
    def intersection(self) -> Set[str]:
        return self.inferred_vars & self.gt_vars

    @property
    def union(self) -> Set[str]:
        return self.inferred_vars | self.gt_vars

    @property
    def jaccard(self) -> float:
        return _safe_div(len(self.intersection), len(self.union))

    @property
    def precision(self) -> float:
        return _safe_div(len(self.intersection), len(self.inferred_vars))

    @property
    def recall(self) -> float:
        return _safe_div(len(self.intersection), len(self.gt_vars))

    def to_dict(self) -> Dict[str, float]:
        return {
            "action_coverage_jaccard": self.jaccard,
            "action_coverage_precision": self.precision,
            "action_coverage_recall": self.recall,
            "action_inferred_var_count": float(len(self.inferred_vars)),
            "action_gt_var_count": float(len(self.gt_vars)),
        }


@dataclass
class StateDiffAccResult:
    """
    State diff 准确率：
      推断 action 预测的变量变化（pred_diff）与实际观测变化（obs_diff）的吻合度。

    对每条转移逐字段比较：
      - 若 GT action 规定变量 v 应变化，且 pred_diff/obs_diff 一致 -> 正确
      - 若不一致 -> 错误
    """
    total_var_checks: int = 0
    correct_var_checks: int = 0

    @property
    def accuracy(self) -> float:
        return _safe_div(self.correct_var_checks, self.total_var_checks)

    def to_dict(self) -> Dict[str, float]:
        return {
            "state_diff_accuracy": self.accuracy,
            "state_diff_total": float(self.total_var_checks),
            "state_diff_correct": float(self.correct_var_checks),
        }


@dataclass
class EFSMEvalResult:
    """所有四个维度的汇总结果。"""
    protocol: str
    guard_prf: GuardPRFResult = field(default_factory=GuardPRFResult)
    guard_violation: GuardViolationResult = field(default_factory=GuardViolationResult)
    action_coverage: ActionCoverageResult = field(default_factory=ActionCoverageResult)
    state_diff_acc: StateDiffAccResult = field(default_factory=StateDiffAccResult)

    def to_dict(self) -> Dict[str, float]:
        d: Dict[str, float] = {}
        d.update(self.guard_prf.to_dict())
        d.update(self.guard_violation.to_dict())
        d.update(self.action_coverage.to_dict())
        d.update(self.state_diff_acc.to_dict())
        return d


# ---------------------------------------------------------------------------
# 辅助：从 EFSM 转移中提取 guard/action 涉及的字段名
# ---------------------------------------------------------------------------

# guard/action 函数体中的通用局部变量名，不代表协议字段，排除
_GENERIC_FUNC_NAMES: Set[str] = frozenset({
    "vars", "vars_dict", "mem", "mem_data", "result", "new_vars",
    "name", "val", "ctype", "vmin", "vmax", "constraint",
    "ante_vars", "ante_vals", "cons_vars", "cons_vals",
    "ante_ok", "cons_ok", "n", "v", "k", "i", "j",
    "atype", "param", "changes", "avg_delta", "action_rules",
    "single_constraints", "joint_rules", "joint_rule_means",
})


def _extract_fields_from_closure(fn: Callable, _depth: int = 0) -> Set[str]:
    """
    从函数闭包和默认参数中递归提取协议变量名。

    AprioriGuardLearner / ActionLearner 把约束/action 规则以 dict 形式存在闭包里：
      guard:  single_constraints = {var_name: (ctype, val, ...)}
              joint_rules = [(ante_vars, ante_vals, cons_vars, cons_vals, conf), ...]
      action: action_rules = {var_name: (atype, param)}

    EFSMInferencer 会用默认参数把原始 guard/action 包进 wrapped_guard/wrapped_action，
    因此需要通过 __defaults__ 递归到内层函数。
    """
    if _depth > 3 or not callable(fn):
        return set()
    fields: Set[str] = set()

    # 1. 检查闭包 cells
    closure = getattr(fn, "__closure__", None)
    if closure:
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if isinstance(val, dict):
                for k, v in val.items():
                    if not (isinstance(k, str) and k and not k.startswith("_") and k not in _GENERIC_FUNC_NAMES):
                        continue
                    # 跳过类型注解字典（vtype_map）：其值为纯字符串如 'constant'/'discrete'
                    # 只处理约束字典（single_constraints）：其值为元组如 ('in', frozenset({...}))
                    if isinstance(v, str):
                        continue
                    # action_rules 中 "keep" 表示变量不变，不计为 action 副作用变量
                    if isinstance(v, tuple) and len(v) >= 1 and v[0] == "keep":
                        continue
                    fields.add(k)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        for sublist in (item[0], item[2]):  # ante_vars, cons_vars
                            if isinstance(sublist, (list, tuple)):
                                for name in sublist:
                                    if isinstance(name, str) and not name.startswith("_") and name not in _GENERIC_FUNC_NAMES:
                                        fields.add(name)
            elif callable(val):
                # 递归进嵌套函数
                fields.update(_extract_fields_from_closure(val, _depth + 1))

    # 2. 检查默认参数（wrapped_guard 把 base_guard 作为默认参数传递）
    defaults = getattr(fn, "__defaults__", None) or ()
    for d in defaults:
        if callable(d):
            fields.update(_extract_fields_from_closure(d, _depth + 1))

    return fields


def _extract_inferred_guard_fields(efsm: EFSM) -> Set[str]:
    """从 EFSM 转移的 guard 函数中提取涉及的字段名（通过闭包检查）。"""
    guard_fields: Set[str] = set()
    for tran in efsm.transitions:
        if tran.guard is None:
            continue
        guard_fields.update(_extract_fields_from_closure(tran.guard))
    return guard_fields


def _extract_inferred_action_vars(efsm: EFSM) -> Set[str]:
    """从 EFSM 转移的 action 函数中提取涉及的变量名（通过闭包检查）。"""
    action_vars: Set[str] = set()
    for tran in efsm.transitions:
        if tran.action is None:
            continue
        action_vars.update(_extract_fields_from_closure(tran.action))
    return action_vars


# ---------------------------------------------------------------------------
# 核心评估器
# ---------------------------------------------------------------------------

class GuardFieldEvaluator:
    """
    维度一：Guard 精确率/召回率/F1（字段级对齐匹配）。

    方法：
    - GT 规范中出现的字段名 = 正例集合
    - 推断 EFSM guard 中出现的字段名（归一化为语义名）= 预测集合
    - TP = 两者交集；FP = 预测 - GT；FN = GT - 预测
    """

    def evaluate(self, efsm: EFSM, gt: ProtocolGT) -> GuardPRFResult:
        gt_fields = gt.guard_fields
        raw_inferred = _extract_inferred_guard_fields(efsm)
        # 将字节位名（b7, b6, ...）归一化为语义字段名（fc, unit_id, ...）
        inferred_fields = normalize_var_names(raw_inferred, gt.protocol)

        tp_set = inferred_fields & gt_fields
        fp_set = inferred_fields - gt_fields
        fn_set = gt_fields - inferred_fields

        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)

        return GuardPRFResult(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


class GuardViolationEvaluator:
    """
    维度二：Guard 违规率。

    对实际流量中的每条转移，检查推断 EFSM 的 guard 是否被当前报文字段满足。
    违规 = guard 存在但被实际数据违反（guard(fields) == False）。

    使用 rec.abstract_sym 做 EFSM 状态跟踪，guard 用字节位变量（abstract_vars）求值。
    """

    def evaluate(
        self,
        efsm: EFSM,
        session_traces: Dict[SessionKey, SessionTrace],
    ) -> GuardViolationResult:
        result = GuardViolationResult()

        for st in session_traces.values():
            cur = efsm.start_state
            mem = MemoryContext()
            records = list(st.records)

            for rec in records:
                # 优先用抽象 symbol 查找 EFSM 转移；回退到 msg_type
                sym = rec.abstract_sym if rec.abstract_sym is not None else rec.msg_type
                # guard 函数使用字节位变量名（b7, b6, ...）
                vars_for_guard = rec.abstract_vars if rec.abstract_vars is not None else {}
                if not vars_for_guard:
                    vars_for_guard = {k: float(v) for k, v in rec.fields.items()
                                      if isinstance(v, (int, float)) and k != "direction"}

                cands = efsm._by_state_input.get((cur, sym), [])
                if not cands:
                    continue

                for tran in cands:
                    if tran.guard is None:
                        continue
                    result.total_checks += 1
                    try:
                        ok = tran.guard(vars_for_guard, mem.data)
                    except TypeError:
                        try:
                            ok = tran.guard(vars_for_guard)
                        except Exception:
                            ok = True
                    except Exception:
                        ok = True
                    if not ok:
                        result.violations += 1

                # 步进：用字节位变量跟踪状态
                nxt, _ = efsm.step_with_memory(cur, sym, vars_for_guard, mem)
                if nxt is not None:
                    cur = nxt

        return result


class ActionCoverageEvaluator:
    """
    维度三：Action 副作用覆盖率（推断变量 vs GT 规范变量的 Jaccard）。
    推断变量名通过别名表归一化为语义字段名后再比较。
    """

    def evaluate(self, efsm: EFSM, gt: ProtocolGT) -> ActionCoverageResult:
        raw_inferred = _extract_inferred_action_vars(efsm)
        inferred_vars = normalize_var_names(raw_inferred, gt.protocol)
        gt_vars = gt.action_vars
        return ActionCoverageResult(inferred_vars=inferred_vars, gt_vars=gt_vars)


class StateDiffAccEvaluator:
    """
    维度四：State diff 准确率。

    对每对相邻记录 (rec_i, rec_{i+1})：
      obs_diff  = rec_{i+1}.state_diff（实际观测到的变量变化）
      pred_diff = efsm action(fields_i) 预测的变量变化

    对 GT action 规定的每个 changed_var v：
      - obs_changed  = (obs_diff 中 v 发生了变化)
      - pred_changed = (pred_vars[v] != fields_i[v])
      - correct      = (obs_changed == pred_changed)
    """

    def evaluate(
        self,
        efsm: EFSM,
        gt: ProtocolGT,
        session_traces: Dict[SessionKey, SessionTrace],
    ) -> StateDiffAccResult:
        result = StateDiffAccResult()
        # 协议字节位变量 → 语义字段名 别名表（用于 pred_vars key 归一化）
        from protocol_infer.evaluation.gt_guard_action import PROTOCOL_VAR_ALIASES
        aliases = PROTOCOL_VAR_ALIASES.get(gt.protocol.upper(), {})

        for st in session_traces.values():
            cur = efsm.start_state
            records = list(st.records)

            for i in range(len(records) - 1):
                rec_i = records[i]
                rec_next = records[i + 1]

                # EFSM 状态跟踪使用 abstract_sym；回退到 msg_type
                sym = rec_i.abstract_sym if rec_i.abstract_sym is not None else rec_i.msg_type
                # guard/action 使用字节位变量
                abstract_vars_i = rec_i.abstract_vars if rec_i.abstract_vars is not None else {}
                if not abstract_vars_i:
                    abstract_vars_i = {k: float(v) for k, v in rec_i.fields.items()
                                       if isinstance(v, (int, float)) and k != "direction"}

                cands = efsm._by_state_input.get((cur, sym), [])
                if not cands:
                    continue
                tran = cands[0]

                # 推断 action 预测的变量值（字节位名）
                if tran.action is not None:
                    try:
                        pred_vars_raw = tran.action(abstract_vars_i.copy())
                    except Exception:
                        pred_vars_raw = abstract_vars_i.copy()
                else:
                    pred_vars_raw = abstract_vars_i.copy()

                # 将 pred_vars 的 key 归一化为语义字段名（别名值可能是 list，取第一个）
                def _first_alias(aliases, k):
                    mapped = aliases.get(k, k)
                    return mapped[0] if isinstance(mapped, list) else mapped

                pred_vars = {_first_alias(aliases, k): v for k, v in pred_vars_raw.items()}
                vars_i_semantic = {_first_alias(aliases, k): v for k, v in abstract_vars_i.items()}

                # 实际观测的变量变化（state_diff，来自语义 FieldExtractor）
                obs_diff = rec_next.state_diff

                # GT 匹配用语义 msg_type（fc_01_c2s → fc_01_s2c）
                src_sem = rec_i.msg_type
                dst_sem = rec_next.msg_type
                matching_gt = gt.find_matching_transitions(src_sem, dst_sem)

                for gt_ts in matching_gt:
                    for var in gt_ts.action.changed_vars:
                        result.total_var_checks += 1
                        obs_changed = var in obs_diff
                        pred_changed = (pred_vars.get(var) != vars_i_semantic.get(var))
                        if obs_changed == pred_changed:
                            result.correct_var_checks += 1

                # 步进
                cur = tran.dst

        return result


# ---------------------------------------------------------------------------
# 组合评估器
# ---------------------------------------------------------------------------

class EnhancedEFSMEvaluator:
    """
    组合四个维度的评估器，对外提供统一接口。

    用法
    ----
    evaluator = EnhancedEFSMEvaluator()
    result = evaluator.evaluate(
        efsm=efsm,
        gt=gt,
        session_traces=session_traces,
    )
    print(result.to_dict())
    """

    def __init__(self):
        self._guard_prf = GuardFieldEvaluator()
        self._guard_viol = GuardViolationEvaluator()
        self._action_cov = ActionCoverageEvaluator()
        self._state_diff = StateDiffAccEvaluator()

    def evaluate(
        self,
        efsm: EFSM,
        gt: ProtocolGT,
        session_traces: Dict[SessionKey, SessionTrace],
    ) -> EFSMEvalResult:
        guard_prf = self._guard_prf.evaluate(efsm, gt)
        guard_viol = self._guard_viol.evaluate(efsm, session_traces)
        action_cov = self._action_cov.evaluate(efsm, gt)
        state_diff = self._state_diff.evaluate(efsm, gt, session_traces)

        return EFSMEvalResult(
            protocol=gt.protocol,
            guard_prf=guard_prf,
            guard_violation=guard_viol,
            action_coverage=action_cov,
            state_diff_acc=state_diff,
        )
