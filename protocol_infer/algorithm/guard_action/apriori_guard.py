from typing import Dict, List, Tuple, Optional, Callable, FrozenSet

from protocol_infer.apriori.core import AprioriCore
from protocol_infer.core.algorithm.guard_action import GuardActionLearner


class LinearRelationDetector:
    def __init__(
        self,
        r2_threshold: float = 0.999,
        residual_tol: float = 1e-4,
        min_samples: int = 4,
    ):
        self.r2_threshold = r2_threshold
        self.residual_tol = residual_tol
        self.min_samples = min_samples

    def detect_pairwise(
        self,
        var_instances: List[Dict[str, float]],
        continuous_vars: List[str],
    ) -> List[Tuple[str, str, float, float, float]]:
        '''
            检测 a = k·b + c
            最小二乘法
        '''
        results: List[Tuple[str, str, float, float, float]] = []
        n = len(continuous_vars)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a_name = continuous_vars[i]
                b_name = continuous_vars[j]

                a_vals = [
                    inst[a_name]
                    for inst in var_instances
                    if a_name in inst and b_name in inst
                ]
                b_vals = [
                    inst[b_name]
                    for inst in var_instances
                    if a_name in inst and b_name in inst
                ]

                if len(a_vals) < self.min_samples:
                    continue

                k, c, r2 = self._fit_linear(a_vals, b_vals)

                if abs(k) < 1e-9:
                    continue

                if r2 >= self.r2_threshold:
                    results.append((a_name, b_name, k, c, r2))

        results.sort(key=lambda x: -x[4])
        return self._deduplicate(results)       # 去除双向冗余

    def detect_triplet_sum(
        self,
        var_instances: List[Dict[str, float]],
        continuous_vars: List[str],
    ) -> List[Tuple[str, str, str, float]]:
        '''
            检测 a + b = c
            适配工业协议中常见的字段拆分场景，例如 high_byte + low_byte = total
        '''
        results: List[Tuple[str, str, str, float]] = []
        n = len(continuous_vars)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k == i or k == j:
                        continue
                    a_name = continuous_vars[i]
                    b_name = continuous_vars[j]
                    c_name = continuous_vars[k]

                    triples = [
                        (inst[a_name], inst[b_name], inst[c_name])
                        for inst in var_instances
                        if a_name in inst and b_name in inst and c_name in inst
                    ]

                    if len(triples) < self.min_samples:
                        continue

                    residuals = [abs(a + b - c) for a, b, c in triples]
                    max_res = max(residuals)

                    if max_res <= self.residual_tol:
                        results.append((a_name, b_name, c_name, max_res))

        return results

    def _fit_linear(
        self,
        a_vals: List[float],
        b_vals: List[float],
    ) -> Tuple[float, float, float]:
        n = len(a_vals)
        sum_b = sum(b_vals)
        sum_a = sum(a_vals)
        sum_bb = sum(x * x for x in b_vals)
        sum_ab = sum(a * b for a, b in zip(a_vals, b_vals))

        denom = n * sum_bb - sum_b * sum_b
        if abs(denom) < 1e-12:
            return 0.0, sum_a / n, 0.0

        k = (n * sum_ab - sum_b * sum_a) / denom
        c = (sum_a - k * sum_b) / n

        a_mean = sum_a / n
        ss_tot = sum((a - a_mean) ** 2 for a in a_vals)
        ss_res = sum((a - (k * b + c)) ** 2 for a, b in zip(a_vals, b_vals))

        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        return k, c, r2

    def _deduplicate(
        self,
        results: List[Tuple[str, str, float, float, float]],
    ) -> List[Tuple[str, str, float, float, float]]:
        '''
            去除双向冗余: a=f(b) 和 b=f(a)
        '''
        seen_pairs = set()
        deduped = []
        for item in results:
            pair = frozenset([item[0], item[1]])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                deduped.append(item)
        return deduped


class AprioriGuardLearner(GuardActionLearner):
    def __init__(
        self,
        discrete_threshold: int = 8,
        min_support: float = 0.6,
        min_confidence: float = 0.8,
        delta_tolerance: float = 1e-6,
        continuous_tolerance: float = 0.05,
        linear_r2_threshold: float = 0.999,
        linear_residual_tol: float = 1e-4,
        enable_triplet_sum: bool = True,
    ):
        self.discrete_threshold = discrete_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.delta_tolerance = delta_tolerance
        self.continuous_tolerance = continuous_tolerance
        self.linear_r2_threshold = linear_r2_threshold
        self.linear_residual_tol = linear_residual_tol
        self.enable_triplet_sum = enable_triplet_sum

        self.core = AprioriCore()
        self.linear_detector = LinearRelationDetector(
            r2_threshold=linear_r2_threshold,
            residual_tol=linear_residual_tol,
        )

    def learn(
        self,
        var_instances: List[Dict[str, float]],
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        if not var_instances:
            return None, None

        var_names = list(var_instances[0].keys())
        var_types = {
            name: self._infer_var_type([v[name] for v in var_instances if name in v])
            for name in var_names
        }

        guard = self._learn_guard(var_instances, var_names, var_types)
        action = self._learn_action(var_instances, var_names, var_types)
        return guard, action

    def _infer_var_type(self, values: List[float]) -> str:
        '''
            变量类型推断   
        '''
        unique = set(values)
        if len(unique) == 1:
            return "constant"
        if len(unique) <= self.discrete_threshold:
            return "discrete"
        if len(values) > 1:
            deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            if deltas and max(abs(d - deltas[0]) for d in deltas) < self.delta_tolerance:
                return "sequential"
        return "continuous"

    def _learn_guard(
        self,
        var_instances: List[Dict[str, float]],
        var_names: List[str],
        var_types: Dict[str, str],
    ) -> Optional[Callable]:

        # 1. 单变量约束(为每种类型生成对应的约束形式)
        # eg: constant -> ("eq",    3.0)    ==>   fc == 3
        single_constraints: Dict[str, tuple] = {}
        for name in var_names:
            values = [v[name] for v in var_instances if name in v]
            if not values:
                continue
            vtype = var_types[name]
            if vtype == "constant":
                single_constraints[name] = ("eq", values[0])
            elif vtype == "discrete":
                single_constraints[name] = ("in", frozenset(values))
            elif vtype == "sequential":
                delta = values[1] - values[0] if len(values) > 1 else 0.0
                single_constraints[name] = ("delta", delta)
            else:
                span = max(values) - min(values)
                tol = span * self.continuous_tolerance
                single_constraints[name] = (
                    "range",
                    (min(values) - tol, max(values) + tol),
                )

        # 2. Apriori联合规则(无具体值, 只有变量之间的依赖以及其支持度)
        joint_rules, joint_rule_means = self._mine_joint_rules(
            var_instances, var_names, var_types
        )


        # 3. 连续变量线性关系
        continuous_vars = [n for n in var_names if var_types[n] == "continuous"]
        linear_pairs: List[Tuple[str, str, float, float, float]] = []
        triplet_sums: List[Tuple[str, str, str, float]] = []
        if len(continuous_vars) >= 2:
            linear_pairs = self.linear_detector.detect_pairwise(
                var_instances, continuous_vars
            )
            if self.enable_triplet_sum and len(continuous_vars) >= 3:
                triplet_sums = self.linear_detector.detect_triplet_sum(
                    var_instances, continuous_vars
                )

        def guard(vars: Dict[str, float]) -> bool:
            # guard检查流程
            # 1. 单变量约束检查
            for name, constraint in single_constraints.items():
                if name not in vars:
                    return False
                val = vars[name]
                ctype = constraint[0]
                if ctype == "eq":
                    if val != constraint[1]:
                        return False
                elif ctype == "in":
                    if val not in constraint[1]:
                        return False
                elif ctype == "range":
                    vmin, vmax = constraint[1]
                    if not (vmin <= val <= vmax):
                        return False

            # 2. 关联规则验证(A->B)
            for (
                ante_vars,
                ante_vals,
                cons_vars,
                cons_vals,
                _confidence,
            ) in joint_rules:
                ante_ok = all(      # 前(A)是否满足    
                    name in vars
                    and self._check_joint_binding(
                        name,
                        vars[name],
                        val,
                        var_types,
                        joint_rule_means,
                    )
                    for name, val in zip(ante_vars, ante_vals)  # 将变量名和期望值配对遍历
                )
                if ante_ok:         # 如果前变量满足
                    cons_ok = all(
                        name in vars
                        and self._check_joint_binding(
                            name,
                            vars[name],
                            val,
                            var_types,
                            joint_rule_means,
                        )
                        for name, val in zip(cons_vars, cons_vals)
                    )
                    if not cons_ok:
                        return False

            # 3. 连续变量线性关系
            for a_name, b_name, k, c, _r2 in linear_pairs:      
                if a_name not in vars or b_name not in vars:
                    continue
                expected_a = k * vars[b_name] + c
                abs_tol = max(self.linear_residual_tol, abs(expected_a) * 1e-3)     # 容差值, 对于大的期望值, 容差相应增大
                if abs(vars[a_name] - expected_a) > abs_tol:
                    return False

            for a_name, b_name, c_name, _max_res in triplet_sums:
                if a_name not in vars or b_name not in vars or c_name not in vars:
                    continue
                residual = abs(vars[a_name] + vars[b_name] - vars[c_name])
                if residual > self.linear_residual_tol * 10:                # 比线性约束宽松10倍
                    return False

            return True

        return guard

    def _check_single(self, val: float, constraint) -> bool:
        if constraint is None:
            return True
        ctype = constraint[0]
        if ctype == "eq":
            return val == constraint[1]
        if ctype == "in":
            return val in constraint[1]
        if ctype == "range":
            vmin, vmax = constraint[1]
            return vmin <= val <= vmax
        return True

    def _check_joint_binding(
        self,
        name: str,
        actual_val: float,
        expected_val: float,
        var_types: Dict[str, str],
        joint_rule_means: Dict[str, float],
    ) -> bool:
        vtype = var_types.get(name)
        if vtype == "continuous":
            mean = joint_rule_means.get(name)
            if mean is None:
                return True
            if expected_val >= 0.5:         # 高于均值
                return actual_val >= mean   # 实际值也得高于均值
            return actual_val < mean
        if vtype in ("constant", "discrete"):
            return actual_val == expected_val
        return True

    def _mine_joint_rules(
        self,
        var_instances: List[Dict[str, float]],
        var_names: List[str],
        var_types: Dict[str, str],
    ) -> Tuple[
        List[Tuple[List[str], List[float], List[str], List[float], float]],
        Dict[str, float],
    ]:
        '''
            发现字段间的条件依赖
        '''
        transactions: List[FrozenSet[Tuple[str, float]]] = []
        cached_means: Dict[str, float] = {}     # 保存continuous类型的均值


        # 首先计算连续类型的变量的均值
        for name in var_names:
            if var_types.get(name) == "continuous":
                all_vals = [v[name] for v in var_instances if name in v]
                if all_vals:
                    cached_means[name] = sum(all_vals) / len(all_vals)

        for inst in var_instances:
            items = set()       # 不考虑sequential类型
            for name in var_names:
                if name not in inst:
                    continue
                vtype = var_types[name]
                if vtype in ("constant", "discrete"):
                    items.add((name, inst[name]))       # 直接用原值
                elif vtype == "continuous":
                    mean = cached_means.get(name)       # 用预先计算的均值
                    if mean is None:
                        continue
                    items.add((name, 1.0 if inst[name] >= mean else 0.0))       # 均值二值化(高于均值为1)这样减少值域数量可以挖掘规则
            transactions.append(frozenset(items))

        if not transactions:
            return [], cached_means

        # 使用Apriori算法挖掘频繁项集合关联规则
        fis = self.core.frequent_itemsets(transactions, self.min_support)       # 频繁项集（support >= min_support）
        rules = self.core.association_rules(fis, self.min_confidence)           # 关联规则（confidence >= min_confidence）

        result: List[Tuple[List[str], List[float], List[str], List[float], float]] = []
        for ante, cons, _sup, conf in rules:
            ante_items = sorted(list(ante), key=lambda x: x[0])
            cons_items = sorted(list(cons), key=lambda x: x[0])

            ante_vars = [name for name, _val in ante_items]
            ante_vals = [_val for _name, _val in ante_items]
            cons_vars = [name for name, _val in cons_items]
            cons_vals = [_val for _name, _val in cons_items]

            if ante_vars and cons_vars:
                result.append((ante_vars, ante_vals, cons_vars, cons_vals, conf))

        result.sort(key=lambda x: -x[4])                        # 置信度降序排列
        return result, cached_means

    def _learn_action(
        self,
        var_instances: List[Dict[str, float]],
        var_names: List[str],
        var_types: Dict[str, str],
    ) -> Optional[Callable]:
        if len(var_instances) < 2:
            return lambda vars: vars.copy()

        action_rules = {}

        for name in var_names:
            values = [v[name] for v in var_instances if name in v]
            if not values:
                continue
            vtype = var_types[name]

            if vtype == "constant":
                action_rules[name] = ("keep", None)         # 保持不变
            elif vtype == "sequential":
                delta = values[1] - values[0] if len(values) > 1 else 0.0
                action_rules[name] = ("delta", delta)       # 按步长更新
            elif vtype == "discrete":
                action_rules[name] = ("keep", None)         # 离散的不更新
            else:                                           # continuous: 平均增量较小, 则保持不变, 否则按步长更新
                changes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
                avg_delta = sum(changes) / len(changes) if changes else 0.0
                if abs(avg_delta) < self.delta_tolerance:
                    action_rules[name] = ("keep", None)
                else:
                    action_rules[name] = ("delta", avg_delta)

        def action(vars: Dict[str, float]) -> Dict[str, float]:
            new_vars = vars.copy()
            for name, (atype, param) in action_rules.items():
                if name not in new_vars:
                    continue
                if atype == "delta":
                    new_vars[name] += param
            return new_vars

        return action
