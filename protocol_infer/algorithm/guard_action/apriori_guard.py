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

    # 不加入 guard 约束的衍生特征（非协议语义字段）
    # direction: 对给定转移始终恒定（C2S 或 S2C），约束无意义
    # entropy:   反映 payload 字节多样性，非协议规范字段，学习后导致过拟合
    # len:       包总字节数，是 payload 的衍生属性，由 FC 类型决定，非 guard 语义字段
    _GUARD_SKIP_VARS: frozenset = frozenset({"direction", "entropy", "len"})

    # 不加入 guard 约束的变量名前缀
    # s-前缀（如 s0, s2, s6）：Apriori 静态项，对所有消息均为同一常量值，
    #   无转移判别力（不能区分本转移与其他转移），加入 guard 只增加误判风险。
    # dyn_前缀（如 dyn_1, dyn_0_2b）：DynamicFieldDetector 检测的结构化字段，
    #   多为计数器/序列号（txn_id 等），其值在训练数据中的取值范围不代表协议约束，
    #   不应作为 guard 条件（否则新会话的 txn_id 值域不同会导致误拒）。
    _GUARD_SKIP_PREFIX: tuple = ("s", "dyn_")

    def _learn_guard(
        self,
        var_instances: List[Dict[str, float]],
        var_names: List[str],
        var_types: Dict[str, str],
    ) -> Optional[Callable]:

        # 1. 单变量约束(为每种类型生成对应的约束形式)
        # eg: constant -> ("eq",    3.0)    ==>   fc == 3
        # 当样本数不足 MIN_CONFIDENCE 时，不生成严格约束，
        # 避免在小样本下过拟合（如 fc=0x2b 的 Object ID 字节、小会话的 FC 值等）。
        # discrete 类型：少量训练样本（如 C6 仅有 FC=0x16/0x17 各 1 次 = 4 包）
        # 容易被不同 FC 聚类到同一 cluster 后违反，需同等置信度保护。
        MIN_CONFIDENCE = 6
        single_constraints: Dict[str, tuple] = {}
        for name in var_names:
            if name in self._GUARD_SKIP_VARS:
                continue    # 跳过非协议语义的衍生特征
            if any(name.startswith(pfx) for pfx in self._GUARD_SKIP_PREFIX):
                continue    # 跳过静态全局常量字节（s-前缀），无转移判别力
            values = [v[name] for v in var_instances if name in v]
            if not values:
                continue
            vtype = var_types[name]
            if vtype == "constant":
                if len(values) >= MIN_CONFIDENCE:
                    # b-前缀动态字段若唯一值为 0.0，大概率是多字节字段高字节（采样局限）
                    # 跳过：避免对"quantity高字节=0"此类低信息量字段产生过约束
                    if name.startswith("b") and values[0] == 0.0:
                        pass
                    else:
                        single_constraints[name] = ("eq", values[0])
                # 否则跳过：样本不足，不能判定为真正的常量约束
            elif vtype == "discrete":
                if len(values) >= MIN_CONFIDENCE:
                    single_constraints[name] = ("in", frozenset(values))
                # 否则跳过：样本不足，离散值集合可能因聚类误差而过拟合
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
                        vars[name],     # 实际值
                        val,            # 期望值
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

        # 样本过多时随机采样，避免 Apriori 内存爆炸
        MAX_TRANSACTIONS = 500
        if len(var_instances) > MAX_TRANSACTIONS:
            import random as _random
            _rng = _random.Random(42)
            var_instances = _rng.sample(var_instances, MAX_TRANSACTIONS)

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
                if name in self._GUARD_SKIP_VARS:
                    continue    # 跳过非协议语义特征
                if any(name.startswith(pfx) for pfx in self._GUARD_SKIP_PREFIX):
                    continue    # 跳过静态常量字节，不参与关联规则挖掘
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

        # 样本量过少时跳过 Apriori（全常量变量会导致 2^N 次组合爆炸）
        MIN_SAMPLES = 4
        if len(transactions) < MIN_SAMPLES:
            return [], cached_means

        # 唯一项过多时跳过 Apriori（2^N 项集导致内存爆炸）
        MAX_UNIQUE_ITEMS = 12
        unique_items = set()
        for t in transactions:
            unique_items.update(t)
        if len(unique_items) > MAX_UNIQUE_ITEMS:
            return [], cached_means

        # 所有事务完全相同时跳过 Apriori：
        # 若每条消息的字段值组合都一样，任意两个字段之间都会形成 confidence=1 的关联规则，
        # 产生 C(n,2)*2 条平凡规则，无判别意义（单变量 eq 约束已经覆盖）。
        unique_transactions = set(transactions)
        if len(unique_transactions) == 1:
            return [], cached_means

        # 使用Apriori算法挖掘频繁项集合关联规则
        try:
            fis = self.core.frequent_itemsets(transactions, self.min_support)       # 频繁项集（support >= min_support）
            rules = self.core.association_rules(fis, self.min_confidence)           # 关联规则（confidence >= min_confidence）
        except MemoryError:
            return [], cached_means

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
