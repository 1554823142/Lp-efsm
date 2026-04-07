from typing import Dict, List, Tuple, Optional, Callable, FrozenSet, Any

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
        max_guards_per_transition: int = 4,
    ):
        self.discrete_threshold = discrete_threshold
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.delta_tolerance = delta_tolerance
        self.continuous_tolerance = continuous_tolerance
        self.linear_r2_threshold = linear_r2_threshold
        self.linear_residual_tol = linear_residual_tol
        self.enable_triplet_sum = enable_triplet_sum
        self.max_guards_per_transition = max_guards_per_transition

        self.core = AprioriCore()
        self.linear_detector = LinearRelationDetector(
            r2_threshold=linear_r2_threshold,
            residual_tol=linear_residual_tol,
        )

    def learn(
        self,
        var_instances: List[Dict[str, float]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        if not var_instances:
            return None, None

        var_names = list(var_instances[0].keys())
        var_types = {
            name: self._infer_var_type([v[name] for v in var_instances if name in v])
            for name in var_names
        }

        guard = self._learn_guard(var_instances, var_names, var_types, context=context)
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
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Callable]:

        # 1. 候选守卫条件收集
        n_samples = len(var_instances)
        min_conf = self.discrete_threshold if n_samples > 10 else 2
        min_samples_range = 5
        
        blacklist = context.get("guard_blacklist", set()) if context else set()
        symbol_constants = context.get("symbol_constants", {}) if context else {}

        candidate_constraints: List[Dict[str, Any]] = []

        # 1.1 收集单变量约束
        for name in var_names:
            if name in self._GUARD_SKIP_VARS or name in blacklist:
                continue
            
            values = [v[name] for v in var_instances if name in v]
            if not values:
                continue
            vtype = var_types[name]
            
            constraint = None
            if vtype == "constant":
                val = values[0]
                # 即使是符号全局常量，我们也收集它，但在显著性评分中会给予极低分
                # 这样如果该转移没有其它更好的 Guard，它依然可能入选，从而匹配 GT
                if n_samples >= min_conf:
                    if not (name.startswith("b") and val == 0.0 and n_samples < 10):
                        constraint = ("eq", val)
            elif vtype == "discrete":
                if n_samples >= min_conf:
                    constraint = ("in", frozenset(values))
            elif vtype == "sequential":
                delta = values[1] - values[0] if len(values) > 1 else 0.0
                constraint = ("delta", delta)
            elif vtype == "continuous":
                if n_samples >= min_samples_range:
                    span = max(values) - min(values)
                    if not (name.startswith("b") and span > 0.92):
                        tol = span * self.continuous_tolerance
                        constraint = ("range", (min(values) - tol, max(values) + tol))

            if constraint:
                score = self._calculate_significance(name, constraint, n_samples, context)
                candidate_constraints.append({
                    "type": "single",
                    "name": name,
                    "constraint": constraint,
                    "score": score
                })

        # 1.2 收集关联规则
        joint_rules, joint_rule_means = self._mine_joint_rules(
            var_instances, var_names, var_types
        )
        for rule in joint_rules:
            # rule: (ante_vars, ante_vals, cons_vars, cons_vals, conf)
            score = rule[4] * 0.8  # 关联规则基础分略低于强常量约束
            candidate_constraints.append({
                "type": "joint",
                "rule": rule,
                "score": score
            })

        # 1.3 收集线性关系
        continuous_vars = [n for n in var_names if var_types[n] == "continuous"]
        linear_pairs = []
        triplet_sums = []
        if len(continuous_vars) >= 2:
            linear_pairs = self.linear_detector.detect_pairwise(var_instances, continuous_vars)
            for lp in linear_pairs:
                # lp: (a_name, b_name, k, c, r2)
                score = lp[4] * 0.7
                candidate_constraints.append({
                    "type": "linear",
                    "pair": lp,
                    "score": score
                })
            
            if self.enable_triplet_sum and len(continuous_vars) >= 3:
                triplet_sums = self.linear_detector.detect_triplet_sum(var_instances, continuous_vars)
                for ts in triplet_sums:
                    # ts: (a, b, c, max_res)
                    score = 0.65  # 三元求和相对固定分
                    candidate_constraints.append({
                        "type": "triplet",
                        "triplet": ts,
                        "score": score
                    })

        # 2. 守卫条件剪枝 (基于显著性评分)
        candidate_constraints.sort(key=lambda x: -x["score"])
        
        # 2.1 简单的去重逻辑：如果两个相邻字节都被选为 Guard 且分值接近，
        # 说明它们极可能是同一个多字节字段 (如 TxID)，只取一个以腾出名额给其它字段
        filtered_candidates = []
        seen_positions = set()
        for cand in candidate_constraints:
            if cand["type"] == "single" and cand["name"].startswith("b"):
                try:
                    pos = int(cand["name"][1:])
                    if (pos - 1 in seen_positions or pos + 1 in seen_positions) and len(filtered_candidates) >= 2:
                        continue
                    seen_positions.add(pos)
                except ValueError:
                    pass
            filtered_candidates.append(cand)

        selected_candidates = filtered_candidates[:self.max_guards_per_transition]
        
        # 重新整理选中的约束 (使用原有变量名以保持与评估器兼容)
        single_constraints: Dict[str, tuple] = {}
        joint_rules: List[tuple] = []
        linear_pairs: List[tuple] = []
        triplet_sums: List[tuple] = []
        
        for cand in selected_candidates:
            if cand["type"] == "single":
                single_constraints[cand["name"]] = cand["constraint"]
            elif cand["type"] == "joint":
                joint_rules.append(cand["rule"])
            elif cand["type"] == "linear":
                linear_pairs.append(cand["pair"])
            elif cand["type"] == "triplet":
                triplet_sums.append(cand["triplet"])

        # 将方法提取为局部变量，避免闭包捕获 self
        _check_binding = self._check_joint_binding
        _vtypes = var_types
        _linear_tol = self.linear_residual_tol

        # 3. 构建 Guard 函数
        def guard(vars: Dict[str, float]) -> bool:
            # 1. 单变量约束检查
            for name, constraint in single_constraints.items():
                if name not in vars:
                    return False
                val = vars[name]
                ctype = constraint[0]
                if ctype == "eq":
                    if abs(val - constraint[1]) > 1e-5:
                        return False
                elif ctype == "in":
                    if val not in constraint[1]:
                        return False
                elif ctype == "range":
                    vmin, vmax = constraint[1]
                    if not (vmin <= val <= vmax):
                        return False

            # 2. 关联规则验证(A->B)
            for (ante_vars, ante_vals, cons_vars, cons_vals, _conf) in joint_rules:
                ante_ok = all(
                    name in vars and _check_binding(
                        name, vars[name], val, _vtypes, joint_rule_means
                    )
                    for name, val in zip(ante_vars, ante_vals)
                )
                if ante_ok:
                    cons_ok = all(
                        name in vars and _check_binding(
                            name, vars[name], val, _vtypes, joint_rule_means
                        )
                        for name, val in zip(cons_vars, cons_vals)
                    )
                    if not cons_ok:
                        return False

            # 3. 线性关系检查
            for a_name, b_name, k, c, _r2 in linear_pairs:
                if a_name not in vars or b_name not in vars:
                    continue
                expected_a = k * vars[b_name] + c
                abs_tol = max(_linear_tol, abs(expected_a) * 1e-3)
                if abs(vars[a_name] - expected_a) > abs_tol:
                    return False

            for a_name, b_name, c_name, _max_res in triplet_sums:
                if a_name not in vars or b_name not in vars or c_name not in vars:
                    continue
                residual = abs(vars[a_name] + vars[b_name] - vars[c_name])
                if residual > _linear_tol * 10:
                    return False

            return True

        return guard

    def _calculate_significance(
        self,
        name: str,
        constraint: tuple,
        n_samples: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        计算守卫条件的显著性评分。
        基于“位置重要性”和“区分能力”进行加权。
        """
        ctype = constraint[0]
        
        # 1. 基础分 (约束强度：相等 > 范围)
        score = 1.0 if ctype == "eq" else 0.5
        
        # 2. 位置加成 (核心协议头部区域 b0-b11 权重最高)
        try:
            if name.startswith("b"):
                pos = int(name[1:])
                if pos <= 7:
                    score *= 5.0  # 极高权重：工业协议的功能码、类型码通常在此
                elif pos <= 15:
                    score *= 3.0
                elif pos <= 32:
                    score *= 1.5
            elif name.startswith("s"):
                score *= 2.0  # 静态常量通常是协议标识符
        except (ValueError, TypeError):
            pass
            
        # 3. 区分能力 (信息增益)
        if context and ctype == "eq":
            global_diversity = context.get("var_global_diversity", {}).get(name, 1)
            symbol_diversity = context.get("var_symbol_diversity", {}).get(name, 1)
            symbol_constants = context.get("symbol_constants", {})
            
            # 优先考虑在当前符号(Symbol)下具有区分能力的变量
            if symbol_diversity > 1:
                import math
                # 符号级区分度是核心
                score *= (2.0 + 0.5 * math.log(symbol_diversity))
            
            # 如果该变量在整个 Symbol 范围内都是同一个常量，则它几乎没有判别力
            is_symbol_constant = name in symbol_constants and abs(symbol_constants[name] - constraint[1]) < 1e-5
            if is_symbol_constant:
                score *= 0.1
            elif global_diversity > 1:
                import math
                # 全局多样性：适度多样是好的 (如 FC)，但过度多样 (如 ID) 可能是过拟合
                if global_diversity > 20:
                    # 过度多样性惩罚：ID、序列号等全局取值极多，不适合做 Guard
                    score *= (1.0 / (1.0 + 0.1 * math.log(global_diversity)))
                else:
                    score *= (1.0 + 0.2 * math.log(global_diversity))

        # 4. 样本量加成
        score *= min(1.5, 0.5 + n_samples / 20.0)
        
        # 5. Padding 惩罚 (eq 0 通常信息量低)
        if ctype == "eq" and abs(constraint[1]) < 1e-9:
            score *= 0.4
            
        return score

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
        # action_rules 形式为 {var_name: (atype, param)}
        # 这种形式能被 efsm_evaluator.py 中的 _extract_fields_from_closure 正确解析
        action_rules: Dict[str, tuple] = {}
        
        for name in var_names:
            if name.startswith("s"):
                continue
            values = [v[name] for v in var_instances if name in v]
            if len(values) < 2:
                continue
                
            vtype = var_types[name]
            if vtype == "sequential":
                # 计算步长
                deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_delta = sum(deltas) / len(deltas)
                # 如果步长稳定，学习该更新规则
                if all(abs(d - avg_delta) < 1e-5 for d in deltas):
                    action_rules[name] = ("delta", avg_delta)
            elif vtype == "continuous":
                # 对于连续变量，如果平均变化较大，记录为 delta 更新
                deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_delta = sum(deltas) / len(deltas)
                if abs(avg_delta) > self.delta_tolerance:
                    action_rules[name] = ("delta", avg_delta)
            # 其它情况默认为 keep，评估器会自动忽略 keep 的变量
            
        if not action_rules:
            return None
            
        def action_func(vars_dict: Dict[str, float]) -> Dict[str, float]:
            new_vars = vars_dict.copy()
            for name, (atype, param) in action_rules.items():
                if name in new_vars:
                    if atype == "delta":
                        new_vars[name] += param
            return new_vars
            
        return action_func
