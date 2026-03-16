from typing import Dict, List, Tuple, Optional, Callable, Any


class CrossMessageLearner:
    '''
        跨消息学习器
    '''
    def __init__(
        self,
        r2_threshold: float = 0.999,        # 线性推导规则的接受门槛
        residual_tol: float = 1e-4,         # 两类规则的精度容差
        min_samples: int = 4,               # 学习任何规则所需的最少配对样本数(避免误判)
        delta_tolerance: float = 1e-6,      # 序列递增规则的步长一致性判断
    ):
        self.r2_threshold = r2_threshold
        self.residual_tol = residual_tol
        self.min_samples = min_samples
        self.delta_tolerance = delta_tolerance

    def learn(
        self,
        pair_instances: List[Tuple[Dict[str, float], Dict[str, float]]],        # (src, dst)相邻两条消息的配对数据
    ) -> Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], bool]]:
        if not pair_instances:
            return None

        src_names = set()
        dst_names = set()
        for src, dst in pair_instances:
            src_names.update(src.keys())
            dst_names.update(dst.keys())

        # 三类规则学习
        identity_rules: List[Tuple[str, str]] = []          # 恒等关系(如dst.x == src.y)
        seq_rules: List[Tuple[str, float]] = []             # 序列递增
        linear_rules: List[Tuple[str, str, float, float, float]] = []   # 线性关系


        # 1. 恒等关系
        for dst_var in sorted(dst_names):
            for src_var in sorted(src_names):
                samples = [
                    (src[src_var], dst[dst_var])
                    for src, dst in pair_instances
                    if src_var in src and dst_var in dst
                ]
                if len(samples) < self.min_samples:
                    continue
                if all(abs(y - x) <= self.residual_tol for x, y in samples):
                    identity_rules.append((dst_var, src_var))
        
        # 2. 序列递增关系
        for var in sorted(src_names.intersection(dst_names)):
            deltas = [
                dst[var] - src[var]
                for src, dst in pair_instances
                if var in src and var in dst
            ]
            if len(deltas) < self.min_samples:
                continue
            d0 = deltas[0]
            if max(abs(d - d0) for d in deltas) < self.delta_tolerance:
                seq_rules.append((var, d0))

        identity_set = set(identity_rules)
        seq_set = {v for v, _d in seq_rules}

        # 3. 线性关系
        for dst_var in sorted(dst_names):
            for src_var in sorted(src_names):
                # 跳过已有的前两种关系
                if (dst_var, src_var) in identity_set:
                    continue
                if dst_var == src_var and dst_var in seq_set:
                    continue
                xs = []
                ys = []
                for src, dst in pair_instances:
                    if src_var in src and dst_var in dst:
                        xs.append(src[src_var])
                        ys.append(dst[dst_var])
                if len(xs) < self.min_samples:
                    continue

                k, c, r2 = self._fit_linear(ys, xs)
                if abs(k) < 1e-9:
                    continue
                if r2 >= self.r2_threshold:
                    linear_rules.append((dst_var, src_var, k, c, r2))

        def guard(vars: Dict[str, Any], memory: Optional[Dict[str, Any]] = None) -> bool:
            if not memory:          # 跳过第一条信息(此时无历史记录)
                return True
            
            # 1. 恒等
            for dst_var, src_var in identity_rules:
                if dst_var not in vars or src_var not in memory:
                    return False
                if abs(float(vars[dst_var]) - float(memory[src_var])) > self.residual_tol:
                    return False

            # 2. 序列递增
            for var, delta in seq_rules:
                if var not in vars or var not in memory:
                    return False
                expected = float(memory[var]) + delta
                abs_tol = max(self.residual_tol, abs(expected) * 1e-3)
                if abs(float(vars[var]) - expected) > abs_tol:
                    return False

            # 3. 线性关系
            for dst_var, src_var, k, c, _r2 in linear_rules:
                if dst_var not in vars or src_var not in memory:
                    return False
                expected = k * float(memory[src_var]) + c
                abs_tol = max(self.residual_tol, abs(expected) * 1e-3)
                if abs(float(vars[dst_var]) - expected) > abs_tol:
                    return False

            return True


        # 都不满足则说明无可学习的跨消息依赖
        if not identity_rules and not seq_rules and not linear_rules:
            return None

        return guard

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

