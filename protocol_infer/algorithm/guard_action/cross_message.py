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
    ) -> Tuple[
        Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], bool]],
        Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]]],
    ]:
        if not pair_instances:
            return None, None

        src_names = set()
        dst_names = set()
        for src, dst in pair_instances:
            src_names.update(src.keys())
            dst_names.update(dst.keys())

        # 过滤无判别力的变量：
        # s-前缀（如 s0, s2）是全局静态常量（值恒为某固定值，如 0），
        # 作为 src 变量时会与任何恰好值也为 0 的 dst 变量产生假恒等规则。
        # direction/entropy/len 是协议无关的衍生特征，不应参与跨消息规则学习。
        _SKIP_PREFIX = ("s",)
        _SKIP_VARS = {"direction", "entropy", "len"}
        src_names = {
            n for n in src_names
            if n not in _SKIP_VARS and not any(n.startswith(p) for p in _SKIP_PREFIX)
        }
        dst_names = {
            n for n in dst_names
            if n not in _SKIP_VARS and not any(n.startswith(p) for p in _SKIP_PREFIX)
        }

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
                if not all(abs(y - x) <= self.residual_tol for x, y in samples):
                    continue
                # 要求值多样性：若所有样本中 src_var 只有唯一一个值（如恒为 0.0 或 3.0），
                # 则等式是常量平凡等式而非真实的跨消息依赖，跳过。
                # 对同名变量（curr.X == prev.X）：常量字段的恒等规则无判别力，
                #   真正有意义的是"会变化但每次都保持与上条相同"的字段（如 fc 在同一会话内固定）。
                # 对异名变量（curr.X == prev.Y）：防止巧合零值等。
                distinct_src_vals = {x for x, _y in samples}
                if len(distinct_src_vals) < 2:
                    continue
                identity_rules.append((dst_var, src_var))

        # 2. 序列递增关系
        for var in sorted(src_names.intersection(dst_names)):
            src_vals = [src[var] for src, dst in pair_instances if var in src and var in dst]
            dst_vals = [dst[var] for src, dst in pair_instances if var in src and var in dst]
            if len(src_vals) < self.min_samples:
                continue
            deltas = [d - s for s, d in zip(src_vals, dst_vals)]
            d0 = deltas[0]
            if max(abs(d - d0) for d in deltas) >= self.delta_tolerance:
                continue
            # delta=0 意味着变量不变，与恒等规则 curr.x==prev.x 重复，跳过
            if abs(d0) < self.delta_tolerance:
                continue
            # 要求 src 值有多样性：若 src 的值只有一个（如每次 request.b11 都是 100），
            # delta 的一致性可能只是巧合（不同消息类型的字节布局重合）而非真实依赖。
            if len(set(src_vals)) < 2:
                continue
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
            return None, None

        # cross_action：把当前消息变量写入 memory，供下一条消息的 memory_guard 使用。
        # 同时携带学到的跨消息规则描述（identity/seq/linear），便于检视工具提取。
        def cross_action(
            vars: Dict[str, Any],
            memory: Optional[Dict[str, Any]] = None,
            _identity_rules=identity_rules,
            _seq_rules=seq_rules,
            _linear_rules=linear_rules,
        ) -> Dict[str, Any]:
            if memory is not None:
                mem = memory.data if hasattr(memory, "data") else memory
                if isinstance(mem, dict):
                    mem.update(vars)
            return vars.copy()

        return guard, cross_action

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

