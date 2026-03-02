from typing import Dict, List, Tuple, Optional, Callable
from protocol_infer.core.algorithm.guard_action import GuardActionLearner

class IntervalDeltaLearner(GuardActionLearner):
    def learn(self, var_instances: List[Dict[str, float]]) -> Tuple[Optional[Callable], Optional[Callable]]:
        if not var_instances:
            return None, None
        var_names = list(var_instances[0].keys())
        stats = {}
        for name in var_names:
            values = [v[name] for v in var_instances if name in v]
            if values:
                stats[name] = (min(values), max(values))
        def guard(vars: Dict[str, float]) -> bool:
            for name, (vmin, vmax) in stats.items():
                if name not in vars:
                    return False
                if not (vmin <= vars[name] <= vmax):
                    return False
            return True
        if len(var_instances) < 2:
            def identity(vars: Dict[str, float]) -> Dict[str, float]:
                return vars.copy()
            return guard, identity
        avg_changes = {}
        for name in var_names:
            values = [v[name] for v in var_instances if name in v]
            if len(values) > 1:
                changes = [values[i+1] - values[i] for i in range(len(values)-1)]
                avg_changes[name] = sum(changes) / len(changes)
        def action(vars: Dict[str, float]) -> Dict[str, float]:
            new_vars = vars.copy()
            for name, delta in avg_changes.items():
                if name in new_vars:
                    new_vars[name] += delta
            return new_vars
        return guard, action
