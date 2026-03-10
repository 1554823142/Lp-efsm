from typing import Dict, List, Tuple, Optional, Callable
from protocol_infer.core.algorithm.guard_action import GuardActionLearner

class IntervalDeltaLearner(GuardActionLearner):
    def learn(self, var_instances: List[Dict[str, float]]) -> Tuple[Optional[Callable], Optional[Callable]]:
        if not var_instances:
            return None, None
        var_names = list(var_instances[0].keys())       # 第一个var的keys作为所有var的keys

        # 收集所有变量的值, 并计算取值范围
        stats = {}
        for name in var_names:
            values = [v[name]               # 取该变量的值
                        for v in var_instances 
                        if name in v]       # 过滤掉没有该变量的实例(缺失值)
            if values:
                stats[name] = (min(values), max(values))        # 记录每个变量的取值范围


        def guard(vars: Dict[str, float]) -> bool:  
            '''
                区间法
            '''
            for name, (vmin, vmax) in stats.items():        # 由于闭包的特性, 可以在learn调用完还能维持stats以供guard使用
                if name not in vars:
                    return False
                if not (vmin <= vars[name] <= vmax):            # 满足区间才返回True
                    return False
            return True
        
        # 少量数据的情况, 直接返回当前vars(副本)
        if len(var_instances) < 2:
            def identity(vars: Dict[str, float]) -> Dict[str, float]:
                return vars.copy()              # 即f(x)=x, 一个恒等函数
            return guard, identity

        
        avg_changes = {}
        for name in var_names:
            values = [v[name] 
                        for v in var_instances
                        if name in v]

            if len(values) > 1:
                changes = [values[i+1] - values[i] for i in range(len(values)-1)]   
                avg_changes[name] = sum(changes) / len(changes)

        def action(vars: Dict[str, float]) -> Dict[str, float]:
            '''
                平均增量法
                计算每个变量在相邻触发实例之间的变化量，取平均值作为 action 的更新规则

            '''
            new_vars = vars.copy()
            for name, delta in avg_changes.items():
                if name in new_vars:
                    new_vars[name] += delta
            return new_vars
        
        return guard, action
