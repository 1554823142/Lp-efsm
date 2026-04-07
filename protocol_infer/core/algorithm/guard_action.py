from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable, Any


class GuardActionLearner(ABC):
    """
    Guard/Action 学习算法接口
    """

    @abstractmethod
    def learn(
        self,
        var_instances: List[Dict[str, float]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Callable], Optional[Callable]]:
        """
        根据变量实例序列学习 guard 和 action
        
        Args:
            var_instances: 该转移在所有会话中出现的变量实例列表
            context: 学习上下文（包含全局统计信息等）
        """
        pass
