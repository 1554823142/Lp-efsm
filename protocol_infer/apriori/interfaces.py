from typing import Generic, TypeVar, List, FrozenSet, Any, Iterable
from abc import ABC, abstractmethod

# 泛型类型变量
T = TypeVar("T")
R = TypeVar("R")
I = TypeVar("I")

class TransactionBuilder(ABC, Generic[T]):
    @abstractmethod
    def build(self, items: List[T]) -> List[FrozenSet[Any]]:
        '''
            把原始数据转换成 Apriori 能处理的事务格式
        '''
        ...

class ItemDiscretizer(ABC, Generic[I]):
    @abstractmethod
    def discretize(self, value: I) -> Any:
        '''
        把连续值映射成离散的"item"，方便做频繁项挖掘
        '''
        ...

class ResultInterpreter(ABC, Generic[R]):
    @abstractmethod
    def interpret(self, frequent_itemsets: List[tuple]) -> Iterable[R]:
        '''
            把 Apriori 挖掘出的频繁项集转换成业务可读的格式
        '''
        ...
