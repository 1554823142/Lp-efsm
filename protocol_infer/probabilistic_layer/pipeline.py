from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.interface.prob_trainer import ProbTrainer
from protocol_infer.core.model.efsm import EFSM
from protocol_infer.core.model.pefsm import PEFSM
from protocol_infer.probabilistic_layer.inference.prob_infer import PEFSMInferencer


class ProbabilisticPipeline:
    def __init__(
        self,
        trainer: Optional[ProbTrainer] = None,
    ):
        self.trainer = trainer if trainer is not None else PEFSMInferencer()

    def build_sequences(self, trace: Trace) -> Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]:
        sequences: Dict[SessionKey, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
        for msg in trace.abstract_messages:
            sequences[msg.session_key].append((msg.symbol, msg.vars))
        return sequences

    def run(
        self,
        efsm: EFSM,
        trace: Optional[Trace] = None,
        sequences: Optional[Dict[SessionKey, List[Tuple[str, Dict[str, float]]]]] = None,
    ) -> PEFSM:
        if sequences is None:
            if trace is None:
                raise ValueError("trace 和 sequences 不能同时为空")
            sequences = self.build_sequences(trace)
        return self.trainer.train(efsm, sequences)
