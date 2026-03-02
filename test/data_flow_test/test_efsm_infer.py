import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root / "protocol_infer"))
sys.path.insert(0, str(project_root))

from protocol_infer.core.model.fsm import FSM
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.data_flow_layer.inference.efsm_infer import EFSMInferencer

def build_simple_fsm():
    fsm = FSM()
    s0 = fsm.new_state(is_start=True)
    s1 = fsm.new_state()
    s2 = fsm.new_state()
    fsm.add_transition(src=s0, dst=s1, symbol="A")
    fsm.add_transition(src=s1, dst=s2, symbol="B")
    return fsm

def test_efsm_build_and_step():
    sk1 = SessionKey("1.1.1.1", 1000, "2.2.2.2", 2000, "udp")
    sk2 = SessionKey("3.3.3.3", 3000, "4.4.4.4", 4000, "udp")
    fsm = build_simple_fsm()
    sequences = {
        sk1: [("A", {"len": 10.0}), ("B", {"len": 20.0})],
        sk2: [("A", {"len": 12.0}), ("B", {"len": 25.0})],
    }
    efsm = EFSMInferencer().build_efsm(fsm, sequences)
    assert "len" in efsm.variable_defs
    dst, new_vars = efsm.step(0, "A", {"len": 11.0})
    assert dst == 1
    assert isinstance(new_vars, dict)
    dst2, new_vars2 = efsm.step(1, "B", {"len": 22.0})
    assert dst2 == 2
    assert isinstance(new_vars2, dict)
