import sys
import os
from pathlib import Path
current_file = Path(__file__).resolve()  # pipeline_test.py的绝对路径

# 计算项目根目录（p-efsm文件夹）
# 假设项目根目录在当前文件的上两级
project_root = current_file.parent.parent.parent

print(f"当前文件: {current_file}")
print(f"项目根目录: {project_root}")

# Ensure top-level packages inside protocol_infer are importable (like `core`, `pcap_layer`)
sys.path.insert(0, str(project_root / "protocol_infer"))
# Also add project root for other absolute imports if needed
sys.path.insert(0, str(project_root))


from protocol_infer.control_flow_layer.inference.pta_infer import PTAInfer
from protocol_infer.core.datamodel.session import SessionKey


def test_pta_infer_shared_prefix():
    sk1 = SessionKey("1.1.1.1", 123, "2.2.2.2", 80, "tcp")
    sk2 = SessionKey("3.3.3.3", 111, "4.4.4.4", 80, "tcp")

    sequences = {
        sk1: ["a", "b", "c"],
        sk2: ["a", "b", "d"],
    }

    fsm = PTAInfer().infer(sequences)

    # start state created
    assert fsm.start_state == 0

    # expected states: s0 (start), s1 (a), s2 (b), s3 (c), s4 (d)
    assert fsm._next_state_id == 5

    # transitions: a, b, c, d
    assert len(fsm.transitions) == 4

    # two accepting (end) states (for c and d)
    ends = [sid for sid, s in fsm.states.items() if s.is_end]
    assert len(ends) == 2


def test_pta_infer_empty_sequence():
    sk = SessionKey("1.1.1.1", 1000, "2.2.2.2", 2000, "udp")
    fsm = PTAInfer().infer({sk: []})

    assert fsm.start_state == 0
    assert fsm.states[0].is_end
