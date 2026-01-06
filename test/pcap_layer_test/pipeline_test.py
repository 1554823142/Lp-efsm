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

from protocol_infer.pcap_layer.pipeline import PCAPPipeline


PATH_STR = str(project_root / "Data" / "MODBUS" / "modbus_test_data_part2.pcap")

def test_pcap_to_trace():
    pipeline = PCAPPipeline()
    trace = pipeline.run(PATH_STR)

    print(f"Total events: {len(trace.events)}")

    for i, event in enumerate(trace.events[:10]):
        print(f"[{i}] {event.session_key} len={len(event.payload)}")


if __name__ == "__main__":
    test_pcap_to_trace()