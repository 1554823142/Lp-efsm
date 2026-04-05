import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from protocol_infer.evaluation.ground_truth import ProtocolGroundTruthExtractor

def test_extract_modbus():
    pcap_path = os.path.join("Data", "MODBUS", "modbus_test_data_part1.pcap")
    if not os.path.exists(pcap_path):
        print(f"Skipping test, PCAP not found: {pcap_path}")
        return
        
    extractor = ProtocolGroundTruthExtractor(pcap_path, 'modbus')
    traces = extractor.extract()
    
    print(f"Extracted {len(traces)} sessions.")
    for session, msgs in list(traces.items())[:2]: # show max 2 sessions
        print(f"Session {session}: {len(msgs)} messages")
        for msg in msgs[:5]: # show max 5 messages per session
            print(f"  {msg['direction']}: {msg['msg_type']}")
            
    # Export to json
    output_path = "modbus_gt_test.json"
    extractor.export_to_json(output_path)
    print(f"Check {output_path} for detailed output.")

if __name__ == "__main__":
    test_extract_modbus()
