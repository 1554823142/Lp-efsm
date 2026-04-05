import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from protocol_infer.evaluation.ground_truth import ProtocolGroundTruthExtractor
from protocol_infer.evaluation.metrics import ClusteringEvaluator, FSMEvaluator, EFSMevaluator
from protocol_infer.evaluation.mutation import TraceMutator
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.core.datamodel.trace import Trace

def match_events_with_ground_truth(trace, gt_traces, tolerance=1e-3):
    """
    Match inferred events with ground truth labels based on timestamp.
    Returns: list of (event, label) tuples.
    """
    # Flatten GT traces into a list of (timestamp, label, session_key)
    gt_list = []
    for session_key, msgs in gt_traces.items():
        for msg in msgs:
            gt_list.append({
                'timestamp': msg['timestamp'], 
                'label': msg['msg_type'],
                'session_key': session_key # GT session key format might differ slightly from Scapy
            })
            
    # Sort by timestamp
    gt_list.sort(key=lambda x: x['timestamp'])
    
    # Sort trace events
    events = sorted(trace.events, key=lambda x: x.timestamp)
    
    matched_events = []
    gt_idx = 0
    n_gt = len(gt_list)
    
    matched_count = 0
    
    for ev in events:
        # Find matching GT within tolerance
        best_match_idx = -1
        min_diff = float('inf')
        
        # Search window around current gt_idx
        # Because timestamps are float and might drift slightly or differ in precision
        start_search = max(0, gt_idx - 100)
        end_search = min(n_gt, gt_idx + 100)
        
        for i in range(start_search, end_search):
            # Check session key matching if possible? 
            # Scapy session key: (src, src_port, dst, dst_port, proto)
            # GT session key: ((ip, port), (ip, port)) sorted
            
            diff = abs(ev.timestamp - gt_list[i]['timestamp'])
            if diff < tolerance and diff < min_diff:
                min_diff = diff
                best_match_idx = i
        
        if best_match_idx != -1:
            matched_events.append((ev, gt_list[best_match_idx]['label']))
            gt_idx = best_match_idx + 1 # Advance partially
            matched_count += 1
        else:
            # Event has no label (maybe ignored by GT extractor, e.g. TCP handshake)
            pass
            
    print(f"Matched {matched_count}/{len(events)} events with Ground Truth.")
    return matched_events

def run_evaluation(pcap_paths, protocol_name, n_splits=5):
    if isinstance(pcap_paths, str):
        pcap_paths = [pcap_paths]
        
    print(f"\n--- Evaluation for {protocol_name} ---")
    print(f"PCAPs: {pcap_paths}")
    
    all_gt_traces = defaultdict(list)
    all_trace_events = []
    
    # 1. Load Data from all PCAPs
    for pcap_path in pcap_paths:
        if not os.path.exists(pcap_path):
            print(f"PCAP not found: {pcap_path}")
            continue
            
        print(f"Processing {pcap_path}...")
        
        # Extract GT
        gt_extractor = ProtocolGroundTruthExtractor(pcap_path, protocol_name)
        gt_traces = gt_extractor.extract()
        # Merge GT traces (need unique session keys across files? usually IP/Port differs or we treat them separate)
        # To avoid key collision, append file index to key?
        for k, v in gt_traces.items():
            all_gt_traces[str(k) + pcap_path] = v
            
        # Extract Events
        pcap_pipeline = PCAPPipeline()
        trace = pcap_pipeline.run(pcap_path)
        
        # Match immediately
        matched = match_events_with_ground_truth(trace, gt_traces)
        all_trace_events.extend(matched)

    print(f"Total Matched Events: {len(all_trace_events)}")
    
    # Group by session (Scapy session key)
    # Note: match_events_with_ground_truth returns (ev, label)
    # ev.session_key is from Scapy. If we process multiple files, session keys might collide if IPs are same.
    # But Scapy Trace events usually don't carry file info.
    # For simplicity, assuming distinct sessions or treating same 5-tuple as same session logic across files.
    
    sessions = defaultdict(list)
    for ev, label in all_trace_events:
        sessions[ev.session_key].append((ev, label))
        
    session_keys = list(sessions.keys())
    print(f"Total Sessions: {len(session_keys)}")
    
    if len(session_keys) < n_splits:
        print(f"Warning: Number of sessions ({len(session_keys)}) < n_splits ({n_splits}). Reducing n_splits.")
        n_splits = len(session_keys)
        if n_splits < 2:
            print("Not enough sessions for cross-validation.")
            return

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(session_keys)):
        print(f"\nFold {fold+1}/{n_splits}...")
        
        train_keys = [session_keys[i] for i in train_idx]
        test_keys = [session_keys[i] for i in test_idx]
        
        # Prepare Training Data
        train_events = []
        for k in train_keys:
            train_events.extend([x[0] for x in sessions[k]])
            
        train_trace = Trace(events=train_events)
        
        # Prepare Test Data
        test_events = []
        test_labels = []
        for k in test_keys:
            test_events.extend([x[0] for x in sessions[k]])
            test_labels.extend([x[1] for x in sessions[k]])
            
        # 4. Train Control Flow (Clustering + FSM)
        # Tuning parameters: n_clusters=5 (reduced from 8), k=2 for State Merging
        cf_pipeline = ControlFlowPipeline(algorithm="kmeans", n_clusters=5, k=2) 
        try:
            fsm = cf_pipeline.run(train_trace)
            print(f"  Inferred FSM: {len(fsm.states)} states, {len(fsm.transitions)} transitions")
        except Exception as e:
            print(f"  Training failed: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # 5. Evaluate Clustering (Layer 1)
        test_features = cf_pipeline.featureer.extract(test_events)
        test_symbols = [cf_pipeline.abstractor.abstract(f) for f in test_features]
        
        clustering_eval = ClusteringEvaluator()
        c_metrics = clustering_eval.evaluate(test_labels, test_symbols)
        print(f"  Clustering Metrics: ARI={c_metrics['ARI']:.3f}, NMI={c_metrics['NMI']:.3f}")
        
        # 6. Evaluate FSM (Layer 2)
        test_sequences = []
        current_sess_idx = 0
        for k in test_keys:
            sess_len = len(sessions[k])
            sess_symbols = test_symbols[current_sess_idx : current_sess_idx + sess_len]
            test_sequences.append(sess_symbols)
            current_sess_idx += sess_len
            
        fsm_eval = FSMEvaluator()
        f_metrics = fsm_eval.evaluate(None, fsm, test_traces=test_sequences)
        print(f"  FSM Metrics: Coverage={f_metrics.get('Trace Coverage', 0):.3f}")
        
        fold_metrics.append({**c_metrics, **f_metrics})

    # Average Metrics
    if fold_metrics:
        print("\n--- Average Metrics ---")
        avg_ari = np.mean([m['ARI'] for m in fold_metrics])
        avg_nmi = np.mean([m['NMI'] for m in fold_metrics])
        avg_cov = np.mean([m.get('Trace Coverage', 0) for m in fold_metrics])
        print(f"ARI: {avg_ari:.3f}")
        print(f"NMI: {avg_nmi:.3f}")
        print(f"Trace Coverage: {avg_cov:.3f}")

if __name__ == "__main__":
    # Example usage
    pcap_dir = os.path.join("Data", "MODBUS")
    pcaps = [
        os.path.join(pcap_dir, "modbus_test_data_part1.pcap"),
        # Add more if available
        # os.path.join(pcap_dir, "modbus_test_data_part2.pcap") 
    ]
    
    # Check for part2
    part2 = os.path.join(pcap_dir, "modbus_test_data_part2.pcap")
    if os.path.exists(part2):
        pcaps.append(part2)
        
    run_evaluation(pcaps, "modbus", n_splits=5)
