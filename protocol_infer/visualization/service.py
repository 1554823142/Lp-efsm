import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from protocol_infer.core.datamodel.event import MessageEvent
from protocol_infer.core.datamodel.session import SessionKey
from protocol_infer.core.datamodel.trace import Trace
from protocol_infer.core.model.pefsm import PEFSM
from protocol_infer.pcap_layer.pipeline import PCAPPipeline
from protocol_infer.control_flow_layer.pipeline import ControlFlowPipeline
from protocol_infer.data_flow_layer.pipeline import DataFlowPipeline
from protocol_infer.probabilistic_layer.pipeline import ProbabilisticPipeline
from protocol_infer.evaluation.run_evaluation import (
    run_full_evaluation,
    run_full_evaluation_from_sessions,
    generate_synthetic_sessions,
    _LABELERS,
)
from protocol_infer.evaluation.supervised_eval import _group_sessions, _split_long_sessions, _split_keys, _looks_like_capture, filter_events_for_protocol
from protocol_infer.visualization.replay import ReplayBuilder
from protocol_infer.visualization.serializer import PEFSMSerializer


@dataclass
class LearnedArtifact:
    artifact_id: str
    protocol: str
    data_dir: str
    pefsm: PEFSM
    train_trace: Trace
    control_flow_pipeline: ControlFlowPipeline
    data_flow_pipeline: DataFlowPipeline
    serializer: PEFSMSerializer
    metrics: Dict = field(default_factory=dict)


class VisualizationService:
    # 交互模式允许读取更多样本，避免被大量极小示例 pcap“挤占”导致会话过少
    _INTERACTIVE_PCAP_BUDGET_BYTES = 64 * 1024 * 1024
    _MAX_SINGLE_PCAP_BYTES = 12 * 1024 * 1024  # 单文件限制 12MB，避免巨型文件卡死训练
    _ALLOWED_PROTOCOLS = {"MODBUS", "S7COMM", "DNP3", "IEC104", "ETHERNET_IP", "MQTT"}
    _TRAINING_PROFILES = {
        # 更快响应：牺牲部分覆盖，优先交互体验
        "fast": {"max_pcaps": 4, "max_sessions": 120, "max_events_per_session": 500},
        # 默认：时间与覆盖折中
        "balanced": {"max_pcaps": 8, "max_sessions": 300, "max_events_per_session": 1500},
        # 覆盖优先：适合离线/长训练
        "thorough": {"max_pcaps": 16, "max_sessions": 800, "max_events_per_session": 0},
    }

    @staticmethod
    def _infer_protocol_from_folder_name(folder_name: str) -> Optional[str]:
        raw = (folder_name or "").strip()
        if not raw:
            return None
        upper = raw.upper()
        # 尽量复用前端逻辑：返回值应与前端 protocolSelect 的 value 对齐
        name_map = {
            "MODBUS": "MODBUS",
            "MODBUSTCP": "MODBUS",
            "S7COMM": "S7COMM",
            "S7": "S7COMM",
            "DNP3": "DNP3",
            "IEC104": "IEC104",
            "IEC60870-104": "IEC104",
            "ETHERNET_IP": "ETHERNET_IP",
            "ETHERNETIP": "ETHERNET_IP",
            "MQTT": "MQTT",
        }
        if upper in name_map:
            return name_map[upper]
        compact = upper.replace("-", "").replace("_", "").replace(" ", "")
        if compact in name_map:
            return name_map[compact]
        if "S7" in compact:
            return "S7COMM"
        if "MODBUS" in compact:
            return "MODBUS"
        if "DNP3" in compact:
            return "DNP3"
        if "IEC60870" in compact or "IEC104" in compact:
            return "IEC104"
        if "ETHERNET" in compact and "IP" in compact:
            return "ETHERNET_IP"
        if "MQTT" in compact:
            return "MQTT"
        return None

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root or os.path.join(os.getcwd(), "Data")
        self.serializer = PEFSMSerializer()
        self.replay_builder = ReplayBuilder()
        self._artifacts: Dict[str, LearnedArtifact] = {}

    def list_datasets(self) -> List[Dict[str, str]]:
        results = []
        if not os.path.isdir(self.data_root):
            return results
        for name in sorted(os.listdir(self.data_root)):
            full = os.path.join(self.data_root, name)
            if os.path.isdir(full):
                proto = self._infer_protocol_from_folder_name(name)
                if proto in self._ALLOWED_PROTOCOLS:
                    results.append(
                        {
                            "name": name,
                            "path": full,
                            "protocol": proto,
                            "mode": "pcap",
                            "label": f"{name} (PCAP)",
                        }
                    )
                    results.append(
                        {
                            "name": name,
                            "path": full,
                            "protocol": proto,
                            "mode": "pcap+synthetic",
                            "label": f"{name} (PCAP+Synthetic)",
                        }
                    )
        for proto in sorted(self._ALLOWED_PROTOCOLS):
            results.append(
                {
                    "name": proto,
                    "path": "__synthetic__",
                    "protocol": proto,
                    "mode": "synthetic",
                    "label": f"{proto} (Synthetic)",
                }
            )
        return results

    def learn_from_dataset(
        self,
        protocol: str,
        data_dir: str,
        max_pcaps: int = 6,
        max_sessions: int = 200,
        profile: str = "balanced",
        test_ratio: float = 0.2,
        seed: int = 42,
        dataset_mode: str = "pcap",
        synthetic_sessions: int = 0,
        synthetic_session_len: int = 20,
        prune_mode: str = "none",
        prune_percentile: int = 70,
    ) -> Dict:
        max_pcaps, max_sessions, max_events_per_session, profile = self._resolve_training_limits(
            profile=profile,
            max_pcaps=max_pcaps,
            max_sessions=max_sessions,
        )

        mode = (dataset_mode or "pcap").strip().lower()
        if mode not in {"pcap", "pcap+synthetic", "synthetic"}:
            mode = "pcap"

        pcap_paths: List[str] = []
        base_data_dir = data_dir

        if mode == "synthetic":
            desired = min(120, max_sessions)
            n_synth = int(synthetic_sessions) if int(synthetic_sessions) > 0 else desired
            sessions_all = generate_synthetic_sessions(
                protocol=protocol,
                n_sessions=n_synth,
                session_len=int(synthetic_session_len),
                seed=seed,
            )
            events: List[MessageEvent] = []
            for evs in sessions_all.values():
                events.extend(evs)
            events.sort(key=lambda e: e.timestamp)
            full_trace = Trace(events=events)
            base_data_dir = "__synthetic__"
        else:
            # 确保 data_dir 是绝对路径
            if not os.path.isabs(data_dir):
                data_dir = os.path.join(self.data_root, data_dir)
            base_data_dir = data_dir

            pcap_paths = self._collect_pcaps(data_dir, max_pcaps)
            if not pcap_paths:
                raise RuntimeError(f"No valid PCAP files found in '{data_dir}'")

            full_trace = self._load_trace(protocol, pcap_paths)

            if mode == "pcap+synthetic":
                desired = min(120, max_sessions)
                sessions_existing = _group_sessions(full_trace)
                need = max(0, desired - len(sessions_existing))
                n_synth = int(synthetic_sessions) if int(synthetic_sessions) > 0 else need
                if n_synth > 0:
                    sessions_synth = generate_synthetic_sessions(
                        protocol=protocol,
                        n_sessions=n_synth,
                        session_len=int(synthetic_session_len),
                        seed=seed + 999,
                    )
                    events: List[MessageEvent] = list(full_trace.events)
                    for evs in sessions_synth.values():
                        events.extend(evs)
                    events.sort(key=lambda e: e.timestamp)
                    full_trace = Trace(events=events)
        train_sessions, _, train_keys = self._split_sessions(full_trace, seed, test_ratio, max_sessions)
        train_sessions = self._cap_session_events(train_sessions, max_events_per_session)
        train_trace = Trace(events=[ev for key in train_keys for ev in train_sessions[key]])
        pefsm, cf, df = self._train_pefsm(train_trace, train_sessions)
        prune_info = self._maybe_prune_pefsm(
            pefsm=pefsm,
            prune_mode=prune_mode,
            prune_percentile=prune_percentile,
        )
        model_json = self.serializer.serialize_model(pefsm, protocol.upper())
        metrics = self._build_metrics(
            protocol=protocol,
            data_dir=base_data_dir,
            pcap_paths=pcap_paths,
            max_pcaps=max_pcaps,
            max_sessions=max_sessions,
            test_ratio=test_ratio,
            seed=seed,
            dataset_mode=mode,
            synthetic_sessions=int(synthetic_sessions),
            synthetic_session_len=int(synthetic_session_len),
            sessions_all=_group_sessions(full_trace) if mode != "pcap" else None,
        )

        artifact_id = uuid.uuid4().hex[:12]
        artifact = LearnedArtifact(
            artifact_id=artifact_id,
            protocol=protocol.upper(),
            data_dir=base_data_dir,
            pefsm=pefsm,
            train_trace=train_trace,
            control_flow_pipeline=cf,
            data_flow_pipeline=df,
            serializer=self.serializer,
            metrics=metrics,
        )
        self._artifacts[artifact_id] = artifact

        return {
            "artifact_id": artifact_id,
            "pcap_files": [os.path.basename(path) for path in pcap_paths] if pcap_paths else [],
            "dataset_mode": mode,
            "training_profile": profile,
            "prune": prune_info,
            "effective_limits": {
                "max_pcaps": max_pcaps,
                "max_sessions": max_sessions,
                "max_events_per_session": max_events_per_session,
            },
            "model": model_json,
            "replay": self.serializer.serialize_replay(self.replay_builder.build(pefsm, train_trace)),
            "metrics": metrics,
        }

    @staticmethod
    def _percentile_threshold(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        xs = sorted(values)
        p = max(0, min(99, int(percentile)))
        idx = max(0, min(len(xs) - 1, int((p / 100) * (len(xs) - 1))))
        return float(xs[idx])

    def _maybe_prune_pefsm(
        self,
        pefsm: PEFSM,
        prune_mode: str,
        prune_percentile: int,
    ) -> Dict:
        mode = (prune_mode or "none").strip().lower()
        if mode in {"none", "off", "false", "0"}:
            return {"mode": "none", "enabled": False}

        before = len(pefsm.transitions)
        if before <= 0:
            return {"mode": "none", "enabled": False}

        if mode == "count":
            threshold = int(self._percentile_threshold([float(t.traverse_count or 0) for t in pefsm.transitions], prune_percentile))
            res = pefsm.prune_transitions(min_count=threshold, min_prob=None, combine="or", preserve_end_reachability=True)
            return {"mode": "count", "enabled": True, "percentile": int(prune_percentile), "min_count": int(threshold), **res}

        if mode == "prob":
            threshold = self._percentile_threshold([float(t.prob or 0.0) for t in pefsm.transitions], prune_percentile)
            res = pefsm.prune_transitions(min_count=None, min_prob=float(threshold), combine="or", preserve_end_reachability=True)
            return {"mode": "prob", "enabled": True, "percentile": int(prune_percentile), "min_prob": float(threshold), **res}

        return {"mode": "none", "enabled": False}

    def _resolve_training_limits(
        self,
        profile: str,
        max_pcaps: int,
        max_sessions: int,
    ) -> Tuple[int, int, int, str]:
        p = (profile or "balanced").strip().lower()
        if p not in self._TRAINING_PROFILES:
            p = "balanced"
        base = self._TRAINING_PROFILES[p]
        # 用户填写值仍保留作用，但不超过 profile 上限（避免单次训练过慢）
        eff_pcaps = max(1, min(int(max_pcaps), int(base["max_pcaps"])))
        eff_sessions = max(2, min(int(max_sessions), int(base["max_sessions"])))
        eff_events_per_session = int(base["max_events_per_session"])
        return eff_pcaps, eff_sessions, eff_events_per_session, p

    @staticmethod
    def _downsample_events(events: List[MessageEvent], cap: int) -> List[MessageEvent]:
        if cap <= 0 or len(events) <= cap:
            return events
        if cap == 1:
            return [events[0]]
        # 保留首尾并均匀抽样，尽量覆盖会话全时段行为
        n = len(events)
        idxs = []
        for i in range(cap):
            pos = int(round(i * (n - 1) / (cap - 1)))
            idxs.append(pos)
        # 去重后补齐
        uniq = []
        seen = set()
        for x in idxs:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        j = 0
        while len(uniq) < cap and j < n:
            if j not in seen:
                uniq.append(j)
                seen.add(j)
            j += 1
        uniq.sort()
        return [events[i] for i in uniq]

    def _cap_session_events(
        self,
        sessions: Dict[SessionKey, List[MessageEvent]],
        max_events_per_session: int,
    ) -> Dict[SessionKey, List[MessageEvent]]:
        if max_events_per_session <= 0:
            return sessions
        capped: Dict[SessionKey, List[MessageEvent]] = {}
        for sk, evs in sessions.items():
            capped[sk] = self._downsample_events(evs, max_events_per_session)
        return capped

    def replay_uploaded_pcap(
        self,
        artifact_id: str,
        pcap_path: str,
    ) -> Dict:
        artifact = self._get_artifact(artifact_id)
        trace = self._build_trace_for_existing_model(pcap_path, artifact)
        steps = self.replay_builder.build(artifact.pefsm, trace)
        return self.serializer.serialize_replay(steps)

    def get_artifact(self, artifact_id: str) -> Dict:
        artifact = self._get_artifact(artifact_id)
        return {
            "artifact_id": artifact.artifact_id,
            "protocol": artifact.protocol,
            "data_dir": artifact.data_dir,
            "model": artifact.serializer.serialize_model(artifact.pefsm, artifact.protocol),
            "replay": artifact.serializer.serialize_replay(self.replay_builder.build(artifact.pefsm, artifact.train_trace)),
            "metrics": artifact.metrics,
        }

    def _build_metrics(
        self,
        protocol: str,
        data_dir: str,
        pcap_paths: Optional[List[str]],
        max_pcaps: int,
        max_sessions: int,
        test_ratio: float,
        seed: int,
        dataset_mode: str = "pcap",
        synthetic_sessions: int = 0,
        synthetic_session_len: int = 20,
        sessions_all: Optional[Dict[SessionKey, List[MessageEvent]]] = None,
    ) -> Dict:
        proto_upper = protocol.upper()
        if proto_upper not in _LABELERS:
            return {
                "dataset": {
                    "protocol": proto_upper,
                    "total_sessions": 0,
                    "train_sessions": 0,
                    "test_sessions": 0,
                    "metric_source": "unavailable",
                },
                "core_metrics": {
                    "session_replay_accuracy": 0.0,
                    "sessions_full_replay_ok": 0.0,
                    "sessions_replay_evaluated": 0.0,
                    "step_replay_accuracy": 0.0,
                    "steps_matched": 0.0,
                    "steps_total": 0.0,
                    "steps_resynced": 0.0,
                    "guard_precision": 0.0,
                    "guard_recall": 0.0,
                    "guard_f1": 0.0,
                    "guard_tp": 0.0,
                    "guard_fp": 0.0,
                    "guard_fn": 0.0,
                },
                "metrics_notes": {
                    "primary_zh": "主指标需在已注册协议下运行完整评估后计算。",
                    "guard_reference_zh": "当前协议未注册监督评估器，无 Guard 参考指标。",
                },
                "note": f"No supervised evaluator registered for protocol '{proto_upper}'.",
            }

        mode = (dataset_mode or "pcap").strip().lower()
        if mode == "pcap":
            eval_result = run_full_evaluation(
                protocol=proto_upper,
                data_dir=data_dir,
                pcap_paths=pcap_paths,
                seed=seed,
                test_ratio=test_ratio,
                max_sessions=max_sessions,
                max_pcaps=max_pcaps,
            )
        else:
            labeler = _LABELERS[proto_upper]
            if mode == "synthetic":
                desired = min(120, max_sessions)
                n_synth = int(synthetic_sessions) if int(synthetic_sessions) > 0 else desired
                sessions_all = generate_synthetic_sessions(
                    protocol=proto_upper,
                    n_sessions=n_synth,
                    session_len=int(synthetic_session_len),
                    seed=seed,
                )
            if sessions_all is None:
                sessions_all = {}
            eval_result = run_full_evaluation_from_sessions(
                protocol=proto_upper,
                sessions_all=sessions_all,
                labeler=labeler,
                seed=seed,
                test_ratio=test_ratio,
                max_sessions=max_sessions,
            )
        enhanced = eval_result.enhanced_efsm or {}
        legacy = eval_result.legacy_efsm or {}

        keys_ok = {"guard_precision", "guard_recall", "guard_f1"}
        metric_source = "unavailable"
        preferred: Dict = {}
        if keys_ok.issubset(enhanced.keys()):
            preferred = enhanced
            metric_source = "enhanced_efsm"
        elif keys_ok.issubset(legacy.keys()):
            preferred = legacy
            metric_source = "legacy_efsm"

        rm = dict(eval_result.replay_metrics or {})

        def _replay_float(key: str, default: float = 0.0) -> float:
            v = rm.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        core_metrics: Dict[str, float] = {
            "session_full_match_rate": _replay_float("session_replay_accuracy"),
            "message_match_accuracy": _replay_float("step_replay_accuracy"),
            "sessions_evaluated": _replay_float("sessions_replay_evaluated"),
            "steps_total": _replay_float("steps_total"),
            "steps_resynced": _replay_float("steps_resynced"),
            "guard_precision": float(preferred.get("guard_precision", 0.0)),
            "guard_recall": float(preferred.get("guard_recall", 0.0)),
            "guard_f1": float(preferred.get("guard_f1", 0.0)),
        }

        # guard_tp/fp/fn 仅 Enhanced Guard 字段级评估才会有。
        # legacy_efsm 时这些键应当缺失，否则会被 UI 当成“0”展示，造成误解。
        if "guard_tp" in preferred or "guard_fp" in preferred or "guard_fn" in preferred:
            core_metrics["guard_tp"] = float(preferred.get("guard_tp", 0.0))
            core_metrics["guard_fp"] = float(preferred.get("guard_fp", 0.0))
            core_metrics["guard_fn"] = float(preferred.get("guard_fn", 0.0))

        return {
            "dataset": {
                "protocol": eval_result.protocol,
                "total_sessions": eval_result.total_sessions,
                "train_sessions": eval_result.train_sessions,
                "test_sessions": eval_result.test_sessions,
                "metric_source": metric_source,
                "replay_eval_on": eval_result.replay_on,
            },
            "core_metrics": core_metrics,
            "metrics_notes": {
                "primary_zh": (
                    "【主指标】session_replay_accuracy = 完整重放成功的会话数 / 参与评估的会话数"
                    "（该会话内每一步都能在已学习的 PEFSM 上匹配转移并走通；与页面回放逻辑一致）。"
                    "replay_eval_on 表示该指标是在 test 集还是 train 集上算的。"
                    "step_replay_accuracy = 匹配成功的消息步数 / 总步数。"
                    "steps_resynced 表示通过“同 symbol 全局重同步”恢复状态的步数。"
                ),
                "guard_reference_zh": (
                    "【参考】guard_precision / guard_recall / guard_f1 为与协议 GT 对比的 Guard「字段级」PRF，"
                    "仅作辅助，不等同于端到端重放成功率。"
                ),
            },
        }

    def _get_artifact(self, artifact_id: str) -> LearnedArtifact:
        if artifact_id not in self._artifacts:
            raise KeyError(f"Unknown artifact_id: {artifact_id}")
        return self._artifacts[artifact_id]

    def _collect_pcaps(self, data_dir: str, max_pcaps: int) -> List[str]:
        if not os.path.isdir(data_dir):
            return []

        candidates: List[Tuple[int, str, str]] = []
        for fn in os.listdir(data_dir):
            if not fn.lower().endswith('.pcap'):
                continue
            full = os.path.join(data_dir, fn)
            if not _looks_like_capture(full):
                continue
            try:
                size = os.path.getsize(full)
                if size > self._MAX_SINGLE_PCAP_BYTES:
                    continue  # 跳过单体过大的文件
            except OSError:
                continue
            candidates.append((size, fn, full))

        # 以“更小文件优先”缩短解析时间，同时尝试在预算内包含更多文件以增加会话多样性
        candidates.sort(key=lambda item: (item[0], item[1].lower()))
        if max_pcaps <= 0:
            return []

        selected: List[str] = []
        total_bytes = 0
        for size, _name, full in candidates:
            if len(selected) >= max_pcaps:
                break
            if selected and total_bytes + size > self._INTERACTIVE_PCAP_BUDGET_BYTES:
                continue
            selected.append(full)
            total_bytes += size

        # 预算过紧时兜底：至少补足若干文件，避免只拿到 1~2 个碎片 pcap
        min_files = min(max_pcaps, 4)
        if len(selected) < min_files:
            chosen = set(selected)
            for _size, _name, full in candidates:
                if len(selected) >= min_files:
                    break
                if full in chosen:
                    continue
                selected.append(full)
                chosen.add(full)

        if not selected and candidates:
            selected.append(candidates[0][2])

        return selected

    def _load_trace(self, protocol: str, pcap_paths: List[str]) -> Trace:
        pipeline = PCAPPipeline()
        events: List[MessageEvent] = []
        for path in pcap_paths:
            trace = pipeline.run(path)
            events.extend(trace.events)
        events = filter_events_for_protocol(protocol, events)
        events.sort(key=lambda e: e.timestamp)
        return Trace(events=events)

    def _split_sessions(
        self,
        trace: Trace,
        seed: int,
        test_ratio: float,
        max_sessions: int,
    ) -> Tuple[Dict[SessionKey, List[MessageEvent]], Dict[SessionKey, List[MessageEvent]], List[SessionKey]]:
        sessions_all = _group_sessions(trace)
        desired_min_sessions = min(120, max_sessions)
        if len(sessions_all) < desired_min_sessions:
            sessions_all = _split_long_sessions(sessions_all, target_sessions=desired_min_sessions)
        keys = list(sessions_all.keys())
        if len(keys) > max_sessions:
            rng = random.Random(seed)
            rng.shuffle(keys)
            keys = keys[:max_sessions]
            sessions_all = {key: sessions_all[key] for key in keys}

        if len(sessions_all) < 2:
            # 与 run_full_evaluation 一致：单会话时按时间切分 train/test，避免“全量训练”与评估流水线不对齐
            only_key = next(iter(sessions_all.keys()))
            evs = sessions_all[only_key]
            split_at = max(1, min(int(len(evs) * (1.0 - test_ratio)), len(evs) - 1))
            train_evs, test_evs_raw = evs[:split_at], evs[split_at:]
            test_key = SessionKey(
                ip1=only_key.ip1,
                port1=only_key.port1,
                ip2=only_key.ip2,
                port2=only_key.port2,
                protocol=only_key.protocol,
                segment_id=int(getattr(only_key, "segment_id", 0)) + 1,
            )
            test_evs = [
                MessageEvent(
                    session_key=test_key,
                    timestamp=e.timestamp,
                    payload=e.payload,
                    direction=e.direction,
                )
                for e in test_evs_raw
            ]
            train_sessions = {only_key: train_evs}
            test_sessions = {test_key: test_evs} if test_evs else {}
            return train_sessions, test_sessions, [only_key]

        train_keys, test_keys = _split_keys(list(sessions_all.keys()), test_ratio, seed)
        train_sessions = {key: sessions_all[key] for key in train_keys}
        test_sessions = {key: sessions_all[key] for key in test_keys}
        return train_sessions, test_sessions, train_keys

    def _train_pefsm(
        self,
        train_trace: Trace,
        train_sessions: Dict[SessionKey, List[MessageEvent]],
    ) -> Tuple[PEFSM, ControlFlowPipeline, DataFlowPipeline]:
        cf = ControlFlowPipeline(use_apriori=True)
        fsm = cf.run(train_trace)
        df = DataFlowPipeline(abstractor=cf.abstractor, symbol_featureer=cf.featureer)
        efsm = df.run(
            trace=train_trace,
            fsm=fsm,
            sessions=train_sessions,
            precomputed_sess_features=cf.get_sess_features(),
            apriori_positions=cf.get_apriori_positions(),
            apriori_static_items=cf.get_apriori_static_items(),
        )
        pefsm = ProbabilisticPipeline().run(efsm=efsm, trace=train_trace)
        return pefsm, cf, df

    def _build_trace_for_existing_model(self, pcap_path: str, artifact: LearnedArtifact) -> Trace:
        trace = PCAPPipeline().run(pcap_path)
        sessions = _group_sessions(trace)
        artifact.data_flow_pipeline.run(
            trace=trace,
            fsm=artifact.pefsm,
            sessions=sessions,
            precomputed_sess_features=None,
            apriori_positions=getattr(artifact.data_flow_pipeline.feature_processor, "apriori_positions", None),
            apriori_static_items=getattr(artifact.data_flow_pipeline.feature_processor, "apriori_static_items", None),
        )
        return trace
