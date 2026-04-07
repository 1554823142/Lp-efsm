"""
模块一：字段级完整提取

在每条 trace 记录里附加完整的字段键值对（fields）、状态快照（state_snap）
和与上一包的变量 diff（state_diff）。state_diff 是 action 副作用的直接观测证据。

与现有 FeatureProcessor 的区别：
  - FeatureProcessor.extract_vars() 只返回 apriori 选出的少量字节位变量。
  - FieldExtractor 额外解析已知协议的语义字段（fc、tid、length、...），
    并跟踪逐包的变量变化，输出 state_diff。
  - 未知协议退化为纯字节位向量，不丢失已有能力。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from protocol_infer.core.datamodel.event import Direction, MessageEvent
from protocol_infer.core.datamodel.session import SessionKey


# ---------------------------------------------------------------------------
# 协议字段解析器（可扩展）
# ---------------------------------------------------------------------------

class _ProtocolFieldParser:
    """子类实现 parse()，返回协议语义字段字典。"""

    name: str = "unknown"

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        raise NotImplementedError


class _ModbusFieldParser(_ProtocolFieldParser):
    """
    Modbus TCP MBAP + PDU 字段：
      txn_id, proto_id, length, unit_id, fc, [data_bytes]
    """
    name = "MODBUS"

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        result: Dict[str, Any] = {"direction": direction.name}
        if len(payload) < 8:
            return result
        txn_id = int.from_bytes(payload[0:2], "big")
        proto_id = int.from_bytes(payload[2:4], "big")
        length = int.from_bytes(payload[4:6], "big")
        unit_id = payload[6]
        fc = payload[7]
        result.update({
            "txn_id": txn_id,
            "proto_id": proto_id,
            "length": length,
            "unit_id": unit_id,
            "fc": fc,
        })
        # 请求：读/写寄存器带起始地址和数量
        if direction == Direction.C2S and fc in (0x01, 0x02, 0x03, 0x04, 0x05, 0x06) and len(payload) >= 12:
            result["start_addr"] = int.from_bytes(payload[8:10], "big")
            result["quantity"] = int.from_bytes(payload[10:12], "big")
        # 响应：带字节计数
        if direction == Direction.S2C and fc in (0x01, 0x02, 0x03, 0x04) and len(payload) >= 9:
            result["byte_count"] = payload[8]
        # 异常：fc | 0x80
        if fc & 0x80:
            result["is_exception"] = 1
            if len(payload) >= 9:
                result["exception_code"] = payload[8]
        return result


class _IEC104FieldParser(_ProtocolFieldParser):
    """
    IEC 60870-5-104 APCI 字段：
      start, apdu_len, ctrl_bytes, type_id, vsq, cause_of_tx, common_addr
    """
    name = "IEC104"

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        result: Dict[str, Any] = {"direction": direction.name}
        if len(payload) < 6 or payload[0] != 0x68:
            return result
        apdu_len = payload[1]
        c1, c2, c3, c4 = payload[2], payload[3], payload[4], payload[5]
        result.update({"apdu_len": apdu_len, "ctrl1": c1, "ctrl2": c2, "ctrl3": c3, "ctrl4": c4})

        # I 格式：最低位 == 0
        frame_type = "I" if (c1 & 0x01) == 0 else ("S" if (c1 & 0x03) == 1 else "U")
        result["frame_type"] = frame_type

        if frame_type == "I" and apdu_len >= 4 and len(payload) >= 10:
            ns = ((c2 << 8) | c1) >> 1        # 发送序列号
            nr = ((c4 << 8) | c3) >> 1        # 接收序列号
            result.update({"ns": ns, "nr": nr})

        if frame_type == "I" and len(payload) >= 7:
            type_id = payload[6]
            result["type_id"] = type_id
        if len(payload) >= 8:
            result["vsq"] = payload[7]
        if len(payload) >= 10:
            result["cause_of_tx"] = int.from_bytes(payload[8:10], "little")
        if len(payload) >= 12:
            result["common_addr"] = int.from_bytes(payload[10:12], "little")
        return result


class _DNP3FieldParser(_ProtocolFieldParser):
    """
    DNP3 数据链路帧字段：
      start, length, ctrl, dest, src, func, prm
    """
    name = "DNP3"

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        result: Dict[str, Any] = {"direction": direction.name}
        if len(payload) < 10 or payload[0] != 0x05 or payload[1] != 0x64:
            return result
        length = payload[2]
        ctrl = payload[3]
        dest = int.from_bytes(payload[4:6], "little")
        src = int.from_bytes(payload[6:8], "little")
        func = ctrl & 0x0F
        prm = 1 if (ctrl & 0x40) else 0
        dir_bit = 1 if (ctrl & 0x80) else 0
        result.update({
            "length": length,
            "ctrl": ctrl,
            "dest": dest,
            "src": src,
            "func": func,
            "prm": prm,
            "dir_bit": dir_bit,
        })
        return result


class _EtherNetIPFieldParser(_ProtocolFieldParser):
    """
    EtherNet/IP 封装头字段：
      command, encap_len, session_handle, status, options, interface_handle, timeout, item_count, service
    """
    name = "ETHERNET_IP"

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        result: Dict[str, Any] = {"direction": direction.name}
        if len(payload) < 24:
            return result
        command = int.from_bytes(payload[0:2], "little")
        encap_len = int.from_bytes(payload[2:4], "little")
        session_handle = int.from_bytes(payload[4:8], "little")
        status = int.from_bytes(payload[8:12], "little")
        options = int.from_bytes(payload[20:24], "little")
        result.update({
            "command": command,
            "encap_len": encap_len,
            "session_handle": session_handle,
            "status": status,
            "options": options,
        })
        if len(payload) >= 32:
            result["interface_handle"] = int.from_bytes(payload[24:28], "little")
            result["timeout"] = int.from_bytes(payload[28:30], "little")
            result["item_count"] = int.from_bytes(payload[30:32], "little")
        if len(payload) > 40:
            result["service"] = payload[40]
        return result


class _GenericFieldParser(_ProtocolFieldParser):
    """
    通用解析器：提取长度、熵值和前 N 个字节位。
    用于未知协议或回退情形。
    """
    name = "GENERIC"

    def __init__(self, byte_positions: Optional[List[int]] = None):
        self._positions = byte_positions or list(range(min(16, 256)))

    def parse(self, payload: bytes, direction: Direction) -> Dict[str, Any]:
        import math
        result: Dict[str, Any] = {"direction": direction.name, "pkt_len": len(payload)}
        # 字节熵
        if payload:
            counts: Dict[int, int] = {}
            for b in payload:
                counts[b] = counts.get(b, 0) + 1
            ent = 0.0
            for c in counts.values():
                p = c / len(payload)
                ent -= p * math.log(p + 1e-12)
            result["entropy"] = round(ent, 4)
        for pos in self._positions:
            if pos < len(payload):
                result[f"b{pos}"] = payload[pos]
        return result


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

_PARSERS: Dict[str, _ProtocolFieldParser] = {
    "MODBUS": _ModbusFieldParser(),
    "MODBUSTCP": _ModbusFieldParser(),
    "IEC104": _IEC104FieldParser(),
    "IEC60870-104": _IEC104FieldParser(),
    "DNP3": _DNP3FieldParser(),
    "ETHERNET_IP": _EtherNetIPFieldParser(),
    "ETHERNETIP": _EtherNetIPFieldParser(),
}
_GENERIC_PARSER = _GenericFieldParser()


def _get_parser(protocol: str) -> _ProtocolFieldParser:
    return _PARSERS.get(protocol.upper(), _GENERIC_PARSER)


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    """
    单包的完整提取结果。

    Attributes
    ----------
    event:        原始 MessageEvent
    msg_type:     协议标注的消息类型（与现有 ProtocolLabeler 输出对齐）
    fields:       协议语义字段键值对（模块一新增）
    state_snap:   截至本包的累计变量快照（模块一新增）
    state_diff:   与上一包的变量 diff（模块一新增，action 观测证据）
    abstract_sym: 经 abstractor 映射的抽象符号（如 C0/C1，由 run_evaluation 填入）
    abstract_vars: FeatureProcessor.extract_vars 输出的字节位变量（由 run_evaluation 填入）
    """
    event: MessageEvent
    msg_type: str
    fields: Dict[str, Any] = field(default_factory=dict)
    state_snap: Dict[str, Any] = field(default_factory=dict)
    state_diff: Dict[str, Any] = field(default_factory=dict)
    abstract_sym: Optional[str] = None
    abstract_vars: Optional[Dict[str, float]] = None


@dataclass
class SessionTrace:
    """
    一条会话的提取结果列表。
    """
    session_key: SessionKey
    records: List[TraceRecord] = field(default_factory=list)

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# 字段提取器
# ---------------------------------------------------------------------------

class FieldExtractor:
    """
    对一个或多个 MessageEvent 序列做字段级提取，输出 TraceRecord 列表。

    参数
    ----
    protocol:       协议名称（MODBUS / IEC104 / DNP3 / 通用）
    labeler:        可选，接受 MessageEvent 返回 msg_type 字符串的可调用对象。
                    若为 None，则从 fields 中推断 msg_type（fc / type_id / func）。
    byte_positions: 通用解析器的字节位列表（仅在未知协议时生效）
    """

    def __init__(
        self,
        protocol: str = "GENERIC",
        labeler=None,
        byte_positions: Optional[List[int]] = None,
    ):
        self._parser: _ProtocolFieldParser = _get_parser(protocol)
        if isinstance(self._parser, _GenericFieldParser) and byte_positions:
            self._parser = _GenericFieldParser(byte_positions)
        self._labeler = labeler

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract_session(
        self,
        events: Sequence[MessageEvent],
        session_key: Optional[SessionKey] = None,
    ) -> SessionTrace:
        """
        对单条会话的事件序列提取字段，返回 SessionTrace。
        """
        sk = session_key or (events[0].session_key if events else SessionKey("?", 0, "?", 0, "?"))
        session_trace = SessionTrace(session_key=sk)
        prev_snap: Dict[str, Any] = {}

        for ev in events:
            fields = self._parser.parse(ev.payload or b"", ev.direction)
            msg_type = self._infer_msg_type(ev, fields)

            # 累计快照：将本包字段合并进状态（保留跨包变量，如事务 ID、序列号）
            snap = dict(prev_snap)
            snap.update(fields)

            # diff：仅记录本包与上一包不同的字段
            diff: Dict[str, Any] = {}
            for k, v in fields.items():
                if k not in prev_snap or prev_snap[k] != v:
                    diff[k] = v

            record = TraceRecord(
                event=ev,
                msg_type=msg_type,
                fields=fields,
                state_snap=dict(snap),
                state_diff=diff,
            )
            session_trace.records.append(record)
            prev_snap = snap

        return session_trace

    def extract_sessions(
        self,
        sessions: Dict[SessionKey, List[MessageEvent]],
    ) -> Dict[SessionKey, SessionTrace]:
        """
        批量提取多条会话，返回 {SessionKey: SessionTrace}。
        """
        return {sk: self.extract_session(evs, sk) for sk, evs in sessions.items()}

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _infer_msg_type(self, ev: MessageEvent, fields: Dict[str, Any]) -> str:
        if self._labeler is not None:
            return self._labeler(ev)
        # 回退：从字段推断
        dir_tag = "c2s" if ev.direction == Direction.C2S else "s2c"
        if "fc" in fields:
            return f"fc_{fields['fc']:02x}_{dir_tag}"
        if "type_id" in fields:
            return f"type_{fields['type_id']:02x}_{dir_tag}"
        if "func" in fields:
            return f"func_{fields['func']:01x}_{dir_tag}"
        if "command" in fields:
            return f"cmd_{fields['command']:04x}_{dir_tag}"
        return f"unknown_{dir_tag}"


# ---------------------------------------------------------------------------
# 工具函数：将 SessionTrace 转换回 (symbol, vars_dict) 格式
# ---------------------------------------------------------------------------

def session_trace_to_pairs(
    session_trace: SessionTrace,
    use_snap: bool = False,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    将 SessionTrace 转换为 [(msg_type, fields)] 列表，
    与 efsm_evaluator / EFSMevaluator 期望的 sequences 格式兼容。

    参数
    ----
    use_snap:   True -> 使用 state_snap（累计变量）；False -> 使用 fields（本包字段）
    """
    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for rec in session_trace.records:
        vars_dict = {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in (rec.state_snap if use_snap else rec.fields).items()
                     if k != "direction"}
        pairs.append((rec.msg_type, vars_dict))
    return pairs


def build_pairs_from_sessions(
    sessions: Dict[SessionKey, List[MessageEvent]],
    protocol: str = "GENERIC",
    labeler=None,
) -> Dict[SessionKey, List[Tuple[str, Dict[str, Any]]]]:
    """
    一步到位：从原始 sessions 生成 EFSMevaluator 所需的 sequences 格式。
    """
    extractor = FieldExtractor(protocol=protocol, labeler=labeler)
    session_traces = extractor.extract_sessions(sessions)
    return {sk: session_trace_to_pairs(st) for sk, st in session_traces.items()}
