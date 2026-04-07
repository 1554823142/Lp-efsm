from typing import List

from protocol_infer.core.datamodel.event import MessageEvent


def _norm_byte(value: int) -> float:
    return max(0.0, min(float(value) / 255.0, 1.0))


def _norm_int(value: int, scale: int) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(float(value) / float(scale), 1.0))


def _header_bytes(payload: bytes, width: int = 8) -> List[float]:
    return [_norm_byte(payload[i]) if i < len(payload) else 0.0 for i in range(width)]


def _mqtt_remaining_length(payload: bytes) -> int:
    multiplier = 1
    value = 0
    for i in range(1, min(len(payload), 5)):
        encoded = payload[i]
        value += (encoded & 0x7F) * multiplier
        if (encoded & 0x80) == 0:
            return value
        multiplier *= 128
    return 0


def infer_protocol_family(event: MessageEvent) -> str:
    payload = event.payload or b""
    ports = {int(event.session_key.port1), int(event.session_key.port2)}

    if len(payload) >= 8 and payload[2:4] == b"\x00\x00" and (502 in ports or payload[6] <= 247):
        return "MODBUS"
    if len(payload) >= 4 and payload[0] == 0x03 and payload[1] == 0x00:
        if 102 in ports or (len(payload) >= 18 and payload[7] == 0x32):
            return "S7COMM"
    if len(payload) >= 4 and payload[0] == 0x05 and payload[1] == 0x64:
        return "DNP3"
    if len(payload) >= 6 and payload[0] == 0x68:
        return "IEC104"
    if 44818 in ports or (len(payload) >= 24 and int.from_bytes(payload[0:2], "little") in (0x0065, 0x006F, 0x0070, 0x0072)):
        return "ETHERNET_IP"
    mqtt_type = (payload[0] >> 4) if payload else 0
    if 1883 in ports or 8883 in ports or (payload and 1 <= mqtt_type <= 14):
        return "MQTT"
    return "GENERIC"


def extract_protocol_semantic_features(event: MessageEvent) -> List[float]:
    payload = event.payload or b""
    proto = infer_protocol_family(event)

    flags = [
        1.0 if proto == "MODBUS" else 0.0,
        1.0 if proto == "S7COMM" else 0.0,
        1.0 if proto == "DNP3" else 0.0,
        1.0 if proto == "IEC104" else 0.0,
        1.0 if proto == "ETHERNET_IP" else 0.0,
        1.0 if proto == "MQTT" else 0.0,
    ]
    semantic = [0.0] * 6

    if proto == "MODBUS" and len(payload) >= 8:
        fc = payload[7]
        semantic[0] = _norm_byte(fc)
        semantic[1] = _norm_byte(payload[6])
        semantic[2] = _norm_int(int.from_bytes(payload[4:6], "big"), 260)
        if len(payload) >= 12:
            semantic[3] = _norm_int(int.from_bytes(payload[8:10], "big"), 65535)
            semantic[4] = _norm_int(int.from_bytes(payload[10:12], "big"), 65535)
        semantic[5] = 1.0 if (fc & 0x80) else 0.0

    elif proto == "S7COMM" and len(payload) >= 17:
        semantic[0] = _norm_byte(payload[8])
        semantic[1] = _norm_byte(payload[17]) if len(payload) > 17 else 0.0
        semantic[2] = _norm_int(int.from_bytes(payload[11:13], "big"), 65535)
        semantic[3] = _norm_int(int.from_bytes(payload[13:15], "big"), 1024)
        semantic[4] = _norm_int(int.from_bytes(payload[15:17], "big"), 1024)
        semantic[5] = _norm_int(int.from_bytes(payload[2:4], "big"), 65535)

    elif proto == "DNP3" and len(payload) >= 12:
        ctrl = payload[3]
        semantic[0] = _norm_int(ctrl & 0x0F, 15)
        semantic[1] = 1.0 if (ctrl & 0x40) else 0.0
        semantic[2] = 1.0 if (ctrl & 0x80) else 0.0
        semantic[3] = _norm_byte(payload[2])
        semantic[4] = _norm_int(payload[10] & 0x3F, 63)
        semantic[5] = _norm_int(payload[11] & 0x1F, 31)

    elif proto == "IEC104" and len(payload) >= 6:
        c1, c2, c3, c4 = payload[2], payload[3], payload[4], payload[5]
        frame_code = 0.0 if (c1 & 0x01) == 0 else (0.5 if (c1 & 0x03) == 1 else 1.0)
        semantic[0] = frame_code
        semantic[1] = _norm_byte(payload[6]) if len(payload) > 6 else 0.0
        semantic[2] = _norm_int((((c2 << 8) | c1) >> 1), 32767)
        semantic[3] = _norm_int((((c4 << 8) | c3) >> 1), 32767)
        semantic[4] = _norm_int(int.from_bytes(payload[8:10], "little"), 65535) if len(payload) >= 10 else 0.0
        semantic[5] = _norm_int(int.from_bytes(payload[10:12], "little"), 65535) if len(payload) >= 12 else 0.0

    elif proto == "ETHERNET_IP" and len(payload) >= 24:
        command = int.from_bytes(payload[0:2], "little")
        encap_len = int.from_bytes(payload[2:4], "little")
        status = int.from_bytes(payload[8:12], "little")
        item_count = int.from_bytes(payload[30:32], "little") if len(payload) >= 32 else 0
        semantic[0] = _norm_int(command, 65535)
        semantic[1] = _norm_int(encap_len, 4096)
        semantic[2] = 1.0 if int.from_bytes(payload[4:8], "little") != 0 else 0.0
        semantic[3] = 1.0 if status == 0 else _norm_int(status, 65535)
        semantic[4] = _norm_int(item_count, 8)
        semantic[5] = _norm_byte(payload[40]) if len(payload) > 40 else 0.0

    elif proto == "MQTT" and payload:
        first = payload[0]
        semantic[0] = _norm_int(first >> 4, 15)
        semantic[1] = _norm_int((first >> 1) & 0x03, 3)
        semantic[2] = 1.0 if (first & 0x08) else 0.0
        semantic[3] = 1.0 if (first & 0x01) else 0.0
        semantic[4] = _norm_int(_mqtt_remaining_length(payload), 4096)
        semantic[5] = _norm_int(int.from_bytes(payload[2:4], "big"), 4096) if len(payload) >= 4 else 0.0

    return _header_bytes(payload) + flags + semantic
