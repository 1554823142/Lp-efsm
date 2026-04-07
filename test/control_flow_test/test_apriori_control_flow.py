from protocol_infer.control_flow_layer.features.apriori_feature_extraction import AprioriFeatureExtraction
from protocol_infer.core.datamodel.event import Direction, MessageEvent
from protocol_infer.core.datamodel.session import SessionKey


def _event(idx: int, marker: int) -> MessageEvent:
    payload = bytearray(24)
    payload[0] = 0x03
    payload[1] = 0x00
    payload[7] = 0x32
    payload[17] = marker
    payload[18] = idx % 4
    return MessageEvent(
        session_key=SessionKey("1.1.1.1", 1025 + idx, "2.2.2.2", 102, "TCP"),
        timestamp=float(idx),
        payload=bytes(payload),
        direction=Direction.C2S if idx % 2 == 0 else Direction.S2C,
    )


def test_apriori_keeps_deeper_discriminative_positions():
    events = [_event(i, 0x04 if i < 30 else 0x05) for i in range(60)]
    feat = AprioriFeatureExtraction.from_events(events, max_positions=24, max_mining_events=60)
    assert 17 in feat.positions


def test_informative_positions_ignore_pure_constants():
    events = [_event(i, 0x04 if i < 30 else 0x05) for i in range(60)]
    ranked = AprioriFeatureExtraction._select_informative_positions(events, 24)
    ranked_pos = [pos for pos, _score in ranked]
    assert 17 in ranked_pos
    assert 0 not in ranked_pos
