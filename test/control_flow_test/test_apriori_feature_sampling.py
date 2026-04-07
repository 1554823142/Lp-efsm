from protocol_infer.control_flow_layer.features.apriori_feature_extraction import (
    AprioriFeatureExtraction,
)
from protocol_infer.core.datamodel.event import Direction, MessageEvent
from protocol_infer.core.datamodel.session import SessionKey


def _event(idx: int) -> MessageEvent:
    return MessageEvent(
        session_key=SessionKey("1.1.1.1", 1025, "2.2.2.2", 102, "TCP"),
        timestamp=float(idx),
        payload=bytes([idx % 256, (idx * 3) % 256, 0x32, 0x01]),
        direction=Direction.C2S,
    )


def test_sample_events_caps_size_and_is_deterministic():
    events = [_event(i) for i in range(100)]

    sampled_a = AprioriFeatureExtraction._sample_events(events, limit=10, seed=7)
    sampled_b = AprioriFeatureExtraction._sample_events(events, limit=10, seed=7)

    assert len(sampled_a) == 10
    assert [ev.timestamp for ev in sampled_a] == [ev.timestamp for ev in sampled_b]
    assert [ev.timestamp for ev in sampled_a] == sorted(ev.timestamp for ev in sampled_a)


def test_from_events_uses_sampled_subset_for_mining(monkeypatch):
    events = [_event(i) for i in range(50)]
    calls = {}

    def fake_mine(self, mining_events):
        calls["mine_len"] = len(mining_events)
        return []

    def fake_discover_field_groups(cls, events_all, **kwargs):
        calls["group_len"] = len(events_all)
        return [], {}

    monkeypatch.setattr(
        "protocol_infer.apriori.miners.StaticFieldMiner.mine",
        fake_mine,
    )
    monkeypatch.setattr(
        AprioriFeatureExtraction,
        "_discover_field_groups",
        classmethod(fake_discover_field_groups),
    )

    AprioriFeatureExtraction.from_events(events, max_mining_events=12)

    assert calls["mine_len"] == 12
    assert calls["group_len"] == 12
