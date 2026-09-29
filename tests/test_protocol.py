import pytest

from hotshardgnn.protocol import VersionedStore


def test_logged_updates_handoff_and_delayed_ack():
    store = VersionedStore(3)
    store.add(9, 0, {"feature": [1, 2], "memory": 0, "optimizer": [0.0]})
    store.begin(9, 2)
    store.write(9, 0, 0, 1, "memory", 10)
    store.write(9, 0, 0, 2, "optimizer", [0.25])
    store.publish(9)
    assert store.read(9, 1) == store.read(9, 0)
    with pytest.raises(ValueError, match="Stale"):
        store.write(9, 0, 0, 3, "memory", 100)
    store.write(9, 2, 1, 3, "memory", 11)
    assert store.read(9, 0)["memory"] == 10
    with pytest.raises(ValueError, match="exactly once"):
        store.write(9, 2, 1, 3, "memory", 11)
    for rank in (0, 1):
        store.acknowledge(9, 0, rank)
    assert store.read(9, 0)["memory"] == 10
    store.acknowledge(9, 0, 2)
    with pytest.raises(KeyError):
        store.read(9, 0)


@pytest.mark.parametrize("failure", ["target_loss", "log_overflow"])
def test_abort_keeps_committed_source(failure):
    store = VersionedStore(2, log_capacity=0 if failure == "log_overflow" else 4)
    store.add(1, 0, {"memory": 1})
    store.begin(1, 1)
    store.write(1, 0, 0, 1, "memory", 2)
    if failure == "target_loss":
        store.abort(1)
    assert store.records[1].owner == 0
    assert store.read(1, 0) == {"memory": 2}
    with pytest.raises(ValueError, match="No prepared"):
        store.publish(1)
