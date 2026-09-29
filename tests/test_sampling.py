from types import SimpleNamespace

import numpy as np

from hotshardgnn.sampling import TemporalSampler


def test_equal_and_future_timestamps_excluded():
    data = SimpleNamespace(src=np.array([0, 0, 0]), dst=np.array([1, 2, 3]),
                           times=np.array([1., 2., 2.]), num_nodes=4, meta={"time_scale": 1.})
    sampler = TemporalSampler(data)
    tree = sampler.sample([0], [2.], [3, 2])
    assert tree.valid.tolist() == [[True, False, False]]
    assert tree.edge_ids[0, 0] == 0
    assert tree.children.nodes[1] == 1
    assert tree.children.times[1] == 1
    assert not tree.children.valid[1].any()


def test_future_events_do_not_change_past_sample():
    common = dict(num_nodes=3, meta={"time_scale": 1.})
    a = SimpleNamespace(src=np.array([0]), dst=np.array([1]), times=np.array([1.]), **common)
    b = SimpleNamespace(src=np.array([0, 0]), dst=np.array([1, 2]), times=np.array([1., 4.]), **common)
    first = TemporalSampler(a).sample([0], [3.], [2])
    second = TemporalSampler(b).sample([0], [3.], [2])
    np.testing.assert_array_equal(first.children.nodes, second.children.nodes)
    np.testing.assert_array_equal(first.valid, second.valid)
