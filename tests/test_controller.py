from dataclasses import replace

import numpy as np
import pytest

from hotshardgnn.controller import (Controller, ControllerConfig, load_cost,
                                   remote_cost, worker_loads)


def fixture(seed=8):
    rng = np.random.default_rng(seed)
    requests = rng.integers(0, 6, (80, 4)).astype(float)
    requests[:8, 1] += np.arange(30, 110, 10)
    owners = np.arange(80) % 4
    owners[:8] = 0
    sizes = rng.integers(16, 128, 80)
    consumers = np.full(4, 500.0)
    cfg = ControllerConfig(heat_percentile=70, min_utility=0, gamma=0.001, epsilon=.3)
    controller = Controller(80, 4, cfg)
    for window in range(3):
        controller.observe(requests, window)
        controller.calibrate(requests, owners, sizes, consumers)
    controller.freeze_scales()
    return controller, requests, owners, sizes, consumers


def test_remote_gain_matches_full_objective():
    c, requests, owners, sizes, _ = fixture()
    for node in range(20):
        target = (owners[node] + 1) % 4
        changed = owners.copy()
        changed[node] = target
        expected = sizes[node] * (requests[node, target] - requests[node, owners[node]])
        assert remote_cost(requests, owners, sizes) - remote_cost(requests, changed, sizes) == expected


def test_ema_and_zero_budget():
    c = Controller(2, 2)
    c.observe(np.array([[10, 0], [0, 20]]), 0)
    np.testing.assert_allclose(c.forecast, [[7, 0], [0, 14]])
    c.observe(np.zeros((2, 2)), 1)
    np.testing.assert_allclose(c.forecast, [[2.1, 0], [0, 4.2]])
    assert c.plan(None, None, None, None, 0, 1) == []


def test_budget_feasibility_and_objective_property():
    moved = 0
    for seed in range(15):
        c, requests, owners, sizes, consumers = fixture(seed)
        for budget in (1, 64, 200, 1000):
            moves = c.plan(requests, owners, sizes, consumers, budget, 3)
            assert sum(m.size for m in moves) <= budget
            assert len({m.node for m in moves}) == len(moves)
            changed = owners.copy()
            for m in moves:
                assert owners[m.node] == m.source
                changed[m.node] = m.target
            before = worker_loads(owners, c.forecast.sum(1), consumers)
            after = worker_loads(changed, c.forecast.sum(1), consumers)
            np.testing.assert_allclose(before.sum(), after.sum())
            for m in moves:
                assert after[m.target] <= (1 + c.config.epsilon) * after.mean() + 1e-8
            gain = ((remote_cost(c.forecast, owners, sizes) - remote_cost(c.forecast, changed, sizes)) / c.r0
                    + (load_cost(before) - load_cost(after)) / c.l0
                    - c.config.gamma * sum(m.size for m in moves) / budget)
            assert gain >= -1e-8
            moved += len(moves)
    assert moved > 0


def test_cooldown_only_starts_on_commit():
    c, requests, owners, sizes, consumers = fixture()
    moves = c.plan(requests, owners, sizes, consumers, 1000, 3)
    assert moves
    assert (c.last_move < 0).all()
    c.commit(moves, 3)
    for window in (4, 5, 6):
        planned = c.plan(requests, owners, sizes, consumers, 1000, window)
        assert not ({m.node for m in planned} & {m.node for m in moves})


def test_cold_records_expire_and_invalid_input():
    c = Controller(1, 2)
    c.observe(np.ones((1, 2)), 0)
    c.observe(np.zeros((1, 2)), 7)
    assert not c.forecast.any()
    with pytest.raises(ValueError):
        c.observe(np.array([[-1, 0]]), 8)
    with pytest.raises(ValueError):
        replace(ControllerConfig(), gamma=-1)
