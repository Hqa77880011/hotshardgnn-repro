"""Equations (1)--(6) and Algorithm 1 of the supplied HotShardGNN manuscript."""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np


@dataclass(frozen=True)
class ControllerConfig:
    rho: float = 0.7
    heat_percentile: float = 90.0
    external_threshold: float = 0.30
    destination_share: float = 0.10
    cooldown: int = 3
    epsilon: float = 0.10
    reserve: float = 0.15
    min_utility: float = 0.01
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.05
    eta: float = 0.0
    one_swap: bool = True

    def __post_init__(self):
        values = [v for v in vars(self).values() if not isinstance(v, bool)]
        if not all(np.isfinite(values)):
            raise ValueError("Controller parameters must be finite")
        if not 0 <= self.rho <= 1 or not 0 <= self.reserve < 1:
            raise ValueError("rho must be in [0,1], reserve in [0,1)")
        if not 0 <= self.heat_percentile <= 100:
            raise ValueError("heat_percentile must be in [0,100]")
        if not all(0 <= x <= 1 for x in (self.external_threshold, self.destination_share)):
            raise ValueError("Request thresholds must be in [0,1]")
        if min(self.cooldown, self.epsilon, self.min_utility,
               self.alpha, self.beta, self.gamma, self.eta) < 0:
            raise ValueError("Weights, slack, cooldown and minimum utility must be nonnegative")


@dataclass(frozen=True)
class Move:
    node: int
    source: int
    target: int
    size: int
    utility: float


def remote_cost(requests, owners, sizes):
    """State-weighted remote-demand proxy; this is not measured network traffic."""
    local = requests[np.arange(len(owners)), owners]
    return float(np.dot(sizes, requests.sum(axis=1) - local))


def load_cost(loads):
    """Squared coefficient of variation, defined as zero for zero total work."""
    mean = float(np.mean(loads))
    return float(np.mean(((loads - mean) / mean) ** 2)) if mean > 0 else 0.0


def worker_loads(owners, service, consumer_work):
    return np.asarray(consumer_work, dtype=float) + np.bincount(
        owners, weights=service, minlength=len(consumer_work)
    )


def heat_score(signals):
    """Sum median-normalized telemetry columns; zero medians use scale one."""
    signals = np.asarray(signals, dtype=float)
    medians = np.median(signals, axis=0)
    return (signals / np.where(medians > 0, medians, 1.0)).sum(axis=1)


class Controller:
    """Forecast, calibrate and plan moves; call commit only after publication."""

    def __init__(self, nodes, workers, config=None):
        self.config = config or ControllerConfig()
        self.forecast = np.zeros((nodes, workers), dtype=float)
        self.last_move = np.full(nodes, -10**9, dtype=np.int64)
        self.last_active = np.full(nodes, -10**9, dtype=np.int64)
        self.calibration = []
        self.r0 = self.l0 = None

    def observe(self, requests, window):
        requests = np.asarray(requests, dtype=float)
        if requests.shape != self.forecast.shape or not np.isfinite(requests).all() or (requests < 0).any():
            raise ValueError("requests must be a finite nonnegative [nodes, workers] matrix")
        active = requests.sum(axis=1) > 0
        self.last_active[active] = window
        self.forecast *= 1 - self.config.rho
        self.forecast += self.config.rho * requests
        self.forecast[window - self.last_active > max(1, 2 * self.config.cooldown)] = 0

    def calibrate(self, requests, owners, sizes, consumer_work):
        self.calibration.append((remote_cost(requests, owners, sizes), load_cost(
            worker_loads(owners, requests.sum(axis=1), consumer_work))))

    def freeze_scales(self):
        if not self.calibration:
            raise ValueError("At least one static calibration window is required")
        r0, l0 = np.median(self.calibration, axis=0)
        # A perfectly balanced calibration has zero dispersion; use explicit floors.
        self.r0, self.l0 = max(float(r0), 1.0), max(float(l0), 1e-6)

    def plan(self, requests, owners, sizes, consumer_work, budget, window,
             signals=None, random_seed=None):
        """Return feasible moves without changing ownership or cooldown history."""
        if budget < 0 or int(budget) != budget:
            raise ValueError("budget must be a nonnegative integer number of bytes")
        if budget == 0:
            return []
        if self.r0 is None:
            raise ValueError("freeze_scales must be called before selection")
        c = self.config
        owners = np.asarray(owners, dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.int64)
        requests = np.asarray(requests, dtype=float)
        if requests.shape != self.forecast.shape or sizes.shape != owners.shape:
            raise ValueError("Mismatched node/worker dimensions")
        if (sizes <= 0).any() or (owners < 0).any() or (owners >= requests.shape[1]).any():
            raise ValueError("State sizes must be positive and owners in range")
        service = self.forecast.sum(axis=1)
        initial = worker_loads(owners, service, consumer_work)
        loads = initial.copy()
        mean = float(loads.mean())
        totals = requests.sum(axis=1)
        active = totals > 0
        if not active.any() or mean == 0:
            return []
        heat = np.zeros(len(owners))
        signals = totals[:, None] if signals is None else np.asarray(signals)
        heat[active] = heat_score(signals[active])
        threshold = np.percentile(heat[active], c.heat_percentile)
        external = np.zeros(len(owners))
        external[active] = 1 - requests[np.arange(len(owners)), owners][active] / totals[active]
        candidates = np.flatnonzero(
            active & (heat > threshold) & (external > c.external_threshold)
            & (initial[owners] > mean) & (window - self.last_move > c.cooldown)
        )

        def utility(node, source, target, current):
            work = service[node]
            # Only two terms of the squared CV change; the mean is preserved.
            delta_l = (current[source] ** 2 + current[target] ** 2
                       - (current[source] - work) ** 2
                       - (current[target] + work) ** 2) / (len(current) * mean**2)
            delta_r = sizes[node] * (self.forecast[node, target] - self.forecast[node, source])
            age = window - self.last_move[node]
            q = max(0.0, 1 - age / c.cooldown) if c.cooldown else 0.0
            return (c.alpha * delta_r / self.r0 + c.beta * delta_l / self.l0
                    - c.gamma * sizes[node] / budget - c.eta * q)

        def feasible(node, target, current):
            return current[target] + service[node] <= (1 + c.epsilon) * mean + 1e-10

        ranked = []
        for node in candidates:
            source = int(owners[node])
            for target in np.flatnonzero(requests[node] >= c.destination_share * totals[node]):
                if target == source or not feasible(node, target, initial):
                    continue
                u = utility(node, source, target, initial)
                if u > 0 and u >= c.min_utility:
                    ranked.append(Move(int(node), source, int(target), int(sizes[node]), float(u)))
        ranked.sort(key=lambda m: (-m.utility / m.size, m.node, m.target))
        if random_seed is not None:
            np.random.default_rng(random_seed).shuffle(ranked)
        selected, rejected, used, chosen = [], [], 0, set()
        release_at = math.floor(0.75 * len(ranked))
        for index, move in enumerate(ranked):
            cap = budget if index >= release_at else math.floor((1 - c.reserve) * budget)
            u = utility(move.node, move.source, move.target, loads)
            if (move.node not in chosen and used + move.size <= cap and u > 0
                    and u >= c.min_utility and feasible(move.node, move.target, loads)):
                selected.append(Move(move.node, move.source, move.target, move.size, float(u)))
                chosen.add(move.node)
                used += move.size
                loads[move.source] -= service[move.node]
                loads[move.target] += service[move.node]
            else:
                rejected.append(move)
        if c.one_swap and selected and random_seed is None:
            # One bounded refinement: release the lowest-utility accepted move,
            # then evaluate pairs from the best 64 rejected, distinct nodes.
            old = min(selected, key=lambda m: (m.utility, m.node))
            pool = [m for m in rejected if m.node not in chosen][:64]
            best_gain, best_pair, best_loads = 0.0, None, None
            for first, second in combinations(pool, 2):
                if first.node == second.node or used - old.size + first.size + second.size > budget:
                    continue
                trial = loads.copy()
                trial[old.source] += service[old.node]
                trial[old.target] -= service[old.node]
                pair = []
                for move in (first, second):
                    u = utility(move.node, move.source, move.target, trial)
                    if not feasible(move.node, move.target, trial) or u <= 0 or u < c.min_utility:
                        break
                    trial[move.source] -= service[move.node]
                    trial[move.target] += service[move.node]
                    pair.append(Move(move.node, move.source, move.target, move.size, float(u)))
                if len(pair) != 2:
                    continue
                receiving = {m.target for m in selected if m != old} | {first.target, second.target}
                if any(trial[p] > (1 + c.epsilon) * mean + 1e-10 for p in receiving):
                    continue
                dr = sum(sizes[m.node] * (self.forecast[m.node, m.target]
                         - self.forecast[m.node, m.source]) for m in (first, second))
                dr -= sizes[old.node] * (self.forecast[old.node, old.target]
                                        - self.forecast[old.node, old.source])
                gain = (c.alpha * dr / self.r0 + c.beta * (load_cost(loads) - load_cost(trial)) / self.l0
                        - c.gamma * (first.size + second.size - old.size) / budget)
                if gain > best_gain:
                    best_gain, best_pair, best_loads = gain, pair, trial
            if best_pair is not None:
                selected.remove(old)
                selected.extend(best_pair)
                loads = best_loads
        return selected

    def commit(self, moves, window):
        for move in moves:
            self.last_move[move.node] = window
