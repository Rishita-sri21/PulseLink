"""Packet-loss and outage models for the channel simulator."""
from __future__ import annotations

import numpy as np


class BernoulliLoss:
    """Independent loss: every packet is dropped with probability ``p``."""

    def __init__(self, p: float, rng: np.random.Generator):
        self.p, self.rng = float(p), rng

    def lost(self) -> bool:
        return self.p > 0 and self.rng.random() < self.p


class GilbertElliottLoss:
    """Two-state bursty loss (simple Gilbert model).

    In the *bad* state every packet is lost, in the *good* state none is.
    Parameterised by the long-run loss rate and the mean burst length, which
    are easier to reason about than the transition probabilities::

        P(bad -> good) = r = 1 / burst_length
        P(good -> bad) = q = loss * r / (1 - loss)
        stationary loss = q / (q + r) = loss
    """

    def __init__(self, loss: float, burst_length: float, rng: np.random.Generator):
        loss = float(np.clip(loss, 0.0, 0.99))
        self.r = 1.0 / max(1.0, burst_length)
        self.q = loss * self.r / (1.0 - loss) if loss > 0 else 0.0
        self.rng = rng
        self.bad = rng.random() < loss

    def lost(self) -> bool:
        if self.bad:
            self.bad = self.rng.random() >= self.r
        else:
            self.bad = self.rng.random() < self.q
        return self.bad


def make_loss_model(loss_rate: float, burst_length: float, rng: np.random.Generator):
    if burst_length <= 1.0:
        return BernoulliLoss(loss_rate, rng)
    return GilbertElliottLoss(loss_rate, burst_length, rng)


class OutageSchedule:
    """Intervals during which the link is down (e.g. wearer walks away)."""

    def __init__(self, outages: list[tuple[float, float]] | None = None):
        self.intervals = sorted((float(s), float(s) + float(d)) for s, d in (outages or []) if d > 0)

    def is_down(self, t: float) -> bool:
        return any(s <= t < e for s, e in self.intervals)

    def next_up(self, t: float) -> float:
        """Earliest time >= t at which the link is up."""
        for s, e in self.intervals:
            if s <= t < e:
                t = e
        return t

    def total_downtime(self) -> float:
        return sum(e - s for s, e in self.intervals)

    @staticmethod
    def random(duration_s: float, n: int, mean_len_s: float, rng: np.random.Generator) -> "OutageSchedule":
        starts = np.sort(rng.uniform(0, duration_s, size=n))
        lens = rng.exponential(mean_len_s, size=n)
        return OutageSchedule(list(zip(starts, lens)))
