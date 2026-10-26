"""Synthetic telemetry for a fictional service, "orders-api", with labelled anomalies.

Everything is generated from a seed, so the same seed always gives the same
dataset. The generator is deliberately simple and documented: the point is a
transparent test bed for detectors, not a realistic simulation of any real system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS = [
    "requests_per_s",
    "cpu_pct",
    "memory_pct",
    "p95_latency_ms",
    "error_rate_pct",
    "net_out_mb",
    "failed_logins",
]

ANOMALY_TYPES = [
    "cpu_runaway",
    "memory_leak",
    "latency_degradation",
    "error_burst",
    "traffic_drop",
    "exfiltration_pattern",
    "auth_bruteforce",
    "subtle_multi_metric",
]

#: Plain-language description of each injected anomaly, shown in reports and the dashboard.
ANOMALY_DESCRIPTIONS = {
    "cpu_runaway": "CPU jumps by about 40 points without extra traffic (for example, a runaway process).",
    "memory_leak": "Memory climbs steadily by up to 35 points over the episode.",
    "latency_degradation": "p95 latency rises 2.5 times while CPU stays normal (for example, a slow dependency).",
    "error_burst": "Error rate rises by about 6 percentage points.",
    "traffic_drop": "Requests fall to about 15% of normal (for example, an upstream outage).",
    "exfiltration_pattern": "Outbound data rises about 4 times with normal request volume.",
    "auth_bruteforce": "Failed sign-ins jump to 40 to 80 a minute.",
    "subtle_multi_metric": (
        "CPU, latency, memory and error rate all rise a little at once; each change alone is near the edge "
        "of normal variation."
    ),
}


@dataclass(frozen=True)
class Episode:
    kind: str
    start: int  # row index
    length: int  # minutes

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame  # one row per minute: timestamp, METRICS..., label
    episodes: list[Episode]
    train_end: int  # rows [0, train_end) are the training period

    @property
    def train(self) -> pd.DataFrame:
        return self.frame.iloc[: self.train_end]

    @property
    def test(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_end :]


def _place_episodes(
    rng: np.random.Generator, lo: int, hi: int, kinds: list[str], taken: list[Episode]
) -> list[Episode]:
    placed: list[Episode] = []
    for kind in kinds:
        length = int(rng.integers(45, 100)) if kind == "memory_leak" else int(rng.integers(8, 45))
        for _ in range(200):
            start = int(rng.integers(lo, hi - length))
            if kind == "exfiltration_pattern":
                # Night-time, when there's less legitimate traffic to hide in.
                start = start - (start % 1440) + int(rng.integers(60, 300))
            ep = Episode(kind, start, length)
            if all(ep.end + 30 < o.start or ep.start > o.end + 30 for o in taken + placed):
                placed.append(ep)
                break
    return placed


def generate(seed: int = 42, days: int = 14, train_days: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    n = days * 1440
    t = np.arange(n)
    minute_of_day = t % 1440
    day = t // 1440
    hour = minute_of_day / 60

    # Daily load curve: quiet overnight, peak early afternoon; lighter at weekends.
    daily = 0.35 + 0.65 * np.clip(np.sin((hour - 6) / 24 * 2 * np.pi * 1.0) * 0.5 + 0.5, 0, 1) ** 1.5
    weekend = np.where(day % 7 >= 5, 0.7, 1.0)
    req = 120 * daily * weekend * rng.lognormal(0, 0.08, n)

    cpu = np.clip(8 + 0.32 * req + rng.normal(0, 2.0, n), 0, 100)
    # Memory creeps up during the day and is reclaimed overnight.
    memory = 52 + 6 * (minute_of_day / 1440) + rng.normal(0, 1.0, n)
    latency = (80 + 0.15 * req) * rng.lognormal(0, 0.06, n)
    error_rate = 0.4 + np.abs(rng.normal(0, 0.15, n))
    net_out = 0.6 * req * rng.lognormal(0, 0.07, n)
    failed_logins = rng.poisson(0.6 + 0.004 * req).astype(float)

    data = {
        "requests_per_s": req,
        "cpu_pct": cpu,
        "memory_pct": memory,
        "p95_latency_ms": latency,
        "error_rate_pct": error_rate,
        "net_out_mb": net_out,
        "failed_logins": failed_logins,
    }

    train_end = train_days * 1440
    # Two anomalies in training (real training data is never perfectly clean),
    # and every type at least once in the test period.
    train_eps = _place_episodes(rng, 600, train_end - 600, ["cpu_runaway", "error_burst"], [])
    test_kinds = ANOMALY_TYPES + [str(k) for k in rng.choice(ANOMALY_TYPES, size=5, replace=True)]
    test_eps = _place_episodes(rng, train_end + 120, n - 120, test_kinds, train_eps)
    episodes = sorted(train_eps + test_eps, key=lambda e: e.start)

    labels = np.array(["normal"] * n, dtype=object)
    for ep in episodes:
        s = slice(ep.start, ep.end)
        k = ep.length
        if ep.kind == "cpu_runaway":
            data["cpu_pct"][s] = np.clip(data["cpu_pct"][s] + 40, 0, 99.5)
            data["p95_latency_ms"][s] *= 1.3
        elif ep.kind == "memory_leak":
            data["memory_pct"][s] += np.linspace(2, 35, k)
        elif ep.kind == "latency_degradation":
            data["p95_latency_ms"][s] *= 2.5
        elif ep.kind == "error_burst":
            data["error_rate_pct"][s] += rng.normal(6, 1, k).clip(3)
        elif ep.kind == "traffic_drop":
            factor = 0.15
            data["requests_per_s"][s] *= factor
            data["cpu_pct"][s] = np.clip(8 + 0.32 * data["requests_per_s"][s] + rng.normal(0, 2, k), 0, 100)
            data["net_out_mb"][s] *= factor
        elif ep.kind == "exfiltration_pattern":
            data["net_out_mb"][s] *= 4
        elif ep.kind == "subtle_multi_metric":
            data["cpu_pct"][s] += 6
            data["p95_latency_ms"][s] *= 1.15
            data["memory_pct"][s] += 2.5
            data["error_rate_pct"][s] += 0.35
        elif ep.kind == "auth_bruteforce":
            data["failed_logins"][s] += rng.integers(40, 80, k)
        labels[s] = ep.kind

    frame = pd.DataFrame({"timestamp": pd.date_range("2026-03-02", periods=n, freq="min"), **data, "label": labels})
    for m in METRICS:
        frame[m] = frame[m].round(3)
    return Dataset(frame=frame, episodes=episodes, train_end=train_end)
