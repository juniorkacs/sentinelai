"""Feature engineering.

Raw metrics follow a strong daily and weekly cycle, so "unusual" has to mean
unusual *for this time*. Every feature is therefore expressed as a robust
z-score against a profile (median and robust scale) for each hour of the day,
kept separately for weekdays and weekends, and learned from training data.
These normalised features are what both detectors see, and they are also what
the explanations report, so they keep a plain meaning: "how many typical
deviations away from normal for this hour".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.data import METRICS

#: Ratios capture relationships between metrics, for example data sent per request.
RATIOS = {
    "net_out_per_request": ("net_out_mb", "requests_per_s"),
    "cpu_per_request": ("cpu_pct", "requests_per_s"),
}
FEATURES = METRICS + list(RATIOS)

#: Readable names for dashboards and reports.
LABELS = {
    "requests_per_s": "Requests per second",
    "cpu_pct": "CPU %",
    "memory_pct": "Memory %",
    "p95_latency_ms": "p95 latency (ms)",
    "error_rate_pct": "Error rate %",
    "net_out_mb": "Outbound data (MB/min)",
    "failed_logins": "Failed sign-ins per minute",
    "net_out_per_request": "Outbound data per request",
    "cpu_per_request": "CPU per request",
}

MAD_SCALE = 1.4826  # makes MAD comparable to a standard deviation for normal data
IQR_SCALE = 1.349  # likewise for the interquartile range


def _mad(s: pd.Series) -> float:
    return float((s - s.median()).abs().median())


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (num, den) in RATIOS.items():
        out[name] = out[num] / out[den].clip(lower=1.0)
    return out


@dataclass
class HourProfile:
    """Median and scale per (weekend?, hour) slot for each feature, fitted on training data only."""

    median: pd.DataFrame  # index: slot 0-47 (weekday hours 0-23, weekend hours 24-47), columns: FEATURES
    scale: pd.DataFrame

    @staticmethod
    def slots(timestamps: pd.Series) -> np.ndarray:
        return np.asarray(timestamps.dt.hour + 24 * (timestamps.dt.dayofweek >= 5))

    @classmethod
    def fit(cls, train: pd.DataFrame) -> HourProfile:
        df = add_ratios(train)
        slot = pd.Series(cls.slots(df["timestamp"]), index=df.index)
        med = df.groupby(slot)[FEATURES].median()
        g = df.groupby(slot)[FEATURES]
        mad = g.agg(_mad) * MAD_SCALE
        iqr = (g.quantile(0.75) - g.quantile(0.25)) / IQR_SCALE
        # For small counts (such as failed sign-ins at night) the MAD is often 0,
        # which would turn a single extra event into a huge z-score. The IQR and
        # a small absolute floor keep the scale sensible; both are robust to the
        # occasional anomaly in the training data.
        scale = mad.where(mad >= iqr, iqr).clip(lower=1e-3)
        return cls(median=med, scale=scale)

    def zscores(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = add_ratios(df)
        slots = self.slots(frame["timestamp"])
        med = self.median.reindex(slots).to_numpy()
        scale = self.scale.reindex(slots).to_numpy()
        z = (frame[FEATURES].to_numpy() - med) / scale
        return pd.DataFrame(z, columns=FEATURES, index=df.index)

    def typical(self, timestamps: pd.Series) -> pd.DataFrame:
        slots = self.slots(timestamps)
        return pd.DataFrame(self.median.reindex(slots).to_numpy(), columns=FEATURES, index=timestamps.index)

    def band(self, timestamps: pd.Series, feature: str, k: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        slots = self.slots(timestamps)
        med = self.median[feature].reindex(slots).to_numpy()
        sc = self.scale[feature].reindex(slots).to_numpy()
        return med - k * sc, med + k * sc
