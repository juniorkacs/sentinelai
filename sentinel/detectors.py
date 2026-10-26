"""Two detectors with the same interface, so they can be compared fairly.

* ``RobustBaseline``: the largest absolute hour-of-day z-score across features.
  Simple, fast and directly explainable, but it judges each feature on its own.
* ``IsolationForestDetector``: scikit-learn's Isolation Forest on the same
  z-scores. It can flag unusual *combinations* that no single feature shows,
  at the cost of an anomaly score that has no natural unit.

Both thresholds come from the training period only, using Tukey's "far out"
fence on the training scores (Q3 + 3 x IQR). A high percentile would be simpler,
but the training data contains a few anomalous minutes (as real data does), and
those pull a 99.5th-percentile threshold up into the anomaly range. Quartiles
are not affected by a small share of contaminated points.

Isolation Forest hyperparameters were chosen on a separate validation dataset
(seed 1000, see ``sentinel/tuning.py``), never on the reported test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sentinel.features import FEATURES, HourProfile

FENCE_K = 3.0


def robust_threshold(scores: np.ndarray, k: float = FENCE_K) -> float:
    q1, q3 = np.percentile(scores, [25, 75])
    return float(q3 + k * (q3 - q1))


@dataclass
class Contribution:
    feature: str
    z: float
    #: Share of the anomaly attributed to this feature, 0-1 (sums to about 1 across features).
    share: float


@dataclass
class Detector:
    name: str
    profile: HourProfile
    threshold: float = 0.0
    train_scores: np.ndarray = field(default_factory=lambda: np.empty(0))

    def score_z(self, z: np.ndarray) -> np.ndarray:  # pragma: no cover (abstract)
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> Detector:
        z = self.profile.zscores(train).to_numpy()
        self._fit_model(z)
        self.train_scores = self.score_z(z)
        self.threshold = robust_threshold(self.train_scores)
        return self

    def _fit_model(self, z: np.ndarray) -> None:
        pass

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.score_z(self.profile.zscores(df).to_numpy())

    def explain_z(self, z_row: np.ndarray) -> list[Contribution]:  # pragma: no cover (abstract)
        raise NotImplementedError

    def explain(self, df: pd.DataFrame) -> list[list[Contribution]]:
        z = self.profile.zscores(df).to_numpy()
        return [self.explain_z(row) for row in z]


class RobustBaseline(Detector):
    def __init__(self, profile: HourProfile):
        super().__init__(name="robust_baseline", profile=profile)

    def score_z(self, z: np.ndarray) -> np.ndarray:
        return np.abs(z).max(axis=1)

    def explain_z(self, z_row: np.ndarray) -> list[Contribution]:
        a = np.abs(z_row)
        total = a.sum() or 1.0
        return sorted(
            (Contribution(f, float(z_row[i]), float(a[i] / total)) for i, f in enumerate(FEATURES)),
            key=lambda c: -c.share,
        )


class IsolationForestDetector(Detector):
    #: Chosen on the validation dataset (see sentinel/tuning.py).
    N_ESTIMATORS = 400
    MAX_SAMPLES = 4096

    def __init__(self, profile: HourProfile, seed: int = 0):
        super().__init__(name="isolation_forest", profile=profile)
        # Trees see |z| (distance from typical, either direction), clipped so one
        # extreme value can't dominate. Explanations still report the signed z.
        self.clip = 12.0
        self.model = IsolationForest(
            n_estimators=self.N_ESTIMATORS, max_samples=self.MAX_SAMPLES, contamination="auto", random_state=seed
        )

    def _prep(self, z: np.ndarray) -> np.ndarray:
        return np.clip(np.abs(z), 0, self.clip)

    def _fit_model(self, z: np.ndarray) -> None:
        self.model.fit(self._prep(z))

    def score_z(self, z: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points; negate so higher = more anomalous.
        return np.asarray(-self.model.score_samples(self._prep(z)))

    def explain_z(self, z_row: np.ndarray) -> list[Contribution]:
        """Occlusion: reset each feature to typical (z = 0) and see how much the score falls."""
        base = self.score_z(z_row[None, :])[0]
        variants = np.repeat(z_row[None, :], len(FEATURES), axis=0)
        np.fill_diagonal(variants, 0.0)
        drops = np.clip(base - self.score_z(variants), 0, None)
        total = drops.sum()
        if total <= 1e-12:  # no single feature explains it; fall back to deviation size
            a = np.abs(z_row)
            drops, total = a, a.sum() or 1.0
        return sorted(
            (Contribution(f, float(z_row[i]), float(drops[i] / total)) for i, f in enumerate(FEATURES)),
            key=lambda c: -c.share,
        )


def build_detectors(train: pd.DataFrame, seed: int = 0) -> dict[str, Detector]:
    profile = HourProfile.fit(train)
    return {
        "robust_baseline": RobustBaseline(profile).fit(train),
        "isolation_forest": IsolationForestDetector(profile, seed=seed).fit(train),
    }
