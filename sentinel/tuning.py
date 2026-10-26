"""Hyperparameter search for Isolation Forest on a separate validation dataset.

Run ``python -m sentinel.tuning``. The validation data uses a different seed
(1000) from the reported dataset (42), so the reported test metrics were not
used to pick these settings.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from sentinel.data import generate
from sentinel.features import HourProfile

VALIDATION_SEED = 1000


def search() -> list[tuple[float, int, int, bool]]:
    ds = generate(seed=VALIDATION_SEED)
    prof = HourProfile.fit(ds.train)
    ztr, zte = prof.zscores(ds.train).to_numpy(), prof.zscores(ds.test).to_numpy()
    y = (ds.test["label"] != "normal").to_numpy()
    results = []
    for n_est, max_samples, use_abs in itertools.product((200, 400), (256, 1024, 4096), (False, True)):
        a, b = (np.abs(ztr), np.abs(zte)) if use_abs else (ztr, zte)
        m = IsolationForest(n_estimators=n_est, max_samples=max_samples, random_state=0)
        m.fit(np.clip(a, -12, 12))
        ap = float(average_precision_score(y, -m.score_samples(np.clip(b, -12, 12))))
        results.append((round(ap, 3), n_est, max_samples, use_abs))
    return sorted(results, reverse=True)


if __name__ == "__main__":
    print("AP  n_estimators  max_samples  abs(z)")
    for row in search():
        print(*row, sep="\t")
