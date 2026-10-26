import numpy as np
import pandas as pd
import pytest

from sentinel.detectors import robust_threshold


def test_threshold_ignores_a_few_extreme_training_points():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, 10_000)
    dirty = np.concatenate([clean, np.full(80, 50.0)])  # 0.8% contamination
    assert abs(robust_threshold(clean) - robust_threshold(dirty)) < 0.1
    assert np.percentile(dirty, 99.5) > 40  # a percentile would be dragged into the anomalies


def test_thresholds_come_from_training_scores(dets):
    for d in dets.values():
        assert d.threshold == pytest.approx(robust_threshold(d.train_scores))
        assert (d.train_scores > d.threshold).mean() < 0.03


@pytest.mark.parametrize("name", ["robust_baseline", "isolation_forest"])
def test_anomalies_score_higher_than_normal(ds, dets, name):
    scores = dets[name].score(ds.test)
    y = (ds.test["label"] != "normal").to_numpy()
    assert np.median(scores[y]) > np.percentile(scores[~y], 95)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("auth_bruteforce", {"failed_logins"}),
        # cpu_runaway also slows responses by 30%, so latency is a fair explanation too.
        ("cpu_runaway", {"cpu_pct", "cpu_per_request", "p95_latency_ms"}),
        ("exfiltration_pattern", {"net_out_mb", "net_out_per_request"}),
        ("error_burst", {"error_rate_pct"}),
        ("traffic_drop", {"requests_per_s", "net_out_mb", "cpu_pct", "cpu_per_request", "net_out_per_request"}),
    ],
)
@pytest.mark.parametrize("name", ["robust_baseline", "isolation_forest"])
def test_explanations_point_at_the_injected_signal(ds, dets, name, kind, expected):
    ep = next(e for e in ds.episodes if e.kind == kind and e.start >= ds.train_end)
    mid = ds.frame.iloc[[ep.start + ep.length // 2]]
    top = dets[name].explain(mid)[0][0]
    assert top.feature in expected


@pytest.mark.parametrize("name", ["robust_baseline", "isolation_forest"])
def test_explanation_shares_sum_to_one(ds, dets, name):
    for contribs in dets[name].explain(ds.test.head(20)):
        assert sum(c.share for c in contribs) == pytest.approx(1.0, abs=1e-6)
        assert all(c.share >= 0 for c in contribs)


def test_isolation_forest_is_reproducible(ds):
    from sentinel.detectors import build_detectors

    a = build_detectors(ds.train)["isolation_forest"].score(ds.test.head(100))
    b = build_detectors(ds.train)["isolation_forest"].score(ds.test.head(100))
    assert np.array_equal(a, b)


def test_scores_for_a_single_row(dets, ds):
    row = pd.DataFrame(ds.test.head(1))
    for d in dets.values():
        assert d.score(row).shape == (1,)
