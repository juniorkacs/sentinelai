import numpy as np
from sklearn.metrics import f1_score

from sentinel.evaluate import evaluate, group_alerts


def test_group_alerts_merges_short_gaps():
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], dtype=bool)
    scores = np.arange(len(flags), dtype=float)
    alerts = group_alerts(flags, scores, offset=100, gap=3)
    assert [(a.start, a.end) for a in alerts] == [(101, 106), (111, 112)]
    assert alerts[0].peak_index == 105


def test_group_alerts_empty():
    assert group_alerts(np.zeros(5, dtype=bool), np.zeros(5)) == []


def test_metrics_agree_with_sklearn(ds, dets):
    det = dets["robust_baseline"]
    report, scores, _ = evaluate(det, ds)
    y = (ds.test["label"] != "normal").to_numpy()
    assert report.f1 == round(f1_score(y, scores > det.threshold), 3)
    assert report.anomalous_minutes == int(y.sum())


def test_regression_floor(ds, dets):
    """Guards against silent degradation. Floors are well below current results."""
    base, _, _ = evaluate(dets["robust_baseline"], ds)
    iso, _, _ = evaluate(dets["isolation_forest"], ds)
    assert base.f1 >= 0.85 and base.event_recall >= 0.9
    assert iso.average_precision >= 0.75
