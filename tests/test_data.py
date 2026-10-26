import numpy as np

from sentinel.data import ANOMALY_DESCRIPTIONS, ANOMALY_TYPES, METRICS, generate


def test_same_seed_same_data():
    a, b = generate(seed=5), generate(seed=5)
    assert a.frame.equals(b.frame)
    assert a.episodes == b.episodes


def test_different_seed_different_data():
    assert not generate(seed=5).frame.equals(generate(seed=6).frame)


def test_shape_and_ranges(ds):
    f = ds.frame
    assert len(f) == 14 * 1440
    assert not f[METRICS].isna().any().any()
    assert (f[METRICS] >= 0).all().all()
    assert f["cpu_pct"].max() <= 100 and f["memory_pct"].max() <= 100
    assert f["timestamp"].is_monotonic_increasing


def test_every_anomaly_type_appears_in_the_test_period(ds):
    kinds = {e.kind for e in ds.episodes if e.start >= ds.train_end}
    assert kinds == set(ANOMALY_TYPES)
    assert set(ANOMALY_DESCRIPTIONS) == set(ANOMALY_TYPES)


def test_labels_match_episodes_and_do_not_overlap(ds):
    labels = ds.frame["label"].to_numpy()
    covered = np.zeros(len(labels), dtype=bool)
    for e in ds.episodes:
        assert (labels[e.start : e.end] == e.kind).all()
        assert not covered[e.start : e.end].any()
        covered[e.start : e.end] = True
    assert (labels[~covered] == "normal").all()


def test_training_data_is_mostly_but_not_entirely_clean(ds):
    share = (ds.train["label"] != "normal").mean()
    assert 0 < share < 0.01
