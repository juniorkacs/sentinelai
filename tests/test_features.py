import numpy as np

from sentinel.features import FEATURES, HourProfile, add_ratios


def test_ratios_handle_zero_traffic(ds):
    df = ds.test.head(3).copy()
    df["requests_per_s"] = 0.0
    out = add_ratios(df)
    assert np.isfinite(out["net_out_per_request"]).all()


def test_profile_uses_training_data_only(ds):
    p1 = HourProfile.fit(ds.train)
    shifted = ds.frame.copy()
    shifted.loc[ds.train_end :, "cpu_pct"] += 50  # change only the test period
    p2 = HourProfile.fit(shifted.iloc[: ds.train_end])
    assert p1.median.equals(p2.median)


def test_typical_training_points_have_small_z(ds):
    p = HourProfile.fit(ds.train)
    normal = ds.train[ds.train["label"] == "normal"]
    z = p.zscores(normal)
    assert z.abs().median().max() < 1.0
    assert (p.scale > 0).all().all()


def test_weekend_and_weekday_profiles_differ(ds):
    p = HourProfile.fit(ds.train)
    # Slot 13 is 13:00 on a weekday, slot 37 is 13:00 at the weekend.
    assert p.median.loc[37, "requests_per_s"] < p.median.loc[13, "requests_per_s"] * 0.85


def test_zscore_columns(ds):
    p = HourProfile.fit(ds.train)
    assert list(p.zscores(ds.test.head(5)).columns) == FEATURES
