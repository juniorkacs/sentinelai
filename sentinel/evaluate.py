"""Evaluation on the held-out test period.

Point-level metrics treat each minute as one prediction. Event-level metrics
reflect how an on-call engineer experiences a detector: was each incident
noticed, how quickly, and how many false alarms arrived per day?
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from sentinel.data import Dataset, Episode
from sentinel.detectors import Detector

#: Flags separated by no more than this many quiet minutes are merged into one alert.
MERGE_GAP_MIN = 3


@dataclass
class Alert:
    start: int  # row index
    end: int  # exclusive
    peak_score: float
    peak_index: int


def group_alerts(flags: np.ndarray, scores: np.ndarray, offset: int = 0, gap: int = MERGE_GAP_MIN) -> list[Alert]:
    """Merge flagged minutes into alerts. `offset` converts local indices to dataset rows."""
    idx = np.flatnonzero(flags)
    alerts: list[Alert] = []
    if idx.size == 0:
        return alerts
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap + 1:
            alerts.append(_alert(start, prev + 1, scores, offset))
            start = i
        prev = i
    alerts.append(_alert(start, prev + 1, scores, offset))
    return alerts


def _alert(s: int, e: int, scores: np.ndarray, offset: int) -> Alert:
    peak = s + int(np.argmax(scores[s:e]))
    return Alert(start=s + offset, end=e + offset, peak_score=float(scores[peak]), peak_index=peak + offset)


@dataclass
class Report:
    model: str
    threshold: float
    test_minutes: int
    anomalous_minutes: int
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float
    episodes: int
    episodes_detected: int
    event_recall: float
    median_detection_delay_min: float | None
    alerts: int
    false_alerts: int
    false_alerts_per_day: float
    recall_by_type: dict[str, float]


def evaluate(det: Detector, ds: Dataset) -> tuple[Report, np.ndarray, list[Alert]]:
    test = ds.test
    y = (test["label"] != "normal").to_numpy()
    scores = det.score(test)
    flags = scores > det.threshold
    alerts = group_alerts(flags, scores, offset=ds.train_end)

    test_eps = [e for e in ds.episodes if e.start >= ds.train_end]
    detected, delays = [], []
    for ep in test_eps:
        local = slice(ep.start - ds.train_end, ep.end - ds.train_end)
        hits = np.flatnonzero(flags[local])
        detected.append(hits.size > 0)
        if hits.size:
            delays.append(int(hits[0]))

    def overlaps(a: Alert, ep: Episode) -> bool:
        return a.start < ep.end and ep.start < a.end

    false_alerts = sum(1 for a in alerts if not any(overlaps(a, ep) for ep in test_eps))
    days = len(test) / 1440

    by_type: dict[str, list[bool]] = {}
    for ep, hit in zip(test_eps, detected, strict=True):
        by_type.setdefault(ep.kind, []).append(hit)

    r = Report(
        model=det.name,
        threshold=round(det.threshold, 4),
        test_minutes=len(test),
        anomalous_minutes=int(y.sum()),
        precision=round(float(precision_score(y, flags, zero_division=0)), 3),
        recall=round(float(recall_score(y, flags, zero_division=0)), 3),
        f1=round(float(f1_score(y, flags, zero_division=0)), 3),
        average_precision=round(float(average_precision_score(y, scores)), 3),
        roc_auc=round(float(roc_auc_score(y, scores)), 3),
        episodes=len(test_eps),
        episodes_detected=int(sum(detected)),
        event_recall=round(sum(detected) / len(test_eps), 3) if test_eps else 0.0,
        median_detection_delay_min=float(np.median(delays)) if delays else None,
        alerts=len(alerts),
        false_alerts=false_alerts,
        false_alerts_per_day=round(false_alerts / days, 2),
        recall_by_type={k: round(sum(v) / len(v), 2) for k, v in sorted(by_type.items())},
    )
    return r, scores, alerts


def write_reports(reports: list[Report], ds: Dataset, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps([asdict(r) for r in reports], indent=2))
    lines = [
        "# SentinelAI evaluation report",
        "",
        f"Synthetic dataset, seed-generated: {len(ds.train):,} training minutes and {len(ds.test):,} test minutes. "
        f"The test period contains {reports[0].episodes} labelled anomaly episodes "
        f"({reports[0].anomalous_minutes:,} anomalous minutes). Thresholds come from the training period only.",
        "",
        "| Metric | " + " | ".join(r.model for r in reports) + " |",
        "|---|" + "---|" * len(reports),
    ]
    rows = [
        ("Precision (per minute)", "precision"),
        ("Recall (per minute)", "recall"),
        ("F1 (per minute)", "f1"),
        ("Average precision (PR-AUC)", "average_precision"),
        ("ROC-AUC", "roc_auc"),
        ("Episodes detected", "episodes_detected"),
        ("Event recall", "event_recall"),
        ("Median detection delay (min)", "median_detection_delay_min"),
        ("Alerts raised", "alerts"),
        ("False alerts", "false_alerts"),
        ("False alerts per day", "false_alerts_per_day"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(str(getattr(r, key)) for r in reports) + " |")
    lines += [
        "",
        "## Episode recall by anomaly type",
        "",
        "| Type | " + " | ".join(r.model for r in reports) + " |",
        "|---|" + "---|" * len(reports),
    ]
    for t in reports[0].recall_by_type:
        lines.append(f"| {t} | " + " | ".join(str(r.recall_by_type.get(t, "")) for r in reports) + " |")
    lines += [
        "",
        "## Reading these numbers",
        "",
        "- These results describe performance on **synthetic data generated by this project**. They do not predict "
        "performance on real telemetry, which has messier seasonality, drift and anomaly shapes.",
        "- Per-minute recall is lower than event recall because a detector only needs to flag part of an episode "
        "for an engineer to notice it, and gradual anomalies (such as memory leaks) start below any threshold.",
        "- A flagged minute is a prompt for investigation, not a confirmed incident.",
        "- How the settings were chosen: Isolation Forest hyperparameters were picked on a separate validation "
        "dataset (seed 1000). The threshold rule (Q3 + 3 x IQR of training scores) replaced a 99.5th-percentile "
        "rule after the percentile proved sensitive to anomalies in the training data. That change was made "
        "during development while test results were visible, so treat these numbers as somewhat optimistic.",
    ]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n")


def episodes_frame(ds: Dataset) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) | {"end": e.end} for e in ds.episodes])
