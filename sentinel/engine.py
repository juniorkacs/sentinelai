"""Everything the API needs, built once at start-up: data, fitted detectors,
scores, alerts and evaluation results. Building takes a few seconds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.data import ANOMALY_DESCRIPTIONS, Dataset, generate
from sentinel.detectors import Contribution, Detector, build_detectors
from sentinel.evaluate import Alert, Report, evaluate
from sentinel.features import FEATURES, LABELS, add_ratios


@dataclass
class ModelRun:
    detector: Detector
    report: Report
    scores: np.ndarray  # test period
    alerts: list[Alert]


class Engine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ds: Dataset = generate(seed=seed)
        self.detectors = build_detectors(self.ds.train, seed=0)
        self.runs: dict[str, ModelRun] = {}
        for name, det in self.detectors.items():
            report, scores, alerts = evaluate(det, self.ds)
            self.runs[name] = ModelRun(det, report, scores, alerts)

    @property
    def test(self) -> pd.DataFrame:
        return self.ds.test

    def label_for(self, start: int, end: int) -> str | None:
        """The labelled episode that overlaps rows [start, end), if any."""
        for ep in self.ds.episodes:
            if start < ep.end and ep.start < end:
                return ep.kind
        return None

    def explain_rows(self, model: str, df: pd.DataFrame, top: int = 4) -> list[list[dict[str, object]]]:
        det = self.detectors[model]
        typical = det.profile.typical(df["timestamp"])
        values = add_ratios(df)
        out = []
        for i, contribs in enumerate(det.explain(df)):
            out.append([self._contrib(c, values.iloc[i], typical.iloc[i]) for c in contribs[:top]])
        return out

    @staticmethod
    def _contrib(c: Contribution, row: pd.Series, typical: pd.Series) -> dict[str, object]:
        direction = "higher" if c.z > 0 else "lower"
        return {
            "feature": c.feature,
            "label": LABELS[c.feature],
            "value": round(float(row[c.feature]), 3),
            "typical": round(float(typical[c.feature]), 3),
            "z": round(c.z, 2),
            "share": round(c.share, 3),
            "summary": f"{LABELS[c.feature]} is {abs(c.z):.1f} typical deviations {direction} than usual at this time",
        }

    def alert_payload(self, model: str) -> list[dict[str, object]]:
        run = self.runs[model]
        frame = self.ds.frame
        peaks = frame.iloc[[a.peak_index for a in run.alerts]] if run.alerts else frame.iloc[0:0]
        explanations = self.explain_rows(model, peaks) if len(peaks) else []
        out = []
        for a, expl in zip(run.alerts, explanations, strict=True):
            label = self.label_for(a.start, a.end)
            out.append(
                {
                    "id": f"{model}-{a.start}",
                    "start": frame["timestamp"].iloc[a.start].isoformat(),
                    "end": frame["timestamp"].iloc[a.end - 1].isoformat(),
                    "minutes": a.end - a.start,
                    "peak_time": frame["timestamp"].iloc[a.peak_index].isoformat(),
                    "peak_score": round(a.peak_score, 3),
                    "threshold": round(run.detector.threshold, 3),
                    "explanation": expl,
                    # Ground truth exists only because the data is synthetic.
                    "ground_truth": label or "no labelled anomaly (false alert)",
                    "ground_truth_description": ANOMALY_DESCRIPTIONS.get(label or "", ""),
                }
            )
        return out

    def series(self, metric: str) -> dict[str, object]:
        test = self.test
        lo, hi = (
            self.detectors["robust_baseline"].profile.band(test["timestamp"], metric)
            if metric in FEATURES
            else (None, None)
        )
        labels = (test["label"] != "normal").to_numpy()
        return {
            "metric": metric,
            "label": LABELS[metric],
            "timestamps": [t.isoformat() for t in test["timestamp"]],
            "values": test[metric].round(3).tolist(),
            "band_low": None if lo is None else np.round(lo, 3).tolist(),
            "band_high": None if hi is None else np.round(hi, 3).tolist(),
            "labelled_anomaly": labels.astype(int).tolist(),
            "flags": {
                name: (run.scores > run.detector.threshold).astype(int).tolist() for name, run in self.runs.items()
            },
        }
