"""Evaluation run that writes metrics, a Markdown report and charts to reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import precision_recall_curve

from sentinel.data import generate
from sentinel.detectors import build_detectors
from sentinel.evaluate import Report, evaluate, write_reports

COLORS = {"robust_baseline": "#45c8e6", "isolation_forest": "#b39dff"}
NAMES = {"robust_baseline": "Robust baseline", "isolation_forest": "Isolation Forest"}
BG, INK, MUTED, GRID = "#0a1628", "#e4ecf5", "#93a7bf", "#1f3a5c"


def _style(ax: Axes) -> None:
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


#: Extra datasets used only to show how much results move between random draws.
#: Seed 1000 is excluded: it is the validation set used for tuning.
SENSITIVITY_SEEDS = (7, 123, 2024, 31337)


def _seed_sensitivity(md: Path) -> None:
    rows: dict[str, list[Report]] = {}
    for sd in SENSITIVITY_SEEDS:
        d = generate(seed=sd)
        for name, det in build_detectors(d.train).items():
            rows.setdefault(name, []).append(evaluate(det, d)[0])
    lines = [
        "",
        "## Sensitivity to the random draw",
        "",
        f"The same pipeline on {len(SENSITIVITY_SEEDS)} other generated datasets (seeds "
        f"{', '.join(map(str, SENSITIVITY_SEEDS))}). Each cell shows the minimum and maximum across them.",
        "",
        "| Model | F1 | Average precision | Event recall | False alerts per day |",
        "|---|---|---|---|---|",
    ]
    for name, rs in rows.items():

        def rng(key: str, rs: list[Report] = rs) -> str:
            vals = [getattr(r, key) for r in rs]
            return f"{min(vals):g} to {max(vals):g}"

        lines.append(
            f"| {name} | {rng('f1')} | {rng('average_precision')} | {rng('event_recall')} "
            f"| {rng('false_alerts_per_day')} |"
        )
    with md.open("a") as f:
        f.write("\n".join(lines) + "\n")


def run_evaluation(seed: int, out: Path) -> list[Report]:
    ds = generate(seed=seed)
    dets = build_detectors(ds.train)
    results = {name: evaluate(det, ds) for name, det in dets.items()}
    reports = [r for r, _, _ in results.values()]
    write_reports(reports, ds, out)
    _seed_sensitivity(out / "evaluation.md")

    y = (ds.test["label"] != "normal").to_numpy()

    # Precision-recall curves, with each model's operating point.
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150, facecolor=BG)
    _style(ax)
    for name, (r, scores, _) in results.items():
        p, rc, _ = precision_recall_curve(y, scores)
        ax.plot(rc, p, color=COLORS[name], lw=2, label=f"{NAMES[name]} (AP {r.average_precision:.2f})")
        ax.scatter([r.recall], [r.precision], color=COLORS[name], s=40, zorder=3, edgecolor=BG)
    ax.set_xlabel("Recall (share of anomalous minutes flagged)")
    ax.set_ylabel("Precision (share of flags that were anomalous)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_title("Precision-recall on the test week (dots: chosen thresholds)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", facecolor=BG)
    plt.close(fig)

    # Score timeline with labelled episodes shaded.
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), dpi=150, sharex=True, facecolor=BG)
    minutes = np.arange(len(ds.test)) / 1440
    for ax, (name, (_r, scores, _)) in zip(axes, results.items(), strict=True):
        _style(ax)
        for ep in ds.episodes:
            if ep.start >= ds.train_end:
                a = (ep.start - ds.train_end) / 1440
                ax.axvspan(a, a + ep.length / 1440, color="#f0b54a", alpha=0.25, lw=0)
        ax.plot(minutes, scores, color=COLORS[name], lw=0.6)
        ax.axhline(dets[name].threshold, color=INK, lw=0.8, ls="--")
        ax.set_ylabel(NAMES[name], fontsize=9)
    axes[0].set_title(
        "Anomaly score over the test week (shaded: labelled episodes, dashed: threshold)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    axes[1].set_xlabel("Days into the test period")
    fig.tight_layout()
    fig.savefig(out / "score_timeline.png", facecolor=BG)
    plt.close(fig)
    return reports
