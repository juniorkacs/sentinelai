"""Command line: python -m sentinel {evaluate,serve,tuning}."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="sentinel", description="SentinelAI anomaly detection demo")
    sub = p.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="train, evaluate and write reports/")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--out", type=Path, default=Path("reports"))
    ev.add_argument("--min-f1", type=float, default=None, help="exit non-zero if any model's F1 is below this")
    sv = sub.add_parser("serve", help="run the API and dashboard")
    sv.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    sv.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    sub.add_parser("tuning", help="re-run the validation hyperparameter search")
    args = p.parse_args()

    if args.cmd == "evaluate":
        from sentinel.report import run_evaluation

        reports = run_evaluation(args.seed, args.out)
        for r in reports:
            print(
                f"{r.model:18s} F1={r.f1:.3f} AP={r.average_precision:.3f} "
                f"episodes={r.episodes_detected}/{r.episodes} false_alerts/day={r.false_alerts_per_day}"
            )
        if args.min_f1 is not None and any(r.f1 < args.min_f1 for r in reports):
            raise SystemExit(f"F1 fell below the regression floor of {args.min_f1}")
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("sentinel.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.cmd == "tuning":
        from sentinel.tuning import search

        for row in search():
            print(*row, sep="\t")


if __name__ == "__main__":
    main()
