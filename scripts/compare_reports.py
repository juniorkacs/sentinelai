"""Fail if a fresh evaluation differs from the committed one.

Counts must match exactly. Floating-point metrics may differ by at most 0.005,
which allows for tiny numerical differences between machines.
"""

import json
import sys
from pathlib import Path


def close(a: object, b: object, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [m for k in a for m in close(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [m for i, (x, y) in enumerate(zip(a, b, strict=True)) for m in close(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        return [] if abs(float(a) - float(b)) <= 0.005 else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a} != {b}"]


if __name__ == "__main__":
    committed, fresh = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    problems = close(committed, fresh)
    if problems:
        print("Evaluation results changed. Re-run `python -m sentinel evaluate` and commit reports/.")
        print("\n".join(problems))
        sys.exit(1)
    print("Evaluation reproduced: results match the committed report.")
