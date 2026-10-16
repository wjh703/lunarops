"""Token-wise numeric comparison of two LunarOps output directories.

Usage: python compare_outputs.py BASELINE_DIR NEW_DIR [--rtol 1e-9] [--atol 1e-12]

Numeric tokens are compared with |a-b| <= atol + rtol*|a|; other tokens must match.
Lines carrying provenance (fingerprints, hashes, timings, paths) are ignored.
"""

from __future__ import annotations

import argparse
import gzip
import math
import re
import sys
from pathlib import Path

IGNORE = re.compile(
    r"fingerprint|sha256|Sha256|elapsed|Elapsed|performance|createdUtc|created|timestamp|"
    r"version:|lunaropsVersion|pythonVersion|hostname|workingDir|path:|Path:|/data4/|git",
)
NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def read_lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def tokens(line: str) -> list[str]:
    return re.split(r"[\s,:\[\]{}]+", line.strip())


def compare_file(a: Path, b: Path, rtol: float, atol: float) -> tuple[float, float, list[str]]:
    la = [line for line in read_lines(a) if not IGNORE.search(line)]
    lb = [line for line in read_lines(b) if not IGNORE.search(line)]
    issues: list[str] = []
    max_abs = 0.0
    max_rel = 0.0
    if len(la) != len(lb):
        issues.append(f"line count {len(la)} != {len(lb)}")
    for index, (x, y) in enumerate(zip(la, lb)):
        tx, ty = tokens(x), tokens(y)
        if len(tx) != len(ty):
            issues.append(f"line {index}: token count differs\n    - {x[:160]}\n    + {y[:160]}")
            continue
        for u, v in zip(tx, ty):
            if u == v:
                continue
            if NUM.match(u) and NUM.match(v):
                fu, fv = float(u), float(v)
                diff = abs(fu - fv)
                max_abs = max(max_abs, diff)
                if fu != 0.0:
                    max_rel = max(max_rel, diff / abs(fu))
                if diff > atol + rtol * abs(fu):
                    issues.append(f"line {index}: {u} vs {v} (diff {diff:.3e})")
            else:
                issues.append(f"line {index}: {u!r} vs {v!r}")
        if len(issues) > 20:
            issues.append("...")
            break
    return max_abs, max_rel, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--skip", action="append", default=[], help="glob of files to skip")
    args = parser.parse_args()
    failed = 0
    for a in sorted(p for p in args.baseline.rglob("*") if p.is_file()):
        rel = a.relative_to(args.baseline)
        if any(rel.match(pattern) for pattern in args.skip):
            continue
        b = args.new / rel
        if not b.exists():
            print(f"MISSING  {rel}")
            failed += 1
            continue
        try:
            max_abs, max_rel, issues = compare_file(a, b, args.rtol, args.atol)
        except (OSError, UnicodeError) as exc:
            print(f"ERROR    {rel}: {exc}")
            failed += 1
            continue
        status = "OK  " if not issues else "DIFF"
        failed += bool(issues)
        print(f"{status}     {rel}  max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        for issue in issues[:12]:
            print(f"         {issue}")
    for b in sorted(p for p in args.new.rglob("*") if p.is_file()):
        if not (args.baseline / b.relative_to(args.new)).exists():
            print(f"NEW      {b.relative_to(args.new)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
