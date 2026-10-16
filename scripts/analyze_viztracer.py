"""Summarize VizTracer complete events by function and source module."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

SOURCE_RE = re.compile(r"\(([^()]+\.py):\d+\)$")


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    return [
        event
        for event in report.get("traceEvents", ())
        if event.get("ph") == "X" and float(event.get("dur", 0.0)) >= 0.0
    ]


def event_module(name: str) -> str:
    match = SOURCE_RE.search(name)
    if match is None:
        return "<unknown>"
    return Path(match.group(1)).name


def aggregate(events: list[dict]):
    by_thread: dict[tuple[object, object], list[dict]] = defaultdict(list)
    for event in events:
        by_thread[(event.get("pid"), event.get("tid"))].append(event)

    function_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for thread_events in by_thread.values():
        ordered = sorted(
            thread_events,
            key=lambda event: (float(event["ts"]), -float(event["dur"])),
        )
        stack: list[tuple[float, dict, float]] = []
        completed: list[tuple[dict, float]] = []
        for event in ordered:
            start = float(event["ts"])
            end = start + float(event["dur"])
            while stack and start >= stack[-1][0]:
                _, finished, child_duration = stack.pop()
                completed.append((finished, child_duration))
            if stack and end <= stack[-1][0] + 1e-9:
                parent_end, parent, child_duration = stack[-1]
                stack[-1] = (parent_end, parent, child_duration + float(event["dur"]))
            stack.append((end, event, 0.0))
        while stack:
            _, finished, child_duration = stack.pop()
            completed.append((finished, child_duration))

        for event, child_duration in completed:
            name = str(event.get("name", "<unnamed>"))
            duration = float(event["dur"])
            stats = function_stats[name]
            stats[0] += 1
            stats[1] += duration
            stats[2] += max(0.0, duration - child_duration)

    module_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for name, stats in function_stats.items():
        module = event_module(name)
        module_stats[module][0] += stats[0]
        module_stats[module][1] += stats[1]
        module_stats[module][2] += stats[2]
    return function_stats, module_stats


def print_table(title: str, stats: dict[str, list[float]], field: int, limit: int) -> None:
    print(f"\n{title}")
    print(f"{'seconds':>12} {'calls':>12}  name")
    for name, values in sorted(stats.items(), key=lambda item: item[1][field], reverse=True)[:limit]:
        print(f"{values[field] / 1e6:12.6f} {int(values[0]):12d}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    events = load_events(args.trace)
    function_stats, module_stats = aggregate(events)
    print(f"complete_events: {len(events)}")
    print_table("Top functions by inclusive time", function_stats, 1, args.limit)
    print_table("Top functions by exclusive Python time", function_stats, 2, args.limit)
    print_table("Modules by exclusive Python time", module_stats, 2, args.limit)


if __name__ == "__main__":
    main()
