#!/usr/bin/env python3
"""Merge metric CSVs by model into one wide CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ORDER = [
    "clip_score",
    "aesthetic",
    "image_reward",
    "hps_v2",
    "hps_v2_1",
    "pick_score",
    "arniqa",
    "topiq_nr",
    "clipiqa",
    "liqe",
    "musiq",
    "niqe",
    "nrqm",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources-output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    rows: dict[str, dict[str, str]] = {}
    sources: dict[str, dict[str, str]] = {}
    seen_metrics: list[str] = []

    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "model" not in reader.fieldnames:
                raise ValueError(f"{path} does not have a model column")

            metrics = [name for name in reader.fieldnames if name != "model"]
            for metric in metrics:
                if metric not in seen_metrics:
                    seen_metrics.append(metric)

            for row in reader:
                model = row["model"]
                rows.setdefault(model, {"model": model})
                sources.setdefault(model, {"model": model})
                for metric in metrics:
                    value = row.get(metric, "")
                    if value == "":
                        continue
                    rows[model][metric] = value
                    sources[model][metric] = path.name

    ordered_metrics = [m for m in DEFAULT_ORDER if m in seen_metrics]
    ordered_metrics += [m for m in seen_metrics if m not in ordered_metrics]
    fieldnames = ["model"] + ordered_metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(rows):
            writer.writerow({key: rows[model].get(key, "") for key in fieldnames})

    if args.sources_output:
        args.sources_output.parent.mkdir(parents=True, exist_ok=True)
        with args.sources_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for model in sorted(sources):
                writer.writerow(
                    {key: sources[model].get(key, "") for key in fieldnames}
                )


if __name__ == "__main__":
    main()
