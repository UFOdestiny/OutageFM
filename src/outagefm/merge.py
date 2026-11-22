"""Merge completed evaluation shards after verifying exact manifest coverage."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from outagefm.common import load_pickle, read_json, write_json


def merge(directory, reuse_complete=False):
    directory = Path(directory)
    with (directory / ".merge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "metrics.json").exists():
            if not reuse_complete:
                raise ValueError("Preserve completed merged output")
            info, metric = read_json(directory / "evaluation.json"), read_json(directory / "metrics.json")
            if not info.get("coverage_verified") or not metric.get("complete"):
                raise ValueError("Existing output lacks verified complete coverage")
            return dict(output=str(directory), cases=metric["cases"], complete=True, reused_complete=True)
        return merge_locked(directory)


def merge_locked(directory):
    """Caller holds the directory's writer lock through the final metrics write."""
    paths = sorted(directory.glob("shard-*/evaluation.json"))
    if not paths:
        raise ValueError("No shards")
    reference = read_json(paths[0])
    required = reference["shard_count"]
    if len(paths) != required:
        raise ValueError(f"Expected {required} shards, found {len(paths)}")
    root = Path(reference["config"]["foundation"]["manifest_dir"])
    suffix = "-development" if reference["development"] else ""
    manifest = root / f"h{reference['horizon_hours']}" / f"{reference['split']}{suffix}.parquet"
    expected = pd.read_parquet(manifest, columns=["case_id", "target_valid_points"])
    seen = set()
    total = None
    seconds = []
    shard_information = []
    for path in paths:
        info = read_json(path)
        for key in [
            "seed",
            "protocol",
            "split",
            "horizon_hours",
            "development",
            "shard_count",
            "full_cases",
            "checkpoint",
            "config",
        ]:
            if info.get(key) != reference.get(key):
                raise ValueError("Shards describe different evaluation settings")
        index = info["shard_index"]
        if index in seen or not 0 <= index < required:
            raise ValueError("Duplicate or invalid shard index")
        seen.add(index)
        metrics = read_json(path.parent / "metrics.json")
        if not metrics["shard_complete"] or info["max_batches"] is not None:
            raise ValueError("Incomplete/integration shards cannot become scientific results")
        acc = load_pickle(path.parent / "accumulator.pkl")
        rows = pd.DataFrame(acc.cases)
        wanted = expected.iloc[index::required]
        if rows.case_id.tolist() != wanted.case_id.tolist():
            raise ValueError("Shard case order or coverage differs from frozen manifest")
        np.testing.assert_array_equal(rows.county_time_points, wanted.target_valid_points)
        if total is None:
            total = acc
        else:
            total.merge(acc)
        seconds.append(info["elapsed_seconds"])
        shard_information.append(info)
    cases = pd.DataFrame(total.cases)
    assert len(cases) == len(expected) and cases.case_id.is_unique
    metrics = total.compute()
    metrics.update(complete=True, max_batches=None, merged_shards=required)
    cases.to_parquet(directory / "cases.parquet", index=False)
    pd.DataFrame(total.events).to_parquet(directory / "event-cases.parquet", index=False)
    reference.update(
        shard_index=None,
        cases=len(cases),
        merged_shards=[str(p.parent) for p in paths],
        sum_shard_seconds=sum(seconds),
        maximum_shard_seconds=max(seconds),
        elapsed_seconds=max(seconds),
        elapsed_seconds_definition="Maximum shard evaluation-loop time; command resource costs are recorded separately",
        coverage_verified=True,
    )
    if all("prediction_seconds" in item for item in shard_information):
        reference.update(
            prediction_seconds=sum(item["prediction_seconds"] for item in shard_information),
            shard_prediction_seconds=[item["prediction_seconds"] for item in shard_information],
            timing_scope="Sum of adapter prediction wall time across shards; excludes loading and scoring",
        )
    for field in [
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "cpu_self_peak_rss_bytes",
        "step_memory_peak_bytes",
    ]:
        if all(field in item for item in shard_information):
            reference[field] = max(item[field] for item in shard_information)
    reference["peak_memory_scope"] = (
        "Maximum per-process peak across completed shards, not total concurrent GPU memory"
    )
    reference["CPU_memory_scope"] = (
        "cpu_self_peak_rss_bytes is the maximum individual evaluator RSS, excluding loader workers; step_memory_peak_bytes is a shared step cgroup peak, not a per-model attribution"
    )
    write_json(directory / "evaluation.json", reference)
    write_json(directory / "metrics.json", metrics)
    return dict(output=str(directory), cases=len(cases), complete=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--strict-new-output", action="store_true")
    args = parser.parse_args()
    print(merge(args.directory, reuse_complete=not args.strict_new_output))
