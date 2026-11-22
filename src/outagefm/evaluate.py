"""Streaming, mask-aware evaluation of OutageFM on the frozen manifests (one shard; see scripts/evaluate_sharded.py).

Scores WQL, log-WQL, median MAE, RMSE (point semantics of the model) and endpoint-event AP/Brier per window cohort;
writes metrics.json, evaluation.json, cases.parquet, event-cases.parquet and accumulator.pkl. Other model kinds are
loaded from the optional baseline/ package, which is not part of this release.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from outagefm.common import (
    canonical_config,
    canonical_kind,
    environment,
    import_baseline,
    load_pickle,
    read_config,
    seed_everything,
    write_json,
)
from outagefm.data.dataset import FoundationDataset, forecast_inputs
from outagefm.scoring import MaskedCountAccumulator


def host_memory():
    """Linux process RSS and the current shared Slurm step's memory counters."""
    result = {}
    try:
        fields = dict(
            line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line
        )
        result["cpu_self_rss_bytes"] = int(fields["VmRSS"].split()[0]) * 1024
        result["cpu_self_peak_rss_bytes"] = int(fields["VmHWM"].split()[0]) * 1024
        group = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[1]
        root = Path("/sys/fs/cgroup") / group.lstrip("/")
        for name in ["current", "peak"]:
            result[f"step_memory_{name}_bytes"] = int((root / f"memory.{name}").read_text())
    except (OSError, KeyError, IndexError, ValueError):
        pass
    return result


def check_weather_training_information(trained, requested):
    if trained.get("protocol", "H") != requested.get("protocol", "H"):
        raise ValueError("Checkpoint was fitted under a different weather information protocol")
    if requested.get("protocol") == "F":
        # Weather stores listed together in evaluation.equivalent_packed_dirs hold identical values on the evaluated
        # dates, so a checkpoint adapted with one may be scored with another.
        equivalent = set(requested.get("evaluation", {}).get("equivalent_packed_dirs", []))
        for field in ["source_kind", "features", "packed_dir", "feature_center", "feature_scale"]:
            mine, theirs = trained.get("weather", {}).get(field), requested["weather"].get(field)
            if field == "packed_dir" and mine in equivalent and theirs in equivalent:
                continue
            if mine != theirs:
                raise ValueError(f"Checkpoint weather input differs: {field}")


def evaluate(
    config,
    model_name,
    output,
    split="validation",
    horizon_hours=48,
    device="cpu",
    development=False,
    max_batches=None,
    checkpoint=None,
    batch_size=32,
    resume=False,
    shard_index=0,
    shard_count=1,
):
    config = canonical_config(copy.deepcopy(config))
    model_name = canonical_kind(model_name)
    if device == "cuda" and config.get("evaluation", {}).get("matmul_precision"):
        torch.set_float32_matmul_precision(config["evaluation"]["matmul_precision"])
    if config.get("protocol") == "F":
        config["evaluation"]["save_predictions"] = bool(
            development and config["evaluation"].get("save_development_predictions", False)
        )
    if max_batches is not None:
        config["evaluation"]["save_predictions"] = False
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.9)
    config["experiment_name"] = model_name
    aliases = {"chronos2_adapt": "chronos2", "chronos2_univariate": "chronos2"}
    kind = aliases.get(model_name, model_name)
    config["model"]["kind"] = kind
    config["windows"]["manifest_dir"] = str(Path(config["foundation"]["manifest_dir"]) / f"h{horizon_hours}")
    config["windows"]["horizon_steps"] = int(horizon_hours * 4)
    config["evaluation"]["batch_size"] = batch_size
    config["evaluation"]["workers"] = int(config["evaluation"].get("workers", 2 if device == "cpu" else 1))
    if config["evaluation"]["workers"] < 0:
        raise ValueError("Evaluation workers must be nonnegative")
    config["baseline"] = {
        **config.get("baseline", {}),
        "univariate": True,
        "inference_batch_size": batch_size,
    }
    if not 0 <= shard_index < shard_count:
        raise ValueError("Require 0 <= shard-index < shard-count")
    output = Path(output)
    if (output / "metrics.json").exists():
        raise ValueError("Preserve completed evaluations; select a new output")
    if output.exists() and any(output.iterdir()) and (not resume):
        raise ValueError("Partial output exists; inspect it before resuming or select a new directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    ds = FoundationDataset(config, split, horizon_hours, development=development)
    indices = range(shard_index, len(ds), shard_count)
    total_batches = (len(indices) + batch_size - 1) // batch_size
    info = dict(
        config=config,
        seed=42,
        protocol=config["protocol"],
        split=split,
        device=device,
        horizon_hours=horizon_hours,
        development=development,
        max_batches=max_batches,
        shard_index=shard_index,
        shard_count=shard_count,
        full_cases=len(ds),
        expected_shard_cases=len(indices),
        purpose="integration_check" if max_batches is not None else "full_evaluation",
        environment=environment(),
        checkpoint=str(checkpoint) if checkpoint else None,
    )
    if model_name == "outage_fm":
        from outagefm.model import OutageFM

        # A comma-separated checkpoint list evaluates the quantile-averaged ensemble of those members.
        members = []
        info["checkpoint_model_config"], info["checkpoint_updates"] = [], []
        for path in str(checkpoint).split(","):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            saved["config"] = canonical_config(saved["config"])
            check_weather_training_information(saved["config"], config)
            info["checkpoint_model_config"].append(saved["config"]["model"])
            info["checkpoint_updates"].append(saved.get("updates"))
            model_config = copy.deepcopy(saved["config"])
            model_config["evaluation"] = config["evaluation"]
            if config.get("model", {}).get("point"):
                model_config["model"]["point"] = config["model"]["point"]
            member = OutageFM(model_config, saved["normalizer"]).to(device)
            member.load_state_dict(saved["state_dict"], strict=True)
            members.append(member.eval())
        if len(members) == 1:
            model = members[0]
        else:
            from outagefm.model import OutageFMEnsemble

            model = OutageFMEnsemble(
                members,
                config.get("model", {}).get("point", "quantile_mean"),
                config.get("model", {}).get("ensemble_average", "count"),
            ).eval()
    else:
        model = import_baseline().load_for_evaluation(model_name, config, checkpoint, device, info)
    info["host_memory_after_model_load"] = host_memory()
    if not (output / "evaluation-start.json").exists():
        write_json(output / "evaluation-start.json", info)
    progress = output / "progress.pkl"
    accumulator = MaskedCountAccumulator(config["evaluation"]["event_threshold"])
    first = 0
    prediction_seconds = 0.0
    prediction_batches = []
    if progress.exists():
        if not resume:
            raise ValueError("Partial evaluation exists; pass --resume after inspecting it")
        saved = load_pickle(progress)
        accumulator = saved["accumulator"]
        first = saved["next_batch"]
        prediction_seconds = saved.get("prediction_seconds", 0.0)
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    data = DataLoader(
        Subset(ds, indices[first * batch_size :]),
        batch_size=batch_size,
        num_workers=config["evaluation"]["workers"],
        generator=torch.Generator().manual_seed(42),
    )
    start = time.perf_counter()
    last_progress_save = start
    progress_save_seconds = float(config["evaluation"].get("progress_save_seconds", 0))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i, batch in enumerate(data, start=first):
            if max_batches is not None and i >= max_batches:
                break
            safe = forecast_inputs(batch)
            if kind == "outage_fm" or getattr(model, "needs_county_identity", False):
                # Origin and county identity locate pre-origin neighbour histories; no target information.
                safe["county_indices"], safe["origin"] = batch["county_indices"], batch["origin"]
            event_probability = None
            predict_start = time.perf_counter()
            forecast = model.predict(safe)
            if kind == "outage_fm" and model.last_event_probability is not None:
                event_probability = model.last_event_probability.cpu().numpy()
            batch_predict_seconds = time.perf_counter() - predict_start
            prediction_seconds += batch_predict_seconds
            if max_batches is not None:
                prediction_batches.append(batch_predict_seconds)
            accumulator.update(batch, forecast, event_probability)
            if config["evaluation"].get("save_predictions", True):
                payload = dict(
                    target=batch["target"].numpy(),
                    target_mask=batch["target_mask"].numpy(),
                    target_observed_mask=batch["target_observed_mask"].numpy(),
                    point=forecast.point,
                    quantiles=forecast.quantiles,
                    levels=forecast.levels,
                    case_id=np.array(batch["case_id"]),
                    origin=batch["origin"].numpy(),
                    county_indices=batch["county_indices"].numpy(),
                )
                if event_probability is not None:
                    payload["event_probability"] = event_probability
                np.savez_compressed(output / f"predictions-{i:06d}.npz", **payload)
            report_progress = (i + 1) % 25 == 0 or i + 1 == total_batches
            save_progress = report_progress and (
                i + 1 == total_batches or time.perf_counter() - last_progress_save >= progress_save_seconds
            )
            if save_progress:
                saved = dict(
                    accumulator=accumulator,
                    next_batch=i + 1,
                    torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    prediction_seconds=prediction_seconds,
                )
                temp = progress.with_suffix(".tmp")
                with temp.open("wb") as f:
                    pickle.dump(saved, f)
                os.replace(temp, progress)
                last_progress_save = time.perf_counter()
            if report_progress:
                print(
                    json.dumps(
                        dict(
                            batch=i + 1,
                            total_batches=total_batches,
                            elapsed_seconds=time.perf_counter() - start,
                            progress_saved=save_progress,
                        )
                    ),
                    flush=True,
                )
    if not accumulator.cases:
        raise ValueError("No cases scored")
    metrics = accumulator.compute()
    metrics.update(
        max_batches=max_batches,
        shard_complete=max_batches is None and len(accumulator.cases) == len(indices),
        complete=shard_count == 1 and max_batches is None and (len(accumulator.cases) == len(ds)),
    )
    pd.DataFrame(accumulator.cases).to_parquet(output / "cases.parquet", index=False)
    pd.DataFrame(accumulator.events).to_parquet(output / "event-cases.parquet", index=False)
    info.update(elapsed_seconds=time.perf_counter() - start, cases=len(accumulator.cases))
    info.update(host_memory())
    info.update(
        data_loader_start_method=mp.get_start_method(allow_none=True),
        readonly_memmap_spawn_policy="reopen_readonly_files",
    )
    info.update(
        prediction_seconds=prediction_seconds,
        engineering_prediction_batch_seconds=prediction_batches,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        timing_scope="Adapter prediction wall time including device-to-host output; excludes loading and scoring",
    )
    if device == "cuda":
        info.update(
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    write_json(output / "evaluation.json", info)
    with (output / "accumulator.pkl").open("wb") as stream:
        pickle.dump(accumulator, stream)
    write_json(output / "metrics.json", metrics)
    progress.unlink(missing_ok=True)
    return dict(
        output=str(output),
        complete=metrics["complete"],
        cases=metrics["cases"],
        elapsed_seconds=info["elapsed_seconds"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--horizon-hours", type=int, default=48)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = evaluate(
        read_config(args.config),
        args.model,
        args.output,
        args.split,
        args.horizon_hours,
        args.device,
        args.development,
        args.max_batches,
        args.checkpoint,
        args.batch_size,
        args.resume,
        args.shard_index,
        args.shard_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
