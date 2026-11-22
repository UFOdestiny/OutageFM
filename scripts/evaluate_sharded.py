"""Evaluate an OutageFM checkpoint on fixed manifests in GPU shards, then merge.

Usage: evaluate_sharded.py CONFIG CHECKPOINT OUTPUT_ROOT SPLIT[,SPLIT] HORIZON[,HORIZON]
           [--development] [--model NAME] [--shards N] [--gpus N] [--workers N] [--module MODULE]

NAME defaults to outage_fm (other model kinds need the optional baseline/ package, not part of this release).
CHECKPOINT may be a comma-separated list of member checkpoints, which evaluates their ensemble.
--shards is the number of manifest shards (default 4, the protocol used for every reported result: shard k scores
cases k, k+N, ...). Shards run concurrently and are placed round-robin on the visible GPUs (--gpus, default all
visible), so 4 shards on 2 GPUs reproduce the 4-GPU shard layout. --workers overrides evaluation.shard_workers
(DataLoader processes per shard; it does not change results).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from outagefm.common import read_config, read_json, write_json


def visible_gpus():
    """Physical GPU identifiers this process may use, honouring an inherited CUDA_VISIBLE_DEVICES."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        return [g.strip() for g in inherited.split(",") if g.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("root")
    parser.add_argument("splits")
    parser.add_argument("horizons")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--model", default="outage_fm")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--module",
        default="outagefm.evaluate",
        help="Evaluation entry module (default outagefm.evaluate)",
    )
    args = parser.parse_args()
    gpus = visible_gpus()
    if not gpus:
        raise SystemExit("No visible CUDA device")
    if args.gpus is not None:
        if not 1 <= args.gpus <= len(gpus):
            raise SystemExit(f"--gpus must be in 1..{len(gpus)}")
        gpus = gpus[: args.gpus]
    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    root = Path(args.root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    # A saved root config.json is read as JSON (YAML 1.1 would read values such as 1e-05 as strings).
    config = read_json(args.config) if args.config.endswith(".json") else read_config(args.config)
    workers = args.workers if args.workers is not None else int(config["evaluation"].get("shard_workers", 3))
    config["evaluation"]["workers"] = workers
    if (root / "config.json").exists():
        # Shards record the full configuration and merge requires them to agree, so a resumed root keeps the
        # configuration (including the loader worker count) its existing shards were written with.
        previous = read_json(root / "config.json")
        if workers != previous["evaluation"].get("workers"):
            print(
                f"Resuming {root}: keeping evaluation.workers={previous['evaluation'].get('workers')}",
                flush=True,
            )
        config["evaluation"]["workers"] = previous["evaluation"].get("workers")
        if previous != config:
            raise SystemExit(f"{root / 'config.json'} differs from {args.config}; choose a new output root")
    else:
        write_json(root / "config.json", config)
    records = read_json(root / "completed-groups.json") if (root / "completed-groups.json").exists() else []
    threads = max(1, (os.cpu_count() or 2) // max(1, args.shards))
    for horizon in [int(h) for h in args.horizons.split(",")]:
        for split in args.splits.split(","):
            output = root / f"h{horizon}" / split
            if (output / "metrics.json").exists():
                continue
            began = time.perf_counter()
            children, streams = [], []
            try:
                for rank in range(args.shards):
                    shard = output / f"shard-{rank}"
                    if (shard / "metrics.json").exists() and read_json(shard / "metrics.json").get(
                        "shard_complete"
                    ):
                        continue
                    env = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": gpus[rank % len(gpus)],
                        "OMP_NUM_THREADS": str(min(2, threads)),
                        "MKL_NUM_THREADS": str(min(2, threads)),
                        "OPENBLAS_NUM_THREADS": "1",
                    }
                    command = [
                        sys.executable,
                        "-m",
                        args.module,
                        "--config",
                        str(root / "config.json"),
                        "--model",
                        args.model,
                        "--checkpoint",
                        str(args.checkpoint),
                        "--output",
                        str(shard),
                        "--split",
                        split,
                        "--horizon-hours",
                        str(horizon),
                        "--device",
                        "cuda",
                        "--batch-size",
                        str(config["evaluation"]["batch_size"]),
                        "--shard-index",
                        str(rank),
                        "--shard-count",
                        str(args.shards),
                    ]
                    if args.development:
                        command.append("--development")
                    if shard.exists() and any(shard.iterdir()):
                        command.append("--resume")
                    stream = (root / "logs" / f"h{horizon}-{split}-{rank}.log").open("w", encoding="utf-8")
                    streams.append(stream)
                    children.append(
                        subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    )
                for child in children:
                    if child.wait(timeout=6 * 3600):
                        raise RuntimeError(
                            f"Evaluation failed for h{horizon} {split}; inspect {root / 'logs'}"
                        )
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for stream in streams:
                    stream.close()
            subprocess.run([sys.executable, "-m", "outagefm.merge", str(output)], check=True, timeout=600)
            records.append(
                dict(
                    horizon_hours=horizon,
                    split=split,
                    cases=read_json(output / "metrics.json")["cases"],
                    wall_seconds=time.perf_counter() - began,
                    shards=args.shards,
                    gpus=len(gpus),
                )
            )
            write_json(root / "completed-groups.json", records)
            print(records[-1], flush=True)


if __name__ == "__main__":
    main()
