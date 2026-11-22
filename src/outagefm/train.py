"""OutageFM training (Section 4.4): historical pretraining, weather adaptation and frozen-backbone head fitting.

  --stage pretrain   2014-2021 outage windows without weather (config protocol: H)
  --stage adapt      issued weather and warnings on adaptation windows (protocol: F); also fits the risk head and
                     the control heads on a frozen backbone (training.trainable_prefixes)

The initialization is explicit: `training.init_checkpoint` in the config (null = random initialization) or
--init-checkpoint, which overrides it. Runs are resumable with --resume last.pt on the same world size.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset

from outagefm.common import (
    atomic_checkpoint,
    canonical_config,
    environment,
    import_baseline,
    read_config,
    read_json,
    seed_everything,
    to_device,
    write_json,
)
from outagefm.data.dataset import FoundationDataset, FoundationPretrainingDataset, primary_normalizer


class Stage(nn.Module):
    def __init__(self, model, pretrain):
        super().__init__()
        self.model = model
        self.pretrain = pretrain

    def forward(self, batch):
        return self.model.pretrain_loss(batch) if self.pretrain else self.model(batch)


def rng_state(device):
    state = np.random.get_state()
    return dict(
        torch=torch.get_rng_state(),
        numpy=[state[0], state[1].tolist(), state[2], state[3], state[4]],
        python=random.getstate(),
        cuda=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else None,
    )


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], np.uint32), n[2], n[3], n[4]))
    random.setstate(state["python"])
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state(state["cuda"], device)


def set_epoch(dataset, sampler, epoch):
    sampler.set_epoch(epoch)
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base, "set_epoch"):
        base.set_epoch(epoch)


def select_subset(dataset, count, seed=42):
    if count >= len(dataset):
        return dataset
    selected = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False))
    return Subset(dataset, selected.tolist())


class RemainingSampler:
    """Keep the exact rank's sample position when a batch size changes."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.offset = 0

    def __iter__(self):
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self):
        return max(0, len(self.sampler) - self.offset)


def check_resume_layout(saved, config, world_size, stage_name, rebatch=False):
    """A saved data cursor requires the same sampler partition and batch layout."""
    previous = saved["config"]
    if len(saved["rng_by_rank"]) != world_size:
        raise ValueError("Resume world size differs from the saved sampler partition")
    if saved["stage"] != stage_name:
        raise ValueError(
            "Resume training stage differs; use explicit checkpoint initialization for a new stage"
        )
    for section, fields in {
        "data": ["panel_dir"],
        "foundation": ["manifest_dir"],
        "windows": ["history_steps"],
        "model": ["kind", "maximum_horizon_steps"],
        "training": ["batch_size", "learning_rate", "backbone_learning_rate", "warmup_updates", "precision"],
    }.items():
        for field in fields:
            if rebatch and section == "training" and field == "batch_size":
                continue
            if previous.get(section, {}).get(field) != config.get(section, {}).get(field):
                raise ValueError(f"Resume data/optimization configuration differs: {section}.{field}")
    for field, default in [("gradient_accumulation", 1), ("fewshot_fraction", 1.0)]:
        if rebatch and field == "gradient_accumulation":
            continue
        if previous["training"].get(field, default) != config["training"].get(field, default):
            raise ValueError(f"Resume training configuration differs: {field}")
    if previous.get("pretraining", {}) != config.get("pretraining", {}):
        raise ValueError("Resume pretraining data/sampling configuration differs")


def train(
    config,
    stage_name="adapt",
    device="cpu",
    output=None,
    init_checkpoint=None,
    resume=None,
    max_cases=None,
    max_updates=None,
    max_seconds=None,
    validation_cases=None,
    rebatch=False,
):
    config = copy.deepcopy(config)
    output = Path(output or config["output_dir"])
    if output.exists() and any(output.iterdir()) and (not resume):
        raise ValueError("Preserve existing training evidence; resume explicitly or choose a new output")
    output.mkdir(parents=True, exist_ok=True)
    tc = config["training"]
    torch.set_num_threads(tc.get("cpu_threads", 2))
    if rebatch and not resume:
        raise ValueError("Explicit rebatching requires an optimizer checkpoint resume")
    limit_updates = max_updates if max_updates is not None else tc["max_updates"]
    limit_seconds = max_seconds if max_seconds is not None else tc["max_seconds"]
    nvalidation = validation_cases if validation_cases is not None else tc["validation_cases"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        if config.get("evaluation", {}).get("matmul_precision"):
            torch.set_float32_matmul_precision(config["evaluation"]["matmul_precision"])
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group("nccl" if str(device).startswith("cuda") else "gloo")
    seed = int(tc.get("seed", 42))  # 42 for every reported model
    seed_everything(seed)
    normalizer = (
        primary_normalizer(config["foundation"]["manifest_dir"])
        if config["pretraining"].get("disjoint_supervision", True)
        else read_json(Path(config["foundation"]["manifest_dir"]) / "normalizer.json")
    )
    kind = config["model"]["kind"]
    if kind == "outage_fm":
        from outagefm.model import OutageFM

        model = OutageFM(config, normalizer).to(device)
    else:
        model = import_baseline().build_for_training(config, normalizer).to(device)
    if init_checkpoint and resume:
        raise ValueError("Choose initialization or optimizer resume, not both")
    if init_checkpoint is None and not resume:
        if stage_name == "adapt" and "init_checkpoint" not in tc:
            raise ValueError(
                "Set training.init_checkpoint in the config (null for random initialization) "
                "or pass --init-checkpoint"
            )
        init_checkpoint = tc.get("init_checkpoint")
    saved = None
    if init_checkpoint or resume:
        saved = torch.load(resume or init_checkpoint, map_location="cpu", weights_only=True)
        saved["config"] = canonical_config(saved["config"])
        if resume:
            from outagefm.evaluate import check_weather_training_information

            check_weather_training_information(saved["config"], config)
            check_resume_layout(saved, config, world, stage_name, rebatch=rebatch)
        if any(saved["normalizer"][key] != normalizer[key] for key in ["log_mean", "log_std"]):
            raise ValueError("Initialization normalization differs")
        initial_state = saved["state_dict"]
        if hasattr(model, "select_initial_state"):
            initial_state = model.select_initial_state(initial_state)
        new_modules = tuple(
            name
            for name, key in (
                ("hazard_head", "hazard_head"),
                ("marginal_head", "marginal_head"),
                ("trajectory_", "trajectory_head"),
                ("warning_context", "warnings"),
                ("warning_future", "warnings"),
            )
            if config["model"].get(key)
        )
        if kind == "outage_fm" and new_modules and not resume:
            # Optional modules (risk/control heads, warning inputs) are new relative to the initializing checkpoint;
            # everything else must match.
            missing, unexpected = model.load_state_dict(initial_state, strict=False)
            if unexpected or any(not k.startswith(new_modules) for k in missing):
                raise ValueError(f"Initialization mismatch: missing {missing}, unexpected {unexpected}")
        else:
            model.load_state_dict(initial_state, strict=True)
    pretrain = stage_name == "pretrain"
    if tc.get("freeze_encoder"):
        for p in model.core.parameters():
            p.requires_grad_(False)
    if tc.get("trainable_prefixes"):
        # Frozen-backbone head fitting: every parameter outside the listed prefixes keeps its initial value.
        prefixes = tuple(tc["trainable_prefixes"])
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith(prefixes))
    if pretrain:
        pc = config["pretraining"]
        dataset = FoundationPretrainingDataset(
            config,
            pc["windows_per_epoch"],
            pc["history_steps"],
            pc["horizon_steps"],
            pc.get("robust_history", False),
            pc.get("disjoint_supervision", True),
        )
    else:
        fraction = tc.get("fewshot_fraction", 1.0)
        if not 0 < fraction <= 1:
            raise ValueError("fewshot_fraction must be in (0,1]")
        # train_split selects another fixed manifest, e.g. geo_adapt-<state>-<days>days for local adaptation;
        # fewshot_fraction < 1 selects a nested label budget (train-target<pct>pct.parquet).
        dataset = FoundationDataset(
            config,
            tc.get("train_split", "train"),
            model.maximum_horizon // 4,
            target_budget_fraction=fraction if fraction < 1 else None,
        )
    if max_cases is not None:
        dataset = select_subset(dataset, min(max_cases, len(dataset)))
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    gpu_inputs = None
    cache_started = time.perf_counter()
    if tc.get("gpu_input_cache", False) and str(device).startswith("cuda"):
        if (pretrain and kind != "outage_fm") or kind not in {
            "deepar",
            "patchtst_xreg",
            "chronos2_adapt",
            "outage_fm",
        }:
            raise ValueError(
                "GPU input assembly is validated for OutageFM and the supervised weather baselines"
            )
        from outagefm.data.gpu_cache import (
            GPUFoundationInputs,
            GPUPretrainingInputs,
            TrainingIndices,
        )

        gpu_inputs = (GPUPretrainingInputs if pretrain else GPUFoundationInputs)(dataset, device)
        cache_seconds = time.perf_counter() - cache_started
        loader_dataset = TrainingIndices(dataset)
    else:
        loader_dataset = dataset
    # Historical replay during weather adaptation: every n-th update draws a batch of 2014-2021 windows without
    # weather (values and availability masks absent) from the disjoint pretraining pool.
    mix_every = 0 if pretrain else int(tc.get("mix_history_every", 0))
    # Positions within each period that draw history windows (default: the last one).
    history_slots = set(tc.get("mix_history_slots", [mix_every - 1]))
    if mix_every:
        if gpu_inputs is None:
            raise ValueError("Historical replay uses the GPU pretraining sampler")
        from outagefm.data.gpu_cache import GPUPretrainingInputs, TrainingIndices

        history_config = copy.deepcopy(config)
        history_config["protocol"] = "H"
        hc = history_config["pretraining"]
        history_dataset = FoundationPretrainingDataset(
            history_config,
            hc["windows_per_epoch"],
            hc["history_steps"],
            hc["horizon_steps"],
            hc.get("robust_history", False),
            hc.get("disjoint_supervision", True),
        )
        history_sampler = DistributedSampler(
            history_dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True
        )
        history_inputs = GPUPretrainingInputs(history_dataset, device)
        history_loader = DataLoader(
            TrainingIndices(history_dataset),
            batch_size=tc["batch_size"],
            sampler=history_sampler,
            num_workers=0,
            drop_last=True,
            generator=torch.Generator().manual_seed(seed + rank),
        )
        history_epoch = 0
        set_epoch(history_dataset, history_sampler, history_epoch)
        history_iterator = iter(history_loader)
    remaining_sampler = RemainingSampler(sampler)
    workers = 0 if gpu_inputs is not None else tc.get("workers", 0)
    data = DataLoader(
        loader_dataset,
        batch_size=tc["batch_size"],
        sampler=remaining_sampler,
        num_workers=workers,
        persistent_workers=workers > 0 and not pretrain,
        pin_memory=str(device).startswith("cuda"),
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    if not len(data):
        raise ValueError("Not enough rows for the configured local batch and world size")
    val = FoundationDataset(config, "validation", model.maximum_horizon // 4, development=True)
    val = select_subset(val, min(nvalidation, len(val)))
    cached_validation = bool(tc.get("gpu_validation_cache", False) and gpu_inputs is not None)
    if tc.get("gpu_validation_cache", False) and not cached_validation:
        raise ValueError("Cached validation requires the authorized GPU source cache")
    if cached_validation:
        from outagefm.data.gpu_cache import TrainingIndices, cached_validation_batch
    validation = (
        DataLoader(
            TrainingIndices(val) if cached_validation else val,
            batch_size=tc.get("validation_batch_size", tc["batch_size"]),
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
        )
        if rank == 0
        else None
    )
    groups = []
    for prefix, rate in [(True, tc["backbone_learning_rate"]), (False, tc["learning_rate"])]:
        params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and name.startswith("core.") == prefix
        ]
        if params:
            groups.append(dict(params=params, lr=rate, initial_lr=rate))
    optimizer = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    train_stage = Stage(model, pretrain)
    if distributed:
        train_stage = nn.parallel.DistributedDataParallel(
            train_stage,
            device_ids=[local] if str(device).startswith("cuda") else None,
            find_unused_parameters=True,
        )
    epoch = cursor = updates = 0
    samples_in_epoch = training_windows_seen = 0
    batch_transitions = []
    best = float("inf")
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, updates, best = (
            saved["epoch"],
            saved["batches_in_epoch"],
            saved["updates"],
            saved["best_validation_loss"],
        )
        previous_training = saved["config"]["training"]
        samples_in_epoch = saved.get("samples_in_epoch", cursor * previous_training["batch_size"])
        previous_global_batch = (
            previous_training["batch_size"] * previous_training.get("gradient_accumulation", 1) * world
        )
        training_windows_seen = saved.get("training_windows_seen", updates * previous_global_batch)
        batch_transitions = list(saved.get("batch_transitions", []))
        if rebatch:
            batch_transitions.append(
                dict(
                    update=updates,
                    epoch=epoch,
                    samples_in_epoch_per_rank=samples_in_epoch,
                    from_per_gpu_batch=previous_training["batch_size"],
                    to_per_gpu_batch=tc["batch_size"],
                    from_accumulation=previous_training.get("gradient_accumulation", 1),
                    to_accumulation=tc.get("gradient_accumulation", 1),
                    world_size=world,
                    previous_global_batch=previous_global_batch,
                    new_global_batch=tc["batch_size"] * tc.get("gradient_accumulation", 1) * world,
                    reason="Explicit rebatch: weights, optimizer, development scoring and rank sample positions retained",
                )
            )
    else:
        torch.manual_seed(seed + rank)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed(seed + rank)
    set_epoch(dataset, sampler, epoch)
    if not 0 <= samples_in_epoch <= len(sampler):
        raise ValueError("Saved sample position is outside this rank's epoch")
    remaining_sampler.offset = samples_in_epoch
    iterator = iter(data)
    if resume:
        restore_rng(saved["rng_by_rank"][rank], device)
    info = dict(
        config=config,
        stage=stage_name,
        seed=seed,
        rank_rng_rule="base seed plus rank after identical model initialization",
        device=str(device),
        world_size=world,
        training_batch_size=data.batch_size,
        validation_batch_size=validation.batch_size if validation is not None else None,
        gradient_accumulation=tc.get("gradient_accumulation", 1),
        global_batch_size=data.batch_size * tc.get("gradient_accumulation", 1) * world,
        environment=environment(),
        parameters=sum((p.numel() for p in model.parameters())),
        trainable_parameters=sum((p.numel() for p in model.parameters() if p.requires_grad)),
        training_rows=len(dataset),
        validation_rows=len(val),
        max_updates=limit_updates,
        max_seconds=limit_seconds,
        initial_checkpoint=init_checkpoint,
        initialization_stage=saved.get("stage") if saved and init_checkpoint else None,
        initialization_update=saved.get("updates") if saved and init_checkpoint else None,
        resumed_checkpoint=resume,
        batch_transitions=batch_transitions,
        resumed_samples_in_epoch_per_rank=samples_in_epoch if resume else None,
        gpu_input_cache_bytes=gpu_inputs.source_bytes if gpu_inputs is not None else None,
        gpu_input_cache_implementation=type(gpu_inputs).__name__ if gpu_inputs is not None else None,
        gpu_input_cache_initialization_seconds=cache_seconds if gpu_inputs is not None else None,
        actual_training_workers=workers,
        purpose="engineering_check" if max_cases is not None else "scientific_training",
    )
    if rank == 0:
        write_json(output / ("resume-run.json" if resume else "run.json"), info)
    accumulation = tc.get("gradient_accumulation", 1)
    # Optional exponential moving average of weights; validated and saved in place of the raw weights.
    ema_decay = tc.get("ema_decay")
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if ema_decay else None
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    # Optional checkpoint selection on a weighted sum of validation loss parts instead of the total.
    selection = tc.get("selection_weights")
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    previous_seconds = saved.get("active_seconds", 0.0) if resume else 0.0
    new_updates = 0
    timings = []
    while updates < limit_updates:
        step_start = time.perf_counter()
        train_stage.train()
        optimizer.zero_grad(set_to_none=True)
        factor = min(1.0, (updates + 1) / max(tc["warmup_updates"], 1))
        if tc.get("lr_schedule") == "cosine":
            # Cosine decay over the planned update budget, floored at a fixed fraction.
            progress = min(1.0, updates / max(tc["schedule_updates"], 1))
            floor = tc.get("lr_floor_fraction", 0.1)
            factor *= float(floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * factor
        loss_sum = 0.0
        parts_sum = {}
        for micro in range(accumulation):
            if mix_every and updates % mix_every in history_slots:
                try:
                    batch = next(history_iterator)
                except StopIteration:
                    history_epoch += 1
                    set_epoch(history_dataset, history_sampler, history_epoch)
                    history_iterator = iter(history_loader)
                    batch = next(history_iterator)
                batch = to_device(history_inputs.batch(batch), device)
                with (
                    train_stage.no_sync()
                    if distributed and micro + 1 < accumulation
                    else contextlib.nullcontext(),
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=str(device).startswith("cuda") and tc.get("precision") == "bfloat16",
                    ),
                ):
                    loss, parts = train_stage(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at {updates}")
                    (loss / accumulation).backward()
                loss_sum += float(loss.detach()) / accumulation
                for key, value in parts.items():
                    parts_sum["history_" + key] = (
                        parts_sum.get("history_" + key, 0.0) + float(value) / accumulation
                    )
                continue
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                cursor = 0
                samples_in_epoch = 0
                remaining_sampler.offset = 0
                set_epoch(dataset, sampler, epoch)
                iterator = iter(data)
                batch = next(iterator)
            cursor += 1
            if gpu_inputs is not None:
                batch = gpu_inputs.batch(batch)
            samples_in_epoch += len(batch["history"])
            training_windows_seen += len(batch["history"]) * world
            batch = to_device(batch, device)
            sync = (
                train_stage.no_sync()
                if distributed and micro + 1 < accumulation
                else contextlib.nullcontext()
            )
            mixed = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if str(device).startswith("cuda") and tc.get("precision") == "bfloat16"
                else contextlib.nullcontext()
            )
            with sync, mixed:
                loss, parts = train_stage(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at {updates}")
                (loss / accumulation).backward()
            loss_sum += float(loss.detach()) / accumulation
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.0) + float(value) / accumulation
        # Parameters under listed prefixes (e.g. a detached point head) are clipped on their own,
        # so their large regression gradients do not shrink the shared encoder update.
        separate = tuple(tc.get("separate_clip_prefixes", ()))
        if separate:
            own = [p for n, p in model.named_parameters() if n.startswith(separate) and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(own, tc["clip_grad_norm"])
        norm = torch.nn.utils.clip_grad_norm_(
            [p for n, p in model.named_parameters() if not (separate and n.startswith(separate))],
            tc["clip_grad_norm"],
        )
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at {updates}")
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in frozen:
                        continue  # frozen parameters keep their exact initial values (no EMA rounding drift)
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)
                    else:
                        ema[k].copy_(v)
        updates += 1
        new_updates += 1
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        duration = time.perf_counter() - step_start
        timings.append(duration)
        elapsed = previous_seconds + time.perf_counter() - started
        stop = elapsed >= limit_seconds or updates >= limit_updates
        if distributed:
            flag = torch.tensor(int(stop), device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            stop = bool(flag.item())
        if tc.get("schedule_from_seconds") and new_updates == 300 and not resume:
            # Time-matched runs: fix the cosine length once from the measured rate (validation time included),
            # taking the minimum over ranks so every rank uses the same schedule.
            rate = new_updates / max(elapsed - previous_seconds, 1e-6)
            estimate = torch.tensor(float(updates + (limit_seconds - elapsed) * rate), device=device)
            if distributed:
                torch.distributed.all_reduce(estimate, op=torch.distributed.ReduceOp.MIN)
            tc["schedule_updates"] = int(estimate.item())
            if rank == 0:
                print(json.dumps(dict(schedule_updates_from_seconds=tc["schedule_updates"])), flush=True)
        validate = updates % tc["validation_every"] == 0 or stop
        save = validate or updates % tc.get("save_every", tc["validation_every"]) == 0
        validation_loss = None
        if validate:
            if distributed:
                torch.distributed.barrier()
            if rank == 0:
                model.eval()
                if ema is not None:
                    raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.load_state_dict({k: v.to(raw_state[k].dtype) for k, v in ema.items()})
                validation_parts = {}
                with torch.random.fork_rng(devices=[local] if str(device).startswith("cuda") else []):
                    torch.manual_seed(42)
                    if str(device).startswith("cuda"):
                        torch.cuda.manual_seed(42)
                    total = count = 0
                    with torch.inference_mode():
                        for batch in validation:
                            if cached_validation:
                                batch = cached_validation_batch(gpu_inputs, val, batch)
                            with torch.autocast(
                                "cuda",
                                dtype=torch.bfloat16,
                                enabled=str(device).startswith("cuda") and tc.get("precision") == "bfloat16",
                            ):
                                value, parts = Stage(model, pretrain)(to_device(batch, device))
                            if selection:
                                value = sum(float(w) * parts[k] for k, w in selection.items())
                            for key, part in parts.items():
                                validation_parts[key] = validation_parts.get(key, 0.0) + float(part) * len(
                                    batch["history"]
                                )
                            total += float(value) * len(batch["history"])
                            count += len(batch["history"])
                    validation_loss = total / max(count, 1)
                    validation_parts = {
                        f"validation_{k}": v / max(count, 1) for k, v in validation_parts.items()
                    }
            if distributed:
                torch.distributed.barrier()
        if save:
            local_rng = rng_state(device)
            rngs = [None] * world if distributed else [local_rng]
            if distributed:
                torch.distributed.all_gather_object(rngs, local_rng)
            if rank == 0:
                improved = validation_loss is not None and validation_loss < best
                if improved:
                    best = validation_loss
                payload = dict(
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    raw_state_dict=None
                    if ema is None or not validate
                    else {k: v.cpu() for k, v in raw_state.items()},
                    optimizer=optimizer.state_dict(),
                    config=config,
                    normalizer=normalizer,
                    stage=stage_name,
                    updates=updates,
                    active_seconds=previous_seconds + time.perf_counter() - started,
                    epoch=epoch,
                    batches_in_epoch=cursor,
                    samples_in_epoch=samples_in_epoch,
                    training_windows_seen=training_windows_seen,
                    batch_transitions=batch_transitions,
                    best_validation_loss=best,
                    rng_by_rank=rngs,
                )
                atomic_checkpoint(output / "last.pt", payload)
                if improved:
                    temporary = output / "best-link.tmp"
                    temporary.unlink(missing_ok=True)
                    os.link(output / "last.pt", temporary)
                    os.replace(temporary, output / "best.pt")
            if rank == 0 and ema is not None and validate:
                model.load_state_dict(raw_state)
            if distributed:
                torch.distributed.barrier()
        if rank == 0 and (updates % 10 == 0 or save):
            row = dict(
                update=updates,
                epoch=epoch,
                batches_in_epoch=cursor,
                training_loss=loss_sum,
                validation_loss=validation_loss,
                elapsed_seconds=elapsed,
                step_seconds=duration,
                gradient_norm=float(norm),
                **parts_sum,
                **(validation_parts if validation_loss is not None else {}),
            )
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if stop:
            break
    gpu_peaks = None
    if str(device).startswith("cuda"):
        peaks = torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            dtype=torch.int64,
            device=device,
        )
        if distributed:
            torch.distributed.all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
        gpu_peaks = peaks.cpu().tolist()
    result = dict(
        updates=updates,
        new_updates=new_updates,
        elapsed_seconds=time.perf_counter() - started,
        active_seconds=previous_seconds + time.perf_counter() - started,
        mean_update_seconds=float(np.mean(timings)) if timings else None,
        median_update_seconds=float(np.median(timings)) if timings else None,
        world_size=world,
        device=str(device),
        best_validation_loss=best if rank == 0 else None,
        training_windows_seen=training_windows_seen,
        batch_transitions=batch_transitions,
        purpose=info["purpose"],
        output=str(output),
    )
    if gpu_peaks is not None:
        result.update(
            maximum_rank_cuda_peak_allocated_bytes=gpu_peaks[0],
            maximum_rank_cuda_peak_reserved_bytes=gpu_peaks[1],
            memory_scope="Maximum rank allocator peak within this training invocation, including validation/checkpointing; excludes final export",
        )
    if gpu_inputs is not None and hasattr(gpu_inputs, "parity_checked"):
        result["pretraining_cpu_parity_cases_per_rank"] = 8 if gpu_inputs.parity_checked else 0
    if cached_validation and rank == 0:
        result["validation_cpu_parity_cases"] = (
            min(8, len(val)) if getattr(gpu_inputs, "validation_parity_checked", False) else 0
        )
    if rank == 0:
        write_json(output / "training-summary.json", result)
        if hasattr(model, "export") and (output / "best.pt").exists():
            model.export(output)
    if distributed:
        torch.distributed.barrier()
    if distributed:
        torch.distributed.destroy_process_group()
    return result


def apply_batch_layout(config, world, batch_size=None, global_batch_size=None):
    """Per-GPU batch and gradient accumulation for the current world size (both are recorded in run.json)."""
    tc = config["training"]
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("--batch-size must be positive")
        tc["batch_size"] = batch_size
    if global_batch_size is not None:
        per_update = world * tc["batch_size"]
        if global_batch_size % per_update:
            raise ValueError(
                f"Global batch {global_batch_size} is not a multiple of world size {world} x "
                f"per-GPU batch {tc['batch_size']}"
            )
        tc["gradient_accumulation"] = global_batch_size // per_update
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["pretrain", "adapt"], default="adapt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--init-checkpoint", help="Overrides training.init_checkpoint of the config")
    parser.add_argument("--resume")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--validation-cases", type=int)
    parser.add_argument(
        "--rebatch",
        action="store_true",
        help="Explicitly change batch layout while retaining rank sample positions and optimizer state",
    )
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size (per GPU)")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Keep this global batch on any number of GPUs by setting training.gradient_accumulation "
        "= global / (world size x per-GPU batch); e.g. 8192 reproduces the 4 x 2048 layout",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config = apply_batch_layout(
        read_config(args.config),
        int(os.environ.get("WORLD_SIZE", "1")),
        args.batch_size,
        args.global_batch_size,
    )
    result = train(
        config,
        args.stage,
        args.device,
        args.output,
        args.init_checkpoint,
        args.resume,
        args.max_cases,
        args.max_updates,
        args.max_seconds,
        args.validation_cases,
        args.rebatch,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
