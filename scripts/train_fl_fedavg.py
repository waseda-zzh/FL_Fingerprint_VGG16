#!/usr/bin/env python3
"""Simulated federated learning (FedAvg) for VGG16 image classification.

Tiny ImageNet layout (default):
  data.root/
    train/<wnid>/images/*.JPEG
    val/images/*.JPEG
    val/val_annotations.txt

Full ImageNet layout (data.mode=imagenet_full):
  data.root/train/<wnid>/*.JPEG
  data.root/val/<wnid>/*.JPEG   (optional for eval)

Run from repo root:
  conda activate fl_blackbox
  python scripts/train_fl_fedavg.py --config configs/fl_imagenet_vgg16.yaml --device cuda

`--device cuda` fails fast if PyTorch was built without CUDA or no GPU is visible.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client_update import train_one_client
from src.datasets import build_train_dataset, build_val_dataset, dataset_targets
from src.dirichlet_partition import (
    dirichlet_partition_indices,
    load_partition,
    partition_cache_path_default,
    save_partition,
)
from src.imagenet_transforms import build_eval_transforms, build_train_transforms
from src.client_uploads import save_client_upload_bundle
from src.load_config import load_fl_config, summarize_config
from src.training_metrics import append_metrics_jsonl, append_round_metrics_csv
from src.model_factory import build_model
from src.server_agg import fedavg_weighted


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_topk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: tuple[int, ...],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = {f"top{k}": 0 for k in topk}
    n = 0
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        max_k = max(topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(y.view(-1, 1).expand_as(pred))
        for k in topk:
            totals[f"top{k}"] += int(correct[:, :k].any(dim=1).sum().item())
        n += int(y.numel())
        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    return {name: float(totals[name]) / max(n, 1) for name in totals}


def select_clients(num_clients: int, m: int, rng: np.random.Generator, sampling: str) -> list[int]:
    if sampling != "random_uniform":
        raise ValueError(f"Unsupported federation.client_sampling: {sampling}")
    replace = m > num_clients
    return [int(x) for x in rng.choice(num_clients, size=int(m), replace=replace)]


def partition_path_for(cfg, artifacts_dir: Path) -> Path:
    if cfg.federation.partition_cache_path:
        return Path(cfg.federation.partition_cache_path).expanduser().resolve()
    return partition_cache_path_default(
        artifacts_dir=artifacts_dir,
        mode=cfg.data.mode,
        num_clients=cfg.federation.num_clients,
        alpha=cfg.federation.dirichlet_alpha,
        partition_seed=cfg.federation.partition_seed,
    )


def load_or_build_partition(cfg, train_ds: Dataset, artifacts_dir: Path) -> list[list[int]]:
    path = partition_path_for(cfg, artifacts_dir)
    labels = dataset_targets(train_ds)
    n = len(train_ds)

    if path.is_file():
        client_indices = load_partition(path)
        if len(client_indices) != cfg.federation.num_clients:
            raise ValueError("Cached partition num_clients mismatch")
        assigned = sum(len(x) for x in client_indices)
        mx = max((max(x) for x in client_indices if x), default=-1)
        if assigned != n or mx >= n or mx < 0:
            raise ValueError("Cached partition does not match dataset size; delete cache and retry")
        return client_indices

    client_indices = dirichlet_partition_indices(
        labels=labels,
        num_clients=cfg.federation.num_clients,
        alpha=cfg.federation.dirichlet_alpha,
        seed=cfg.federation.partition_seed,
    )
    save_partition(path, client_indices)
    return client_indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "fl_imagenet_vgg16.yaml"),
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: auto picks CUDA when available; cuda requires a GPU",
    )
    args = parser.parse_args()

    cfg = load_fl_config(args.config)
    print("=== FL config ===", flush=True)
    print(summarize_config(cfg), flush=True)
    print("=================", flush=True)

    set_seed(cfg.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Requested --device cuda but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch (see environment.yml) and ensure a GPU is visible."
            )
        device = torch.device("cuda", index=0)
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"device={device} name={props.name} capability={props.major}.{props.minor} "
            f"total_mem_GiB={props.total_memory / (1024**3):.2f}",
            flush=True,
        )
    else:
        print(f"device={device}", flush=True)

    train_tf = build_train_transforms(
        input_size=cfg.train.input_size,
        use_random_resized_crop=cfg.train.use_random_resized_crop,
    )
    eval_tf = build_eval_transforms(input_size=cfg.train.input_size)

    train_ds = build_train_dataset(cfg.data, transform=train_tf)
    val_ds, _tiny_meta = build_val_dataset(cfg.data, transform=eval_tf)

    artifacts_dir = (ROOT / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    client_indices = load_or_build_partition(cfg, train_ds, artifacts_dir)
    print(f"partition_cache={partition_path_for(cfg, artifacts_dir)}", flush=True)

    upload_root: Path | None = None
    if cfg.client_uploads.enabled:
        upload_root = Path(cfg.client_uploads.dir).expanduser().resolve()
        upload_root.mkdir(parents=True, exist_ok=True)
        print(f"client_uploads.root={upload_root}", flush=True)

    metrics_jsonl: Path | None = None
    metrics_csv: Path | None = None
    if cfg.metrics.enabled:
        mdir = Path(cfg.metrics.dir).expanduser().resolve()
        mdir.mkdir(parents=True, exist_ok=True)
        metrics_jsonl = mdir / cfg.metrics.jsonl_filename
        metrics_csv = mdir / cfg.metrics.csv_filename
        print(f"metrics.jsonl={metrics_jsonl}", flush=True)
        print(f"metrics.csv={metrics_csv}", flush=True)

    global_model = build_model(cfg.model.name, cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(int(cfg.seed))

    val_loader = None
    if cfg.eval.enabled and val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    local_model = copy.deepcopy(global_model)

    for rnd in range(1, cfg.federation.global_rounds + 1):
        selected = select_clients(
            num_clients=cfg.federation.num_clients,
            m=cfg.federation.clients_per_round,
            rng=rng,
            sampling=cfg.federation.client_sampling,
        )
        if cfg.federation.shuffle_client_order:
            rng.shuffle(selected)

        state_dicts: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        round_loss_num = 0.0
        round_loss_den = 0
        round_participants: list[dict[str, int | float]] = []

        pbar = tqdm(selected, desc=f"round {rnd}/{cfg.federation.global_rounds}", leave=False)
        upload_slot = 0
        for cid in pbar:
            subset = Subset(train_ds, client_indices[cid])
            if len(subset) == 0:
                continue
            loader = DataLoader(
                subset,
                batch_size=cfg.train.batch_size,
                shuffle=True,
                num_workers=cfg.train.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                lr=cfg.train.lr,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
            )

            avg_loss, n = train_one_client(
                model=local_model,
                loader=loader,
                device=device,
                epochs=cfg.federation.local_epochs,
                optimizer=optimizer,
                criterion=criterion,
                use_amp=cfg.train.use_amp,
                max_batches=cfg.train.max_batches_per_client,
            )

            sd_cpu = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
            state_dicts.append(sd_cpu)
            weights.append(float(n))
            round_loss_num += avg_loss * float(n)
            round_loss_den += int(n)
            pbar.set_postfix(client=int(cid), loss=float(avg_loss))

            if cfg.client_uploads.enabled and upload_root is not None:
                save_client_upload_bundle(
                    upload_root,
                    client_id=int(cid),
                    round_idx=int(rnd),
                    upload_index=int(upload_slot),
                    state_dict=sd_cpu,
                    meta={
                        "round": int(rnd),
                        "client_id": int(cid),
                        "upload_index_in_round": int(upload_slot),
                        "subset_num_indices": int(len(client_indices[cid])),
                        "num_samples_used_for_loss": int(n),
                        "fedavg_weight": float(n),
                        "avg_local_loss": float(avg_loss),
                        "local_epochs": int(cfg.federation.local_epochs),
                        "train_batch_size": int(cfg.train.batch_size),
                        "max_batches_per_client": cfg.train.max_batches_per_client,
                    },
                )

            round_participants.append(
                {
                    "client_id": int(cid),
                    "upload_slot_in_round": int(upload_slot),
                    "avg_local_loss": float(avg_loss),
                    "num_samples_trained": int(n),
                    "subset_num_indices": int(len(client_indices[cid])),
                }
            )
            upload_slot += 1

        if not state_dicts:
            raise RuntimeError("No clients contributed in this round (empty subsets?)")

        merged = fedavg_weighted(state_dicts, weights)
        global_model.load_state_dict(merged)

        round_loss = round_loss_num / max(round_loss_den, 1)
        print(f"round={rnd} train_loss={round_loss:.4f} participants={len(state_dicts)}")

        eval_metrics: dict[str, float] | None = None
        if (
            cfg.eval.enabled
            and val_loader is not None
            and (rnd % max(cfg.eval.interval_rounds, 1) == 0)
        ):
            topk = (1, 5) if cfg.model.num_classes >= 5 else (1,)
            eval_metrics = evaluate_topk(
                global_model,
                val_loader,
                device,
                topk=topk,
                max_batches=cfg.eval.max_batches,
            )
            msg = " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
            print(f"round={rnd} eval {msg}")

        if cfg.metrics.enabled and metrics_jsonl is not None and metrics_csv is not None:
            top1 = float(eval_metrics["top1"]) if eval_metrics and "top1" in eval_metrics else None
            top5 = float(eval_metrics["top5"]) if eval_metrics and "top5" in eval_metrics else None
            append_metrics_jsonl(
                metrics_jsonl,
                {
                    "round": int(rnd),
                    "train_loss_weighted": float(round_loss),
                    "num_participants": int(len(state_dicts)),
                    "eval": eval_metrics,
                    "participants": round_participants,
                },
            )
            append_round_metrics_csv(
                metrics_csv,
                round_idx=int(rnd),
                train_loss_weighted=float(round_loss),
                eval_top1=top1,
                eval_top5=top5,
                num_participants=int(len(state_dicts)),
            )

        ckpt_dir = Path(cfg.checkpoint.dir).expanduser().resolve()
        if cfg.checkpoint.save_every_rounds and (rnd % cfg.checkpoint.save_every_rounds == 0):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"global_r{rnd}.pt"
            torch.save({"model": global_model.state_dict(), "round": rnd}, ckpt_path)
            print(f"saved {ckpt_path}")

    print("done")


if __name__ == "__main__":
    # Helps reproducibility for conv kernels on GPU
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    main()
