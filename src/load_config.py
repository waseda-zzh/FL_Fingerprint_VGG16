from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    mode: str
    root: str
    train_subdir: str
    val_subdir: str
    download_url: str


@dataclass
class FederationConfig:
    num_clients: int
    dirichlet_alpha: float
    partition_seed: int
    partition_cache_path: Optional[str]
    global_rounds: int
    local_epochs: int
    clients_per_round: int
    client_sampling: str
    shuffle_client_order: bool


@dataclass
class TrainConfig:
    input_size: int
    use_random_resized_crop: bool
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    num_workers: int
    use_amp: bool
    # If set, each client local training stops after this many mini-batches (across local epochs).
    # Useful for CPU smoke tests; use null for full passes.
    max_batches_per_client: Optional[int] = None


@dataclass
class ModelConfig:
    name: str
    num_classes: int
    pretrained_weights: Optional[str]


@dataclass
class EvalConfig:
    enabled: bool
    interval_rounds: int
    batch_size: int
    # If set, stop eval after this many batches (forward-only); null evaluates full loader.
    max_batches: Optional[int] = None


@dataclass
class LogConfig:
    interval_batches: int


@dataclass
class CheckpointConfig:
    dir: str
    save_every_rounds: int


@dataclass
class ClientUploadsConfig:
    """Save each client's FedAvg upload (post-local-train weights + metadata) per round."""

    enabled: bool
    dir: str


@dataclass
class MetricsConfig:
    """Append per-round training loss and (optional) eval accuracy to jsonl + csv."""

    enabled: bool
    dir: str
    jsonl_filename: str
    csv_filename: str


@dataclass
class FLExperimentConfig:
    seed: int
    data: DataConfig
    federation: FederationConfig
    train: TrainConfig
    model: ModelConfig
    eval: EvalConfig
    log: LogConfig
    checkpoint: CheckpointConfig
    client_uploads: ClientUploadsConfig
    metrics: MetricsConfig


def _req(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(".".join(path))
        cur = cur[p]
    return cur


def load_fl_config(path: str | Path) -> FLExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    data = DataConfig(
        mode=str(_req(raw, "data", "mode")),
        root=str(_req(raw, "data", "root")),
        train_subdir=str(_req(raw, "data", "train_subdir")),
        val_subdir=str(_req(raw, "data", "val_subdir")),
        download_url=str(_req(raw, "data", "download_url")),
    )

    fed = FederationConfig(
        num_clients=int(_req(raw, "federation", "num_clients")),
        dirichlet_alpha=float(_req(raw, "federation", "dirichlet_alpha")),
        partition_seed=int(_req(raw, "federation", "partition_seed")),
        partition_cache_path=raw.get("federation", {}).get("partition_cache_path"),
        global_rounds=int(_req(raw, "federation", "global_rounds")),
        local_epochs=int(_req(raw, "federation", "local_epochs")),
        clients_per_round=int(_req(raw, "federation", "clients_per_round")),
        client_sampling=str(_req(raw, "federation", "client_sampling")),
        shuffle_client_order=bool(_req(raw, "federation", "shuffle_client_order")),
    )

    train_raw = raw.get("train", {})
    if not isinstance(train_raw, dict):
        raise ValueError("train must be a mapping")

    train = TrainConfig(
        input_size=int(_req(raw, "train", "input_size")),
        use_random_resized_crop=bool(_req(raw, "train", "use_random_resized_crop")),
        batch_size=int(_req(raw, "train", "batch_size")),
        lr=float(_req(raw, "train", "lr")),
        momentum=float(_req(raw, "train", "momentum")),
        weight_decay=float(_req(raw, "train", "weight_decay")),
        num_workers=int(_req(raw, "train", "num_workers")),
        use_amp=bool(_req(raw, "train", "use_amp")),
        max_batches_per_client=(
            int(train_raw["max_batches_per_client"])
            if train_raw.get("max_batches_per_client") is not None
            else None
        ),
    )

    model = ModelConfig(
        name=str(_req(raw, "model", "name")),
        num_classes=int(_req(raw, "model", "num_classes")),
        pretrained_weights=raw.get("model", {}).get("pretrained_weights"),
    )

    eval_raw = raw.get("eval", {})
    if not isinstance(eval_raw, dict):
        raise ValueError("eval must be a mapping")

    ev = EvalConfig(
        enabled=bool(_req(raw, "eval", "enabled")),
        interval_rounds=int(_req(raw, "eval", "interval_rounds")),
        batch_size=int(_req(raw, "eval", "batch_size")),
        max_batches=(
            int(eval_raw["max_batches"]) if eval_raw.get("max_batches") is not None else None
        ),
    )

    log = LogConfig(interval_batches=int(_req(raw, "log", "interval_batches")))

    ckpt = CheckpointConfig(
        dir=str(_req(raw, "checkpoint", "dir")),
        save_every_rounds=int(_req(raw, "checkpoint", "save_every_rounds")),
    )

    cu_raw = raw.get("client_uploads")
    if cu_raw is None:
        client_uploads = ClientUploadsConfig(enabled=False, dir="./artifacts/client_uploads")
    else:
        if not isinstance(cu_raw, dict):
            raise ValueError("client_uploads must be a mapping")
        client_uploads = ClientUploadsConfig(
            enabled=bool(_req(raw, "client_uploads", "enabled")),
            dir=str(_req(raw, "client_uploads", "dir")),
        )

    m_raw = raw.get("metrics")
    if m_raw is None:
        metrics = MetricsConfig(
            enabled=False,
            dir="./artifacts/metrics",
            jsonl_filename="training_curves.jsonl",
            csv_filename="training_curves.csv",
        )
    else:
        if not isinstance(m_raw, dict):
            raise ValueError("metrics must be a mapping")
        metrics = MetricsConfig(
            enabled=bool(_req(raw, "metrics", "enabled")),
            dir=str(_req(raw, "metrics", "dir")),
            jsonl_filename=str(_req(raw, "metrics", "jsonl_filename")),
            csv_filename=str(_req(raw, "metrics", "csv_filename")),
        )

    cfg = FLExperimentConfig(
        seed=int(_req(raw, "seed")),
        data=data,
        federation=fed,
        train=train,
        model=model,
        eval=ev,
        log=log,
        checkpoint=ckpt,
        client_uploads=client_uploads,
        metrics=metrics,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: FLExperimentConfig) -> None:
    if cfg.federation.num_clients < 1:
        raise ValueError("federation.num_clients must be >= 1")
    if cfg.federation.clients_per_round < 1:
        raise ValueError("federation.clients_per_round must be >= 1")
    if cfg.federation.dirichlet_alpha <= 0:
        raise ValueError("federation.dirichlet_alpha must be > 0")
    if cfg.data.mode not in ("tiny_imagenet", "imagenet_full"):
        raise ValueError(f"Unknown data.mode: {cfg.data.mode}")
    if cfg.model.num_classes < 2:
        raise ValueError("model.num_classes must be >= 2")
    if cfg.train.max_batches_per_client is not None and cfg.train.max_batches_per_client < 1:
        raise ValueError("train.max_batches_per_client must be >= 1 when set")
    if cfg.eval.max_batches is not None and cfg.eval.max_batches < 1:
        raise ValueError("eval.max_batches must be >= 1 when set")
    if cfg.client_uploads.enabled and not str(cfg.client_uploads.dir).strip():
        raise ValueError("client_uploads.dir must be non-empty when client_uploads.enabled is true")
    if cfg.metrics.enabled:
        if not str(cfg.metrics.dir).strip():
            raise ValueError("metrics.dir must be non-empty when metrics.enabled is true")
        if not str(cfg.metrics.jsonl_filename).strip():
            raise ValueError("metrics.jsonl_filename must be non-empty when metrics.enabled is true")
        if not str(cfg.metrics.csv_filename).strip():
            raise ValueError("metrics.csv_filename must be non-empty when metrics.enabled is true")


def summarize_config(cfg: FLExperimentConfig) -> str:
    lines = [
        f"seed={cfg.seed}",
        f"data.mode={cfg.data.mode} root={cfg.data.root}",
        f"federation K={cfg.federation.num_clients} alpha={cfg.federation.dirichlet_alpha} "
        f"partition_seed={cfg.federation.partition_seed} rounds={cfg.federation.global_rounds} "
        f"local_epochs={cfg.federation.local_epochs} m={cfg.federation.clients_per_round} "
        f"sampling={cfg.federation.client_sampling}",
        f"model {cfg.model.name} num_classes={cfg.model.num_classes}",
        f"train batch={cfg.train.batch_size} lr={cfg.train.lr} input={cfg.train.input_size} "
        f"max_batches_per_client={cfg.train.max_batches_per_client}",
        f"eval enabled={cfg.eval.enabled} interval_rounds={cfg.eval.interval_rounds} "
        f"max_batches={cfg.eval.max_batches}",
        f"client_uploads enabled={cfg.client_uploads.enabled} dir={cfg.client_uploads.dir}",
        f"metrics enabled={cfg.metrics.enabled} dir={cfg.metrics.dir} "
        f"jsonl={cfg.metrics.jsonl_filename} csv={cfg.metrics.csv_filename}",
    ]
    return "\n".join(lines)
