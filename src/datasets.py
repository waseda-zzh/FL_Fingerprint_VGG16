from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.load_config import DataConfig
from src.tiny_imagenet_dataset import TinyImageNetTrainDataset, TinyImageNetValDataset


def build_train_dataset(cfg: DataConfig, transform: Optional[Callable] = None) -> Dataset:
    root = Path(cfg.root).expanduser().resolve()
    if cfg.mode == "tiny_imagenet":
        return TinyImageNetTrainDataset(root, transform=transform)
    if cfg.mode == "imagenet_full":
        train_root = root / cfg.train_subdir
        if not train_root.is_dir():
            raise FileNotFoundError(f"Missing ImageNet train directory: {train_root}")
        return ImageFolder(str(train_root), transform=transform)
    raise ValueError(f"Unknown data.mode: {cfg.mode}")


def build_val_dataset(
    cfg: DataConfig,
    transform: Optional[Callable] = None,
) -> Tuple[Optional[Dataset], Optional[dict]]:
    """Returns (val_dataset, tiny_class_to_idx_or_none)."""
    root = Path(cfg.root).expanduser().resolve()
    if cfg.mode == "tiny_imagenet":
        train_ds = TinyImageNetTrainDataset(root, transform=None)
        val_ds = TinyImageNetValDataset(root, train_ds.class_to_idx, transform=transform)
        return val_ds, train_ds.class_to_idx
    if cfg.mode == "imagenet_full":
        val_root = root / cfg.val_subdir
        if not val_root.is_dir():
            return None, None
        return ImageFolder(str(val_root), transform=transform), None
    raise ValueError(f"Unknown data.mode: {cfg.mode}")


def dataset_targets(dataset: Dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        t = getattr(dataset, "targets")
        return [int(x) for x in t]
    raise AttributeError("Dataset has no targets attribute; cannot partition.")
