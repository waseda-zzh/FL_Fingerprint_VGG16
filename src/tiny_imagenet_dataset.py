from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpeg", ".jpg", ".png"))


def discover_tiny_imagenet_train(root: Path) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing Tiny ImageNet train directory: {train_dir}")

    class_names = sorted([d for d in os.listdir(train_dir) if (train_dir / d).is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples: List[Tuple[str, int]] = []
    for cls in class_names:
        img_dir = train_dir / cls / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing images folder: {img_dir}")
        for fname in sorted(os.listdir(img_dir)):
            if not _is_image_file(fname):
                continue
            samples.append((str(img_dir / fname), class_to_idx[cls]))
    if not samples:
        raise RuntimeError(f"No training images found under {train_dir}")
    return samples, class_to_idx


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, root: str | Path, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.samples, self.class_to_idx = discover_tiny_imagenet_train(self.root)
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)


class TinyImageNetValDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        ann_path = self.root / "val" / "val_annotations.txt"
        images_dir = self.root / "val" / "images"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing val annotations: {ann_path}")
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing val images directory: {images_dir}")

        self.samples: List[Tuple[str, int]] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fname, wnid = parts[0], parts[1]
            if wnid not in self.class_to_idx:
                raise KeyError(f"Unknown wnid in val annotations: {wnid}")
            self.samples.append((str(images_dir / fname), int(self.class_to_idx[wnid])))
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(y)
