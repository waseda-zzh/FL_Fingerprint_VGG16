from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np


def dirichlet_partition_indices(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """Split dataset indices across clients with label-wise Dirichlet allocation.

    For each class c, draw p ~ Dir(alpha,...,alpha) (length K) and allocate class-c
    samples to clients with counts ~ Multinomial(n_c, p). This is the standard
    non-IID construction used in federated learning literature.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1:
        raise ValueError("labels must be 1-D")
    n = int(labels_arr.shape[0])
    rng = np.random.default_rng(int(seed))

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    classes = np.unique(labels_arr)

    for c in classes.tolist():
        idx_c = np.where(labels_arr == int(c))[0]
        rng.shuffle(idx_c)
        if idx_c.size == 0:
            continue
        p = rng.dirichlet(alpha=np.full(num_clients, float(alpha), dtype=np.float64))
        counts = rng.multinomial(int(idx_c.size), p)
        start = 0
        for i in range(num_clients):
            end = start + int(counts[i])
            if end > start:
                client_indices[i].extend(idx_c[start:end].astype(int).tolist())
            start = end
        assert start == int(idx_c.size)

    assigned = sum(len(ci) for ci in client_indices)
    if assigned != n:
        raise RuntimeError(f"Partition size mismatch: {assigned} != {n}")

    return client_indices


def partition_cache_path_default(
    artifacts_dir: Path,
    mode: str,
    num_clients: int,
    alpha: float,
    partition_seed: int,
) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_mode = mode.replace("/", "_").replace("\\", "_")
    return artifacts_dir / f"partition_{safe_mode}_k{num_clients}_a{alpha}_seed{partition_seed}.json"


def save_partition(path: Path, client_indices: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"client_indices": client_indices}, separators=(",", ":")),
        encoding="utf-8",
    )


def load_partition(path: Path) -> List[List[int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "client_indices" not in obj:
        raise ValueError("Invalid partition cache format")
    ci = obj["client_indices"]
    if not isinstance(ci, list):
        raise ValueError("client_indices must be a list")
    return [list(map(int, row)) for row in ci]
