# FL Blackbox Watermark - Training and Drift Analysis
## 中文说明

### 功能概述

本仓库包含两部分核心能力：

- 模型训练（FedAvg）：`scripts/train_fl_fedavg.py`
- 数据下载（Tiny ImageNet）：`scripts/download_tiny_imagenet.py`
- 漂移分析：`scripts/analyze_client_upload_drift.py`

---

### 联邦训练（FedAvg）

#### 训练任务

- 模型：默认 `vgg16_bn`
- 数据：`tiny_imagenet`（也支持 `imagenet_full`）
- 联邦策略：FedAvg，Dirichlet 非 IID 划分
- 评估：可按轮计算 Top-1 / Top-5

#### 主要配置文件

- 常规实验：`configs/fl_imagenet_vgg16.yaml`

#### 训练命令示例

先下载 Tiny ImageNet（可选）：

```bash
python3 scripts/download_tiny_imagenet.py
```

运行完整配置：

```bash
python3 scripts/train_fl_fedavg.py --config configs/fl_imagenet_vgg16.yaml --device cuda
```

#### 训练阶段主要产物

- 分区缓存：`artifacts/partition_*.json`
- 训练指标：`artifacts/metrics/training_curves.csv`、`artifacts/metrics/training_curves.jsonl`
- 客户端上传快照：`artifacts/client_uploads/client_XXX/round_RRRR/upload_UU/{state_dict.pt,meta.json}`
- （可选）检查点：`artifacts/checkpoints/`

#### 常用训练配置项

- `federation.num_clients` / `federation.clients_per_round` / `federation.global_rounds`
- `federation.dirichlet_alpha`（越小越非 IID）
- `train.local_epochs` / `train.batch_size` / `train.lr`
- `eval.enabled` / `eval.interval_rounds`
- `client_uploads.enabled`（开启后才会生成漂移分析输入）
- `metrics.enabled`

---

### 漂移分析（client uploads）

### 输入目录结构

脚本默认读取如下目录结构（可通过参数传入自定义路径）：

```text
<client_uploads>/
  client_XXX/
    round_RRRR/
      upload_UU/
        state_dict.pt
        meta.json   # 可选，用于读取 avg_local_loss
```

### 主要输出文件

默认输出在 `client_uploads` 的父目录下：

- `client_upload_drift_report.csv`  
  客户端级汇总：`mean_l2_step`、`max_l2_step`、`total_l2_path`、`l2_first_to_last`、`drift_label`
- `client_upload_drift_per_round.csv`  
  逐步记录（每个客户端每个相邻轮一步）
- `client_upload_drift_matrices/client_XXXX_param_drift.csv`  
  参数级宽表（每行一个参数键）
- `client_upload_neuron_matrices/client_XXXX_neuron_drift.csv`  
  神经元级绝对漂移宽表
- `client_upload_neuron_rel_matrices/client_XXXX_neuron_rel_drift.csv`  
  神经元级相对漂移宽表（归一化后）

### 神经元相对漂移定义

新增的归一化表使用“相对变化”：

- 标量：`abs(delta) / (abs(prev) + eps)`
- 1 维向量分量：`abs(delta_i) / (abs(prev_i) + eps)`
- 高维张量（按第 0 维分组后每组）：`||delta_group||2 / (||prev_group||2 + eps)`

其中 `eps` 由 `--relative_eps` 控制，默认 `1e-12`。

### 快速开始

```bash
python3 scripts/analyze_client_upload_drift.py
```

指定输入目录：

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/artifacts/client_uploads
```

关闭参数级矩阵，仅输出神经元相关结果：

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/client_uploads --no_param_matrices
```

自定义神经元相对矩阵目录和 eps：

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/client_uploads \
  --neuron_rel_matrix_dir /tmp/neuron_rel \
  --relative_eps 1e-10
```

### 关键参数

- `client_uploads_dir`：输入目录（默认 `artifacts/client_uploads`）
- `--out_csv`：汇总 CSV 路径
- `--out_per_round_csv`：逐步 CSV 路径
- `--matrix_dir`：参数级矩阵目录
- `--no_param_matrices`：不输出参数级矩阵
- `--neuron_matrix_dir`：神经元绝对矩阵目录
- `--neuron_rel_matrix_dir`：神经元相对矩阵目录
- `--no_neuron_matrices`：不输出神经元矩阵（绝对和相对都关闭）
- `--relative_eps`：相对归一化分母的稳定项
- `--stagnant_frac`：低漂移客户端比例阈值（用于打标）

### 内存设计说明

脚本按客户端流式处理，相邻两轮比较后立即落盘。宽表在末尾用 `.npy + mmap` 组装写出，避免保留完整时间链的 `state_dict`。

---

## English

### Overview

This repository has two core parts:

- FL training (FedAvg): `scripts/train_fl_fedavg.py`
- Tiny ImageNet downloader: `scripts/download_tiny_imagenet.py`
- Drift analysis: `scripts/analyze_client_upload_drift.py`

---

### Federated Training (FedAvg)

#### Training Setup

- Model: `vgg16_bn` by default
- Data: `tiny_imagenet` (also supports `imagenet_full`)
- Federation: FedAvg with Dirichlet non-IID partitioning
- Evaluation: optional Top-1 / Top-5 evaluation per round

#### Main Config Files

- Regular experiment: `configs/fl_imagenet_vgg16.yaml`
- Smoke test: `configs/fl_smoke_tiny.yaml`

#### Training Commands

Download Tiny ImageNet (optional):

```bash
python3 scripts/download_tiny_imagenet.py
```

Run smoke test (1 round):

```bash
python3 scripts/train_fl_fedavg.py --config configs/fl_smoke_tiny.yaml --device auto
```

Run the full config:

```bash
python3 scripts/train_fl_fedavg.py --config configs/fl_imagenet_vgg16.yaml --device cuda
```

#### Training Artifacts

- Partition cache: `artifacts/partition_*.json`
- Metrics: `artifacts/metrics/training_curves.csv`, `artifacts/metrics/training_curves.jsonl`
- Client uploads: `artifacts/client_uploads/client_XXX/round_RRRR/upload_UU/{state_dict.pt,meta.json}`
- (Optional) checkpoints: `artifacts/checkpoints/`

#### Key Training Configs

- `federation.num_clients`, `federation.clients_per_round`, `federation.global_rounds`
- `federation.dirichlet_alpha` (smaller means more heterogeneous)
- `train.local_epochs`, `train.batch_size`, `train.lr`
- `train.max_batches_per_client` (useful for smoke runs)
- `eval.enabled`, `eval.interval_rounds`
- `client_uploads.enabled` (must be enabled to generate drift inputs)
- `metrics.enabled`

---

### Drift Analysis (client uploads)

### Expected Input Layout

```text
<client_uploads>/
  client_XXX/
    round_RRRR/
      upload_UU/
        state_dict.pt
        meta.json   # optional, reads avg_local_loss when present
```

### Outputs

By default, outputs are written to the parent directory of `client_uploads`:

- `client_upload_drift_report.csv`  
  client-level summary (`mean_l2_step`, `max_l2_step`, `total_l2_path`, `l2_first_to_last`, `drift_label`)
- `client_upload_drift_per_round.csv`  
  per-step records for each adjacent round pair
- `client_upload_drift_matrices/client_XXXX_param_drift.csv`  
  parameter-level wide matrix (one row per `state_dict` key)
- `client_upload_neuron_matrices/client_XXXX_neuron_drift.csv`  
  neuron-level absolute drift matrix
- `client_upload_neuron_rel_matrices/client_XXXX_neuron_rel_drift.csv`  
  neuron-level normalized (relative) drift matrix

### Relative Neuron Drift Definition

The normalized neuron matrix uses relative change:

- Scalar: `abs(delta) / (abs(prev) + eps)`
- 1D vector entry: `abs(delta_i) / (abs(prev_i) + eps)`
- Higher-dimensional tensor group (grouped by dim-0): `||delta_group||2 / (||prev_group||2 + eps)`

`eps` is controlled by `--relative_eps` (default: `1e-12`).

### Quick Start

```bash
python3 scripts/analyze_client_upload_drift.py
```

With a custom uploads directory:

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/artifacts/client_uploads
```

Disable parameter matrices:

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/client_uploads --no_param_matrices
```

Custom relative-neuron output directory and epsilon:

```bash
python3 scripts/analyze_client_upload_drift.py /path/to/client_uploads \
  --neuron_rel_matrix_dir /tmp/neuron_rel \
  --relative_eps 1e-10
```

### Key Arguments

- `client_uploads_dir`: input root (default: `artifacts/client_uploads`)
- `--out_csv`: summary CSV path
- `--out_per_round_csv`: per-step CSV path
- `--matrix_dir`: parameter matrix directory
- `--no_param_matrices`: skip parameter matrices
- `--neuron_matrix_dir`: absolute neuron matrix directory
- `--neuron_rel_matrix_dir`: relative neuron matrix directory
- `--no_neuron_matrices`: skip neuron matrices (both absolute and relative)
- `--relative_eps`: epsilon for relative normalization
- `--stagnant_frac`: fraction used to label low-drift clients

### Memory Behavior

The script processes each client in a streaming manner, writes step results immediately, and assembles wide matrices from temporary `.npy` columns via `mmap`, avoiding full timeline `state_dict` accumulation in RAM.
