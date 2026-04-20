# FL Blackbox Watermark - Drift Analysis (双语说明 / Bilingual README)

## 中文说明

### 功能概述

本仓库当前提供一个用于分析联邦学习上传模型漂移的脚本：

- 脚本：`scripts/analyze_client_upload_drift.py`
- 目标：统计每个客户端在相邻轮次上传参数之间的变化，并输出多种 CSV 报表
- 特点：流式处理，避免一次性将整条时间链全部加载进内存

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

This repository currently provides a drift analysis script for federated client uploads:

- Script: `scripts/analyze_client_upload_drift.py`
- Goal: measure per-client model drift between adjacent rounds and export CSV reports
- Key feature: streaming implementation to reduce peak memory usage

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
