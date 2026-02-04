# 📁 Project Structure

Complete overview of the GPU Cluster Acceptance Testing Tool repository.

## Directory Tree

```
gpu_cluster_testing/
├── 📄 Dockerfile                          # Container build (AMD64, PyTorch base)
├── 📘 README.md                           # Main documentation (606 lines)
├── 📄 LICENSE                             # MIT License
│
├── 📂 .github/workflows/                  # CI/CD
│   └── ci.yml                             # Build, test, push to ghcr.io
│
├── 📂 docs/                               # Documentation (8 files)
│   ├── 📘 README.md                      # Documentation index
│   ├── 🎓 HOW_IT_WORKS.md                # Architecture and data flow
│   ├── 🔧 TROUBLESHOOTING.md             # Common issues (UCX/UCC, NCCL, OOM)
│   ├── 📁 PROJECT_STRUCTURE.md           # This file
│   ├── 🌐 INFINIBAND_CONFIGURATION.md    # NCCL/IB setup
│   ├── 📊 NCCL_TESTING.md                # Bandwidth testing
│   ├── 📈 TESTING_WORKFLOW.md            # Decision trees, test sequences
│   └── 🧹 CLEANUP_GUIDE.md               # Resource cleanup
│
├── 📂 examples/                           # Kubernetes deployment examples
│   ├── kubernetes-pod-single-gpu.yaml
│   ├── kubernetes-pod-multi-gpu-single-node.yaml
│   ├── kubernetes-statefulset-multi-node-ddp.yaml
│   ├── kubernetes-with-auto-cleanup.yaml
│   ├── kubernetes-mixed-cluster.yaml
│   ├── kubernetes-flexible-nebius-pattern.yaml
│   └── kubernetes-multi-gpu-nebius-optimized.yaml
│
├── 📂 scripts/                            # Runtime scripts
│   └── 🔧 entrypoint.sh                  # Universal environment detection (241 lines)
│
├── 📂 src/                                # Source code
│   ├── 🎓 train.py                       # Main training orchestrator (506 lines)
│   ├── 🔢 data_utils.py                  # Synthetic data generation (147 lines)
│   ├── 📦 dataset_loaders.py             # Real datasets (CIFAR, FashionMNIST) (420 lines)
│   └── 📂 models/                         # Model implementations
│       ├── __init__.py                    # Model exports
│       ├── resnet18.py                    # ResNet-18 (146 lines)
│       ├── resnet.py                      # ResNet-50 (235 lines)
│       └── transformer.py                 # Transformer LM (270 lines)
│
└── 📂 tests/                              # Unit tests
    ├── test_models.py                     # Model architecture tests
    ├── test_data_utils.py                 # Data generation tests
    └── test_dataset_loaders.py            # Dataset loader tests
```

## File Inventory

### 🔴 Core Infrastructure

| File | Lines | Purpose |
|------|-------|---------|
| **Dockerfile** | 65 | AMD64 container with NVIDIA PyTorch 24.07, CUDA 12.5 |
| **README.md** | 606 | Main documentation, quick start, usage examples |
| **.github/workflows/ci.yml** | 116 | CI/CD: validate → test → build → push to ghcr.io |
| **scripts/entrypoint.sh** | 241 | Auto-detects Kubernetes/Docker environment, sets up NCCL |

**Key Features**:
- UCX/UCC library path fix (lines 15-19 in entrypoint.sh)
- Python command auto-detection (python vs python3)
- InfiniBand/RDMA detection and NCCL configuration
- Dynamic GPU count detection

### 🟢 Training Code

| File | Lines | Purpose |
|------|-------|---------|
| **src/train.py** | 506 | DDP orchestrator, supports ResNet18/50 + Transformer |
| **src/models/resnet18.py** | 146 | ResNet-18 (11M params) - Nebius production pattern |
| **src/models/resnet.py** | 235 | ResNet-50 (25M params) - comprehensive testing |
| **src/models/transformer.py** | 270 | Transformer LM - bandwidth testing |
| **src/data_utils.py** | 147 | Synthetic data (torch.randn, zero dependencies) |
| **src/dataset_loaders.py** | 420 | Real datasets with DistributedSampler |

**Models**:
- ResNet-18: 11.7M parameters, 44.6 MB
- ResNet-50: 25.6M parameters, 97.5 MB  
- Transformer: Configurable (1024 d_model, 16 heads, 12 layers)

**Data Modes**:
- `synthetic` - Default, no I/O (torch.randn in VRAM)
- `fashion_mnist` - 30MB, 28x28 grayscale (matches Nebius KubeRay)
- `cifar10` / `cifar100` - 32x32 RGB
- `imagenet` - 224x224 RGB subset

### 📚 Documentation

| File | Lines | Focus | Audience |
|------|-------|-------|----------|
| **HOW_IT_WORKS.md** | 500 | Architecture, data flow, execution | Developers |
| **TROUBLESHOOTING.md** | 218 | UCX/UCC, NCCL, OOM, platform issues | All users |
| **INFINIBAND_CONFIGURATION.md** | 532 | NCCL/IB setup, multi-node | Infra engineers |
| **NCCL_TESTING.md** | 320 | Bandwidth/latency testing | Infra engineers |
| **CLEANUP_GUIDE.md** | 448 | Resource management | Ops teams |
| **PROJECT_STRUCTURE.md** | ~200 | This file | All users |

**Total Documentation**: ~2,400 lines across 6 files

### 🧪 Tests

| Directory | Purpose |
|-----------|---------|
| **tests/** | Unit tests for models, data utils, dataset loaders |

Run with: `python -m pytest tests/ -v`

### 📦 Deployment Examples

7 example files covering:
- Kubernetes (PyTorchJob, plain Pods, StatefulSets)
- Docker (local testing)
- Flexible GPU configuration (Nebius pattern)

## Key Technologies

**Base Image**: `nvcr.io/nvidia/pytorch:24.07-py3`
- CUDA: 12.5.1
- Python: 3.10
- PyTorch: 2.4.0
- NCCL: 2.22.3
- Platform: linux/amd64 (explicit for GPU servers)

**Container Registry**: `ghcr.io/ahmabboud/gpu_cluster_testing:latest` (public)

**Dependencies**:
- Zero runtime dependencies (synthetic data mode)
- Optional: torchvision, datasets (for real data)

## CI/CD Pipeline

**Workflow** (.github/workflows/ci.yml):
1. **Validate**: Python syntax + bash syntax
2. **Test**: Run pytest unit tests (CPU)
3. **Build**: Docker build for AMD64
4. **Push**: ghcr.io on main branch only
5. **Verify**: Pull and inspect pushed image

**Triggers**:
- Push to main/develop
- Pull requests to main

## Code Statistics

```
Language     Files    Lines    Purpose
─────────────────────────────────────────
Python          7     1,733    Training code
Bash            1       241    Entrypoint
Markdown        6     2,409    Documentation
YAML            7       ~600   Examples + CI
Dockerfile      1        65    Container
─────────────────────────────────────────
Total                 5,048    lines
```

## Recent Updates (Feb 2026)

### Fixed
- ✅ UCX/UCC library path conflicts
- ✅ Python command detection (python vs python3)
- ✅ ResNet18 synthetic data shape bug
- ✅ Platform architecture (ARM64 → AMD64 cross-compile)

### Added
- ✅ TROUBLESHOOTING.md with common issues
- ✅ Unit tests in CI/CD
- ✅ ResNet18 model (Nebius pattern)
- ✅ InfiniBand auto-detection

### Removed
- ✅ 11 outdated documentation files
- ✅ Nebius registry references (migrated to ghcr.io)
- ✅ CPU fallback (GPU required, fail fast)

## Usage Quick Reference

**Deploy single GPU test**:
```bash
kubectl apply -f examples/kubernetes-pod-single-gpu.yaml
kubectl logs -f pod/gpu-cluster-test-single-gpu
```

**Deploy multi-GPU test**:
```bash
kubectl apply -f examples/kubernetes-pod-multi-gpu-single-node.yaml
kubectl logs -f pod/gpu-cluster-test-multi-gpu-single-node
```

**Check cluster health**:
```bash
kubectl run test --image=ghcr.io/ahmabboud/gpu_cluster_testing:latest \
  --restart=Never --rm -it -- bash
```

For detailed usage, see [README.md](../README.md).
