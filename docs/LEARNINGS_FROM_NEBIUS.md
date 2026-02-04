# 🎓 Learnings from Nebius Working Tests

## Overview

Analysis of two proven Nebius tests:
1. **NCCL Test** (`nccl-test/`) - MPIJob-based pure NCCL bandwidth testing
2. **Ray Train Test** (`kuberay-tests/`) - ResNet18 training on FashionMNIST

Both are production-ready and verified to work in Nebius environments.

---

## Key Findings

### 1. **Resource Configurations** ⭐

#### Nebius NCCL Test (MPIJob Worker):
```yaml
resources:
  limits:
    cpu: 108           # Full node CPU allocation
    memory: 1600G      # Generous memory (1.6TB)
    nvidia.com/gpu: 8  # All GPUs
  requests:
    cpu: 108
    memory: 1200G      # Request 1.2TB, limit 1.6TB
    nvidia.com/gpu: 8
```

#### Our Current Test:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Only 1 GPU per worker
    memory: 16Gi       # Much smaller
```

**Impact**: We're under-utilizing resources. For multi-GPU nodes, we should:
- Request all GPUs on a node (typically 8)
- Allocate sufficient CPU cores (14-16 per GPU = 108-128 total)
- Request more memory (150-200GB per GPU)

---

### 2. **Shared Memory Configuration** ⭐⭐⭐

#### Nebius Uses:
```yaml
volumeMounts:
  - mountPath: /dev/shm
    name: dshm

volumes:
  - emptyDir:
      medium: Memory  # ← RAM-backed shared memory
    name: dshm
```

**Why Critical**:
- PyTorch DataLoader workers use `/dev/shm` for inter-process communication
- Default `/dev/shm` is only 64MB
- Multi-worker DataLoaders will crash with "OSError: [Errno 28] No space left on device"
- NCCL may also use shared memory for intra-node communication

**Our Gap**: We don't mount `/dev/shm` in our examples!

---

### 3. **Init Containers for ulimit** ⭐

#### Nebius Pattern:
```yaml
initContainers:
  - command:
      - sh
      - -c
      - ulimit -Hl unlimited && ulimit -Sl unlimited
    image: busybox:1.27.2
    name: init-limit
    securityContext:
      privileged: true
```

**Purpose**:
- Increase file descriptor limits (NCCL opens many network connections)
- Increase locked memory limits (important for GPU pinned memory)
- Prevents "too many open files" errors in distributed training

**Our Gap**: We rely on cluster defaults.

---

### 4. **Security Context: Privileged Mode** ⚠️

#### Nebius Uses:
```yaml
securityContext:
  privileged: true
```

**Why They Need It**:
- Direct access to InfiniBand devices (`/dev/infiniband/`)
- RDMA requires privileged access
- ulimit changes require privileges

**Our Approach**: We avoid privileged mode for portability, but document it as optional for InfiniBand clusters.

---

### 5. **NCCL Environment Variables** ⭐⭐

#### Nebius MPIJob Args:
```bash
-x NCCL_DEBUG=INFO                    # Verbose logging
-x NCCL_SOCKET_IFNAME=eth0            # Specific network interface
-x NCCL_IB_HCA=mlx5                   # InfiniBand adapter
-x UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
-x SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
-x NCCL_COLLNET_ENABLE=0              # Disable SHARP (InfiniBand collective offload)
```

**Key Insights**:
- Explicitly sets network interface (not auto-detected)
- Lists all 8 InfiniBand adapters (one per GPU)
- Uses UCX (Unified Communication X) for advanced RDMA
- Disables SHARP (may not be available in all clusters)

**Our Approach**: We auto-detect, but should provide these as configuration options.

---

### 6. **Test Size Configurations** ⭐

#### Nebius NCCL Test:
```bash
/opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
```
- Start: 512MB
- End: 8GB
- Factor: 2 (doubles each time)
- GPUs per process: 1

#### Ray All-Reduce Test:
```python
TENSOR_SIZES_BYTES = [
    512 * MB,
    1 * GB,
    2 * GB,
    4 * GB,
    8 * GB,
]
NUM_TRIALS = 5  # Multiple runs for averaging
```

**Our Current Approach**:
- We do full training runs (100 iterations)
- We have NCCL tests but could align sizes better

**Recommendation**: Add these specific sizes to our NCCL test suite for Nebius compatibility.

---

### 7. **Data Loading Strategy** ⭐⭐

#### Ray Test Uses FashionMNIST:
```python
transform = Compose([ToTensor(), Normalize((0.28604,), (0.32025,))])
data_dir = os.path.join(tempfile.gettempdir(), "data")
train_data = FashionMNIST(
    root=data_dir, train=True, download=True, transform=transform
)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
```

**Key Points**:
- Uses real dataset (60K images, 30MB) - downloads on first run
- Uses ResNet18 (11M params) - smaller than our ResNet50 (25M)
- Stores in `/tmp` - no persistent storage needed
- Batch size 128 per GPU

**Why This Works**:
- Small enough to download quickly
- Real data = realistic I/O patterns
- Still tests GPU compute and NCCL

**Our Approach**:
- We use synthetic data (fastest, zero I/O)
- We support CIFAR-10 (170MB) and ImageNet
- ResNet50 (25M params) - larger model

**Trade-off**: Synthetic = pure infrastructure test, Real = closer to production

---

### 8. **PyTorchJob vs MPIJob** ⭐

#### Nebius Uses:
- **MPIJob** for NCCL tests (MPI-based orchestration)
- **Ray** for ML training (Ray-specific orchestration)

#### We Use:
- **PyTorchJob** for ML training (PyTorch-native)
- **Direct NCCL tests** (no MPI dependency)

**Why PyTorchJob is Better for Us**:
- Platform-agnostic (works without MPI or Ray)
- Native PyTorch integration
- Simpler for users to adopt
- Works with PyTorch DDP directly

**When to Use MPIJob**:
- Pure NCCL testing (no PyTorch)
- Legacy MPI codebases
- Specific MPI tuning needs

---

### 9. **Kubernetes Operator Dependencies**

#### Nebius Installs:
```yaml
kubectl apply -k 'github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0'
```

**Our Documentation**: We tell users to install training-operator, but don't automate it.

**Recommendation**: Provide optional Terraform/script to install training-operator.

---

### 10. **Model Choice: ResNet18 vs ResNet50**

#### Ray Uses ResNet18:
```python
model = resnet18(num_classes=10)
model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)  # Adapt for FashionMNIST (1 channel, not 3)
```
- 11M parameters
- Modified first layer for grayscale input

#### We Use ResNet50:
- 25M parameters
- Tests larger gradients (better for bandwidth testing)

**Recommendation**: Keep ResNet50 as default, but add ResNet18 option for faster testing.

---

## Improvements to Implement

### Priority 1: Critical ⭐⭐⭐

1. **Add Shared Memory Mount**
   - Required for multi-worker DataLoaders
   - Prevents crashes in production

2. **Update Resource Requests**
   - Multi-GPU examples should request all GPUs on node
   - Align CPU/memory with GPU count

3. **Add Init Container for ulimit**
   - Prevents file descriptor exhaustion
   - Industry best practice

### Priority 2: Important ⭐⭐

4. **Add NCCL Configuration Examples**
   - Document InfiniBand-specific settings
   - Provide UCX configuration guide

5. **Align NCCL Test Sizes**
   - Use 512MB → 8GB range
   - Match Nebius benchmarks

6. **Add ResNet18 Option**
   - Faster for quick validation
   - Match Ray test patterns

### Priority 3: Nice-to-Have ⭐

7. **Add FashionMNIST Support**
   - Small dataset (30MB)
   - Real data without huge downloads

8. **Document MPIJob Alternative**
   - For users who need MPI compatibility

9. **Automate Training Operator Installation**
   - Optional convenience script

---

## What We're Already Doing Right ✅

1. **Platform Agnostic**: PyTorchJob works anywhere, not just Ray/MPI
2. **Synthetic Data**: Fastest for pure infrastructure testing
3. **Auto-Detection**: We detect environment automatically
4. **No Privileged Mode Required**: More secure by default
5. **Comprehensive Docs**: We explain everything clearly
6. **Multiple Models**: ResNet50 + Transformer for different test patterns
7. **Real Dataset Options**: CIFAR-10, ImageNet already supported

---

## Compatibility Matrix

| Feature | Nebius NCCL | Nebius Ray | Our Tool | Should Add? |
|---------|-------------|------------|----------|-------------|
| Multi-GPU per worker | ✅ 8 GPUs | ✅ 1 GPU | ❌ 1 GPU | ✅ Yes |
| Shared memory mount | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Init container (ulimit) | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Privileged mode | ✅ Yes | ❌ No | ❌ No | ⚠️ Document |
| NCCL env vars | ✅ Explicit | ❌ Auto | ✅ Auto | ⚠️ Add options |
| Test sizes (512MB-8GB) | ✅ Yes | N/A | ⚠️ Different | ✅ Align |
| ResNet18 | ❌ No | ✅ Yes | ❌ No | ⚠️ Add option |
| Real dataset | ❌ No | ✅ FashionMNIST | ✅ CIFAR/ImageNet | ✅ Add FashionMNIST |
| Synthetic data | N/A | ❌ No | ✅ Yes | ✅ Keep |
| Platform agnostic | ❌ MPIJob only | ❌ Ray only | ✅ PyTorchJob | ✅ Keep |

---

## Implementation Plan

### Phase 1: Critical Fixes (Immediate)

```yaml
# 1. Update kubernetes examples with:
volumeMounts:
  - mountPath: /dev/shm
    name: dshm
volumes:
  - emptyDir:
      medium: Memory
    name: dshm

initContainers:
  - name: init-ulimit
    image: busybox:1.27.2
    command: ['sh', '-c', 'ulimit -Hl unlimited && ulimit -Sl unlimited']
    securityContext:
      privileged: true

# 2. Update resource requests for multi-GPU:
resources:
  limits:
    nvidia.com/gpu: 8        # Full node
    memory: 1200Gi           # 150GB per GPU
    cpu: 108                 # 13-14 per GPU
  requests:
    nvidia.com/gpu: 8
    memory: 1000Gi
    cpu: 100
```

### Phase 2: NCCL Alignment (Next)

```python
# 3. Update NCCL test sizes in scripts/run-nccl-tests.sh:
TENSOR_SIZES="512M 1G 2G 4G 8G"  # Match Nebius

# 4. Add NCCL configuration guide with:
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand
NCCL_IB_HCA=mlx5
UCX_NET_DEVICES=mlx5_0:1,...,mlx5_7:1  # For 8-GPU nodes
```

### Phase 3: Dataset Options (Enhancement)

```python
# 5. Add FashionMNIST to dataset_loaders.py:
def load_fashion_mnist(root_dir, batch_size, rank, world_size):
    """Small dataset (30MB) for quick testing"""
    from torchvision.datasets import FashionMNIST
    # Implementation...

# 6. Add ResNet18 to models/:
# Smaller, faster for quick validation
```

---

## Summary

**Main Gaps to Address**:
1. ❌ No shared memory mount → crashes with multi-worker DataLoader
2. ❌ Under-utilization of multi-GPU nodes → poor performance
3. ❌ No ulimit init container → may hit file descriptor limits

**Our Strengths**:
1. ✅ Platform-agnostic (works beyond Nebius)
2. ✅ Synthetic data = fastest testing
3. ✅ Better documentation

**Action Items**:
- Update all Kubernetes examples with shm, init container, proper resource requests
- Add NCCL configuration guide for InfiniBand
- Document multi-GPU worker patterns
- Add ResNet18 and FashionMNIST as options

**Philosophy**:
- Keep our tool **portable** (works on any Kubernetes, not just Nebius)
- Make it **Nebius-optimized** via configuration examples
- Provide **both synthetic and real data** options
- Maintain **zero external dependencies** as default

---

## Next Steps

1. ✅ Document findings (this file)
2. ⬜ Update Kubernetes YAML examples
3. ⬜ Add multi-GPU worker examples
4. ⬜ Create InfiniBand configuration guide
5. ⬜ Add ResNet18 option
6. ⬜ Test on Nebius cluster
