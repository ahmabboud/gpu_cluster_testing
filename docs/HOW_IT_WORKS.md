# 🎓 Understanding the GPU Cluster Testing Tool

## Quick Answer: No Database!

**This tool does NOT use a database.** It generates synthetic data directly in GPU memory using `torch.randn()`. This is intentional - it's a **cluster validation tool**, not a real ML system.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Container Image (12 GB)                        │
│  ghcr.io/ahmabboud/gpu_cluster_testing:latest             │
│                                                             │
│  Base: NVIDIA PyTorch 24.07                                │
│  ├─ CUDA 12.5 + cuDNN                                      │
│  ├─ Python 3.10 + PyTorch 2.4.0                            │
│  └─ NCCL (GPU communication library)                       │
│                                                             │
│  /workspace/                                                │
│  ├─ src/                   (Training code)                 │
│  │  ├─ train.py           (Main orchestrator)              │
│  │  ├─ models/            (ResNet, Transformer)            │
│  │  └─ data_utils.py      (Synthetic data generator)       │
│  │                                                          │
│  ├─ scripts/               (Automation)                    │
│  │  └─ entrypoint.sh      (Environment detector)           │
│  │                                                          │
│  └─ nccl-tests/            (Bandwidth tests)               │
│     └─ build/all_reduce_perf                               │
└─────────────────────────────────────────────────────────────┘
```

---

## How Data Works: NO DATABASE!

### Traditional ML Setup (Not Used Here)

```
Storage/Database          This Tool
────────────────          ──────────
┌──────────────┐          ┌──────────────┐
│   ImageNet   │          │  GPU Memory  │
│   Database   │    VS    │  (VRAM)      │
│   (1 TB)     │          │              │
│              │          │ torch.randn()│
│ Load → CPU   │          │ torch.randint│
│  ↓           │          │              │
│ Preprocess   │          │ Instant!     │
│  ↓           │          │ No I/O!      │
│ GPU          │          └──────────────┘
└──────────────┘
   Slow, Complex         Fast, Simple
```

### Synthetic Data Generation

```python
# src/data_utils.py - The "database" is this simple function!

def generate_synthetic_batch(batch_size, num_channels, height, width, 
                            num_classes, device):
    """Generate random data DIRECTLY in GPU memory"""
    
    # Create random images (RGB, 224x224)
    images = torch.randn(
        batch_size, num_channels, height, width,
        device=device,  # ← Created directly on GPU!
        dtype=torch.float32
    )
    
    # Create random labels (0-999 for ImageNet-style)
    labels = torch.randint(
        0, num_classes,
        (batch_size,),
        device=device,  # ← Created directly on GPU!
        dtype=torch.long
    )
    
    return images, labels

# That's it! No database, no files, no network I/O
```

**Why This Works**:
- Neural networks don't know if data is "real" or random
- GPU compute is identical
- NCCL communication is identical
- Network bandwidth testing works the same
- We're testing **infrastructure**, not model accuracy

---

## Complete Execution Flow

### Step-by-Step: What Happens When You Run a Test

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Start Container                                    │
│                                                             │
│ $ docker run --gpus all --rm \                             │
│     cr.eu-north1.../gpu_cluster_testing:latest \           │
│     --model resnet50 --batch-size 64                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: entrypoint.sh Executes                             │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Detect Environment:                                 │   │
│ │ - Slurm? Check SLURM_PROCID                         │   │
│ │ - Kubernetes? Check KUBERNETES_SERVICE_HOST         │   │
│ │ - Bare metal? Use manual env vars                   │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Map Variables:                                      │   │
│ │ SLURM_PROCID    → RANK                              │   │
│ │ SLURM_NTASKS    → WORLD_SIZE                        │   │
│ │ SLURM_LOCALID   → LOCAL_RANK                        │   │
│ │ SLURM_NODELIST  → MASTER_ADDR (first node)         │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ GPU Detection:                                      │   │
│ │ $ nvidia-smi                                        │   │
│ │ Found 8 × NVIDIA H100 PCIe                          │   │
│ │ → Set BACKEND=nccl                                  │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ Launch: python /workspace/src/train.py --model resnet50   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: train.py - Initialize Distributed Training         │
│                                                             │
│ def setup_distributed():                                   │
│     rank = int(os.environ["RANK"])         # 0            │
│     world_size = int(os.environ["WORLD_SIZE"])  # 8       │
│     local_rank = int(os.environ["LOCAL_RANK"])  # 0-7     │
│                                                             │
│     # Initialize process group (connects all GPUs)         │
│     dist.init_process_group(                               │
│         backend="nccl",  # GPU communication               │
│         init_method="env://",                              │
│         world_size=8,     # 8 GPUs total                   │
│         rank=0            # This process's rank            │
│     )                                                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Create and Wrap Model                              │
│                                                             │
│ # Create model (25M parameters)                            │
│ model = ResNet50(num_classes=1000)                         │
│                                                             │
│ # Move to GPU                                              │
│ model = model.to(device)  # device = cuda:0                │
│                                                             │
│ # Wrap with DistributedDataParallel (DDP)                  │
│ model = DDP(model, device_ids=[local_rank])                │
│ # ↑ This is MAGIC!                                         │
│ # DDP automatically syncs gradients via NCCL               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Training Loop (100 iterations)                     │
│                                                             │
│ for iteration in range(100):                               │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ 5a. Generate Data (NO DATABASE!)               │     │
│   │                                                  │     │
│   │ images, labels = generate_synthetic_batch(      │     │
│   │     batch_size=64,                              │     │
│   │     num_channels=3,   # RGB                     │     │
│   │     height=224,                                 │     │
│   │     width=224,                                  │     │
│   │     num_classes=1000,                           │     │
│   │     device='cuda:0'   # Directly in GPU memory!│     │
│   │ )                                                │     │
│   │                                                  │     │
│   │ Shape: images = [64, 3, 224, 224]              │     │
│   │        labels = [64]                            │     │
│   │ Memory: ~150 MB in GPU VRAM                     │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ 5b. Forward Pass (GPU Compute)                 │     │
│   │                                                  │     │
│   │ output = model(images)                          │     │
│   │ # Runs convolutions, batch norms, ReLU, etc.   │     │
│   │ # Tests GPU compute performance                │     │
│   │                                                  │     │
│   │ Shape: output = [64, 1000]  (predictions)      │     │
│   │ Time: ~30 ms                                    │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ 5c. Compute Loss                               │     │
│   │                                                  │     │
│   │ loss = criterion(output, labels)                │     │
│   │ # CrossEntropyLoss                              │     │
│   │                                                  │     │
│   │ Time: <1 ms                                     │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ 5d. Backward Pass (Gradients + NCCL!)         │     │
│   │                                                  │     │
│   │ loss.backward()                                 │     │
│   │                                                  │     │
│   │ What happens:                                   │     │
│   │ 1. Compute gradients (GPU compute)             │     │
│   │    Time: ~25 ms                                 │     │
│   │                                                  │     │
│   │ 2. DDP triggers NCCL All-Reduce! ← CRITICAL    │     │
│   │    - Each GPU has gradients for 25M params     │     │
│   │    - All-Reduce averages across all 8 GPUs     │     │
│   │    - Uses NVLink (single node) or              │     │
│   │      InfiniBand (multi-node)                   │     │
│   │    - Tests network bandwidth!                  │     │
│   │    Time: ~5-10 ms (depends on network)         │     │
│   │                                                  │     │
│   │ 3. All GPUs now have synchronized gradients    │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   ┌─────────────────────────────────────────────────┐     │
│   │ 5e. Update Weights                             │     │
│   │                                                  │     │
│   │ optimizer.step()                                │     │
│   │ # Update model parameters                       │     │
│   │                                                  │     │
│   │ Time: ~2 ms                                     │     │
│   └─────────────────────────────────────────────────┘     │
│                                                             │
│   Total iteration time: ~47 ms                             │
│   Throughput: 64 × 8 GPUs / 0.047s = ~10,800 samples/sec  │
│                                                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Report Results                                     │
│                                                             │
│ {                                                           │
│   "model": "resnet50",                                     │
│   "world_size": 8,                                         │
│   "batch_size_per_gpu": 64,                                │
│   "global_batch_size": 512,                                │
│   "avg_step_time_ms": 47.2,                                │
│   "throughput_samples_per_second": 14234,                  │
│   "nccl_overhead_ms": 4.8,                                 │
│   "gpu_utilization_pct": 93.5,                             │
│   "backend": "nccl"                                        │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## What Gets Tested

### 1. GPU Compute Performance ✅

**Tested during**:
- Forward pass: Convolutions, matrix multiplications
- Backward pass: Gradient computation

**Metrics**:
- Samples per second
- GPU utilization (target: >90%)
- Step time

**Why synthetic data works**: GPU doesn't care if data is random or real - math is the same!

### 2. NCCL Communication ✅ (Most Important!)

**Tested during**:
- Gradient synchronization (All-Reduce operation)
- Multi-GPU coordination

**What it validates**:
- **Single node**: NVLink bandwidth between GPUs (~400 GB/s for H100)
- **Multi-node**: InfiniBand/Ethernet between servers (~200 GB/s for HDR IB)
- **Latency**: Communication overhead (<10ms is good)
- **Stability**: No NCCL errors during 100+ iterations

**This is the PRIMARY purpose** - validate the network!

### 3. Cluster Stability ✅

**Tested during**:
- Sustained 100+ iteration run
- Continuous GPU load
- Continuous network traffic

**What it validates**:
- GPUs maintain clock speeds (no thermal throttling)
- Power delivery is stable
- Network doesn't have intermittent issues
- No OOM (out of memory) errors

---

## Multi-Node Example

### 4 Nodes × 8 GPUs = 32 GPUs Total

```
Node 0 (Master)           Node 1                Node 2                Node 3
┌──────────────┐         ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ RANK 0-7     │         │ RANK 8-15    │     │ RANK 16-23   │     │ RANK 24-31   │
│              │         │              │     │              │     │              │
│ GPU 0-7      │         │ GPU 0-7      │     │ GPU 0-7      │     │ GPU 0-7      │
│              │         │              │     │              │     │              │
│ NVLink ←→    │         │ NVLink ←→    │     │ NVLink ←→    │     │ NVLink ←→    │
│ (400 GB/s)   │         │ (400 GB/s)   │     │ (400 GB/s)   │     │ (400 GB/s)   │
└──────┬───────┘         └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                        │                     │                     │
       └────────────────────────┼─────────────────────┼─────────────────────┘
                                │    InfiniBand       │
                                │    (200 GB/s)       │
                                │                     │
                          ┌─────┴─────────────────────┴─────┐
                          │   All-Reduce across 32 GPUs     │
                          │   - Synchronize gradients        │
                          │   - Test cross-node network      │
                          └──────────────────────────────────┘
```

**What happens during All-Reduce**:
1. Each GPU computes gradients locally (25M parameters × 4 bytes = 100 MB)
2. NCCL performs All-Reduce:
   - Within node: Use NVLink (fast!)
   - Between nodes: Use InfiniBand (tests network!)
3. All 32 GPUs end up with averaged gradients
4. **This tests your InfiniBand fabric** - the whole point!

---

## The Image Explained

### What's Inside

```
ghcr.io/ahmabboud/gpu_cluster_testing:latest
│
├── Base: nvcr.io/nvidia/pytorch:24.07-py3
│   ├── Ubuntu 22.04
│   ├── CUDA 12.5 (GPU drivers, libraries)
│   ├── cuDNN (optimized neural net ops)
│   ├── Python 3.10
│   ├── PyTorch 2.4.0 (with NCCL support)
│   └── NCCL 2.20+ (GPU communication)
│
├── System Tools (+200 MB)
│   ├── Network diagnostics (ip, ping, netstat)
│   ├── MPI (for NCCL tests)
│   └── Build tools (gcc, make)
│
├── NCCL Test Binaries (+50 MB)
│   └── /workspace/nccl-tests/build/
│       ├── all_reduce_perf     (bandwidth test)
│       ├── all_reduce_perf_mpi (multi-node test)
│       └── ... (other tests)
│
└── Application Code (+5 MB)
    ├── /workspace/src/
    │   ├── train.py           (496 lines - main orchestrator)
    │   ├── models/
    │   │   ├── resnet.py      (235 lines - ResNet-50)
    │   │   └── transformer.py (270 lines - Transformer)
    │   ├── data_utils.py      (148 lines - synthetic data!)
    │   └── dataset_loaders.py (329 lines - optional real data)
    └── /workspace/scripts/
        └── entrypoint.sh      (191 lines - environment detector)

Total Size: ~12 GB
```

### Why This Base Image?

**nvcr.io/nvidia/pytorch:24.07-py3**:
- Official NVIDIA image (tested and optimized)
- CUDA + PyTorch pre-configured
- NCCL pre-installed and working
- Saves us from dependency hell!

### What We Added

1. **Network diagnostics**: So you can debug connectivity issues
2. **NCCL test binaries**: For focused bandwidth testing
3. **Our training code**: The actual test logic
4. **entrypoint.sh**: Auto-detects Slurm/K8s/bare metal

---

## Key Design Decisions

### Why Synthetic Data?

| Approach | Pros | Cons |
|----------|------|------|
| **Real Data** | Realistic, tests I/O | Requires storage, network, setup |
| **Synthetic** ✅ | Zero setup, instant, portable | Not "real" ML |

**For cluster validation, synthetic is better because**:
- We're testing infrastructure, not models
- GPU compute is identical
- NCCL communication is identical
- Removes variables (storage speed, network latency to storage)
- Pure GPU + interconnect testing

### Why Two Test Modes?

**Full Training Tests** (20 minutes):
```python
# Realistic ML workload
model = ResNet50()
for i in range(100):
    data = generate_synthetic_batch()
    loss = model(data)
    loss.backward()  # ← Tests NCCL
```
- Tests complete stack
- Realistic resource usage
- Good for acceptance testing

**NCCL Bandwidth Tests** (5 minutes):
```bash
# Direct NCCL test
./all_reduce_perf -b 8K -e 8G
```
- Isolated network testing
- Quick feedback
- Good for debugging

Both complement each other!

---

## Common Questions

### Q: No database means no real ML training?
**A**: Correct! This is a **validation tool**, not a training system. Think of it like a stress test for your cluster.

### Q: How do you know the results are valid?
**A**: We compare against known benchmarks (H100: ~14k samples/sec, ~400 GB/s NVLink). If your cluster matches, it's good!

### Q: What if I want to use real data?
**A**: You can! Use `--data-mode cifar10` or `--data-mode imagenet`. But for cluster validation, synthetic is recommended.

### Q: Does this replace real ML training?
**A**: No! This validates the cluster. Once validated, run your real ML workloads.

### Q: What about storage performance testing?
**A**: Out of scope. This tests GPU + network only. Test storage separately.

---

## Summary

**The "Database"**: 
```python
torch.randn()  # That's it!
```

**The Image**:
- NVIDIA PyTorch base + our training code + NCCL tests
- ~12 GB, self-contained, works anywhere

**What It Tests**:
1. GPU compute (forward/backward pass)
2. NCCL communication (All-Reduce) ← **Most important!**
3. Cluster stability (sustained load)

**How It Works**:
1. Detect environment (Slurm/K8s/bare metal)
2. Initialize distributed training (connect all GPUs)
3. Generate data in GPU memory (no database!)
4. Train model (tests compute + NCCL)
5. Report performance metrics

**Why It's Effective**:
- Zero dependencies = works anywhere
- Synthetic data = tests what matters
- Self-contained = easy to deploy
- Fast = quick validation

This tool answers one question: **"Is this GPU cluster ready for production ML workloads?"**

The answer comes from GPU utilization, NCCL bandwidth, and stability over 100+ iterations - not from model accuracy!
