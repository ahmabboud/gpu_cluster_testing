# GPU Cluster Acceptance Testing Tool

[![CI](https://github.com/ahmabboud/gpu_cluster_testing/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmabboud/gpu_cluster_testing/actions/workflows/ci.yml)

**Zero-Dependency Distributed Training for Infrastructure Validation**

A portable, scale-agnostic tool for validating GPU cluster health, performance, and interconnect stability. Run acceptance tests on GPU clusters regardless of size or orchestration layer.

**Production-Ready** | **Open Source** | **Battle-Tested**

## Test Types

This tool provides **three types of tests** for comprehensive GPU cluster validation:

| Test Type | What It Validates | Duration | Nodes Required |
|-----------|-------------------|----------|----------------|
| **🔥 Single GPU Training** | GPU compute, memory, PyTorch | 2-3 min | 1 GPU node |
| **📡 Multi-GPU NCCL** | NVLink, intra-node communication | 3-5 min | 1 GPU node (2+ GPUs) |
| **🌐 Multi-Node DDP** | InfiniBand, inter-node RDMA, distributed training | 5-10 min | 2 GPU nodes |

## Quick Start

### Run All Tests (Recommended)

```bash
# Run the complete test suite
./scripts/run-all-tests.sh

# With longer timeout for autoscaling clusters
./scripts/run-all-tests.sh --timeout 900

# Skip multi-node test (single GPU node only)
./scripts/run-all-tests.sh --skip-multinode
```

The script automatically:
- Cleans up previous test resources
- Runs all tests in sequence
- Reports pass/fail for each test
- Cleans up after completion

### Run Individual Tests

```bash
# Single GPU test
kubectl apply -f examples/kubernetes-pod-single-gpu.yaml
kubectl logs -f pod/gpu-cluster-test-single-gpu

# Multi-GPU test (single node)
kubectl apply -f examples/kubernetes-pod-multi-gpu-single-node.yaml
kubectl logs -f pod/gpu-cluster-test-multi-gpu-single-node

# Multi-Node DDP test (requires 2 GPU nodes)
kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml
kubectl logs -f pod/gpu-cluster-test-ddp-0
```

## Features

- **🚀 Zero Dependencies (Default)**: Uses synthetic data generation - no external datasets required
- **📦 Real Dataset Support**: Optional FashionMNIST, CIFAR-10, CIFAR-100, or ImageNet subset
- **🎯 Multiple Models**:
  - **ResNet18** - 11M parameters (faster validation)
  - **ResNet50** - 25M parameters (comprehensive testing)
  - **Transformer** - Configurable size (bandwidth testing)
- **📊 Comprehensive Testing**: Tests GPU compute, memory, and NCCL communication
- **🔧 Universal Compatibility**: Works on any Kubernetes cluster with GPU nodes
- **📈 Performance Profiling**: Detailed metrics including throughput, step time, and NCCL overhead
- **🔍 Diagnostic Rich**: Verbose NCCL logging for troubleshooting
- **⚡ Two Test Modes**:
  - **Full Training Tests**: Realistic workload with ResNet or Transformer models
  - **NCCL Bandwidth Tests**: Direct NCCL performance measurement (see [NCCL Testing Guide](docs/NCCL_TESTING.md))
- **🎯 Flexible Configuration**:
  - Configurable GPU count (1-8+ GPUs per worker)
  - Automatic resource scaling recommendations
  - No hardcoded assumptions
- **🏭 Production Patterns**: Battle-tested configurations
  - ResNet18 + FashionMNIST (lightweight validation)
  - Multi-GPU worker support
  - Shared memory configuration for DataLoader workers
  - Init containers for ulimit configuration
  - InfiniBand/RDMA optimization

## Prerequisites

### Kubernetes Clusters
- **kubectl access** with permissions to create pods/jobs
- **NVIDIA GPU Operator** or device plugin installed
- **GPU nodes labeled** (verify with `kubectl get nodes -L nvidia.com/gpu.product`)
- **Optional**: Kubeflow Training Operator (PyTorchJob CRD) if you want to use `PyTorchJob` examples (`kubectl get crd pytorchjobs.kubeflow.org`)

#### Quick Cluster Verification

```bash
# 1. Check GPU nodes exist
kubectl get nodes -L nvidia.com/gpu.product

# 2. Verify GPU resource allocation
kubectl describe nodes | grep -A 5 "Allocated resources"

# 3. Test GPU access with a simple pod (portable across kubectl versions)
cat <<'YAML' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi-test
spec:
  restartPolicy: Never
  containers:
  - name: cuda
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
YAML

kubectl logs -f pod/nvidia-smi-test
kubectl delete pod nvidia-smi-test --ignore-not-found=true
```

#### Quick Cluster Verification

These commands work from any machine with `kubectl` access to a GPU cluster. No local GPU required.

### 1. Single GPU Test

```bash
# Deploy single-GPU test pod
kubectl apply -f examples/kubernetes-pod-single-gpu.yaml

# Follow logs
kubectl logs -f pod/gpu-cluster-test-single-gpu

# Cleanup
kubectl delete pod gpu-cluster-test-single-gpu --ignore-not-found=true
```

**Expected output (~4,600 samples/sec on H100):**
```
✅ GPU detected: NVIDIA H100 80GB HBM3
✅ NCCL initialized successfully
Throughput: 4609.23 samples/sec
```

### 2. Multi-GPU Test (Single Node with NCCL)

```bash
# Deploy multi-GPU test pod (uses torchrun for DDP)
kubectl apply -f examples/kubernetes-pod-multi-gpu-single-node.yaml

# Follow logs
kubectl logs -f pod/gpu-cluster-test-multi-gpu-single-node

# Cleanup
kubectl delete pod gpu-cluster-test-multi-gpu-single-node --ignore-not-found=true
```

**Expected output (~15,000 samples/sec on 2× H100, scales linearly):**
```
✅ NCCL initialized: nranks=2
✅ Using InfiniBand: mlx5_0
Throughput: 15616.41 samples/sec
```

### 3. Multi-Node DDP Test (Optional)

```bash
# Deploy StatefulSet for multi-node distributed training
kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml

# Follow logs from rank 0
kubectl logs -f pod/gpu-cluster-test-ddp-0

# Cleanup
kubectl delete statefulset gpu-cluster-test-ddp service/gpu-cluster-test-ddp --ignore-not-found=true
```

## Additional Deployment Examples

### Kubernetes (with PyTorchJob CRD)

> **Note**: Requires Kubeflow Training Operator installed. Check with: `kubectl get crd pytorchjobs.kubeflow.org`

**Works on mixed GPU/non-GPU clusters**: The tool automatically schedules pods on GPU nodes using resource requests. No manual node selection needed.

Create a `pytorchjob.yaml`:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: gpu-cluster-acceptance-test
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: ghcr.io/ahmabboud/gpu_cluster_testing:latest
            args:
              - "--model"
              - "resnet50"
              - "--batch-size"
              - "64"
              - "--active-iterations"
              - "100"
            resources:
              limits:
                nvidia.com/gpu: 8  # Ensures scheduling on GPU nodes
              requests:
                nvidia.com/gpu: 8
          # Optional: Explicit node selection for GPU nodes
          nodeSelector:
            accelerator: nvidia-gpu  # Adjust label to match your cluster
          # Optional: Tolerations if GPU nodes are tainted
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: ghcr.io/ahmabboud/gpu_cluster_testing:latest
            args:
              - "--model"
              - "resnet50"
              - "--batch-size"
              - "64"
              - "--active-iterations"
              - "100"
            resources:
              limits:
                nvidia.com/gpu: 8
              requests:
                nvidia.com/gpu: 8
          # Optional: Explicit node selection for GPU nodes
          nodeSelector:
            accelerator: nvidia-gpu  # Adjust label to match your cluster
          # Optional: Tolerations if GPU nodes are tainted
          tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
```

Deploy:
```bash
kubectl apply -f pytorchjob.yaml
```

Monitor:
```bash
kubectl logs -f gpu-cluster-acceptance-test-master-0
```

**For mixed clusters, verify GPU node labels:**
```bash
# Quick verification script
./scripts/verify-k8s-gpu-cluster.sh

# Or manually check GPU node labels
kubectl get nodes --show-labels | grep gpu

# If using custom labels, adjust nodeSelector accordingly
kubectl label nodes <node-name> accelerator=nvidia-gpu
```

**Advanced Example**: For production deployments on mixed clusters with node affinity, tolerations, and anti-affinity rules, see [examples/kubernetes-mixed-cluster.yaml](examples/kubernetes-mixed-cluster.yaml).

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `resnet50` | Model type: `resnet18`, `resnet50`, or `transformer` |
| `--data-mode` | `synthetic` | Data source: `synthetic`, `cifar10`, `cifar100`, or `imagenet` |
| `--data-dir` | `./data` | Directory for dataset storage (if using real datasets) |
| `--batch-size` | `32` | Batch size per GPU |
| `--num-classes` | `1000` | Number of classes (vocab size for transformer) |
| `--learning-rate` | `0.01` | Learning rate |
| `--warmup-iterations` | `50` | Number of warmup iterations |
| `--active-iterations` | `100` | Number of measurement iterations |

### Data Modes

Data modes can be configured via command-line arguments in your Kubernetes manifests:

| Mode | Description | Use Case |
|------|-------------|----------|
| `synthetic` (default) | Generated data, no downloads | ✅ Recommended for acceptance testing |
| `cifar10` | 170MB, auto-download | Lightweight real data validation |
| `cifar100` | 170MB, auto-download | Lightweight real data validation |
| `imagenet` | Requires pre-mounted dataset | Production-scale testing |

**Example**: To use CIFAR-10, modify your pod spec args:
```yaml
args:
  - "--model"
  - "resnet50"
  - "--data-mode"
  - "cifar10"
  - "--num-classes"
  - "10"
```

> **Note:** For acceptance testing, **synthetic mode is recommended** as it eliminates dataset download time and storage requirements, providing consistent, reproducible results focused purely on GPU/network performance.

## Understanding Results

The tool outputs a `results.json` file with the following structure:

```json
{
  "model": "resnet50",
  "world_size": 32,
  "batch_size_per_gpu": 64,
  "global_batch_size": 2048,
  "total_time_seconds": 45.67,
  "average_step_time_seconds": 0.456,
  "throughput_samples_per_second": 4483.2,
  "total_samples": 204800,
  "device": "cuda:0",
  "backend": "nccl"
}
```

### Key Metrics

- **Throughput**: Total samples processed per second (higher is better)
- **Average Step Time**: Time per training step including gradient synchronization
- **Global Batch Size**: Total batch size across all GPUs

### Success Criteria

✅ **Test Passed** if:
- No NCCL errors or timeouts
- Throughput > 0 samples/sec
- All GPUs detected and utilized
- Training completes without OOM errors

⚠️ **Investigate** if:
- Throughput significantly below baseline (see docs/HOW_IT_WORKS.md)
- NCCL initialization failures (see docs/TROUBLESHOOTING.md)
- OOM errors (reduce batch size or use smaller model)

## Troubleshooting

**Common issues:**

1. **UCX/UCC ImportError** - Fixed in container, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#ucxucc-library-conflict)
2. **NCCL timeout** - Check network connectivity and InfiniBand config
3. **Pod stuck in Pending** - Verify GPU nodes and resource requests
4. **OOM errors** - Reduce `--batch-size` or use `resnet18` instead

**Full troubleshooting guide**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## Expected Performance (Baseline)

### Quick Reference (H100 80GB, NVLink)

| Configuration | Model | Batch/GPU | Expected Throughput |
|---------------|-------|-----------|---------------------|
| 1x H100 | ResNet-18 | 128 | ~1,800 samples/sec |
| 8x H100 | ResNet-50 | 64 | ~14,000 samples/sec |
| 32x H100 | ResNet-50 | 64 | ~50,000 samples/sec |
| 8x H100 | Transformer | 32 | ~8,000 samples/sec |

**Note**: Actual performance varies by GPU model, interconnect (NVLink vs InfiniBand vs Ethernet), and NCCL configuration. See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for detailed benchmarks.

## NCCL Bandwidth Testing

For direct NCCL bandwidth/latency measurements, see the **[NCCL Testing Guide](docs/NCCL_TESTING.md)**.

### Quick NCCL Test (Kubernetes)

```bash
# Deploy multi-node DDP test (includes NCCL communication)
kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml
kubectl logs -f pod/gpu-cluster-test-ddp-0

# Cleanup
kubectl delete statefulset gpu-cluster-test-ddp service/gpu-cluster-test-ddp --ignore-not-found=true
```

**Expected bandwidth:**
- **NVLink (intra-node)**: 400-450 GB/s on H100
- **InfiniBand (inter-node)**: 200-240 GB/s on HDR

**When to use NCCL tests vs Full Training:**
- **NCCL tests** (2-5 min): Quick infrastructure validation, network debugging
- **Full training** (10-30 min): Realistic acceptance testing, end-to-end validation

Both approaches are complementary. See [NCCL Testing Guide](docs/NCCL_TESTING.md) for detailed comparison.

## Environment Variables

### Detected Automatically

- **Kubernetes**: `KUBERNETES_SERVICE_HOST`, or set via Pod spec

### Manual Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `RANK` | Global rank of the process | `0` |
| `WORLD_SIZE` | Total number of processes | `1` |
| `LOCAL_RANK` | Local rank on the node | `0` |
| `MASTER_ADDR` | Address of the master node | `localhost` |
| `MASTER_PORT` | Port for distributed communication | `29500` |
| `BACKEND` | PyTorch distributed backend | `nccl` (GPU) / `gloo` (CPU) |
| `NCCL_DEBUG` | NCCL logging level | `INFO` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | Auto-detected |

## CI/CD Pipeline

The repository includes automated CI/CD via GitHub Actions (`.github/workflows/ci.yml`):

| Stage | Trigger | Actions |
|-------|---------|---------|
| **Validate** | All pushes/PRs | Python syntax check, unit tests |
| **Build & Push** | Push to `main` (if code changed) | Build Docker image, push to `ghcr.io` |
| **Smoke Test** | After build | Pull and verify container |

**Automatic image updates**: When you push to `main` and modify `Dockerfile`, `src/`, or `scripts/`, the CI automatically builds and pushes a new image to `ghcr.io/ahmabboud/gpu_cluster_testing:latest`.

## Building from Source

```bash
git clone https://github.com/YOUR_USERNAME/gpu_cluster_testing.git
cd gpu_cluster_testing

# Build container (for AMD64 GPU servers)
docker build --platform linux/amd64 -t ghcr.io/YOUR_USERNAME/gpu_cluster_testing:latest .

# Push to your registry
docker push ghcr.io/YOUR_USERNAME/gpu_cluster_testing:latest

# Update Kubernetes manifests to use your image
sed -i 's|ghcr.io/ahmabboud/|ghcr.io/YOUR_USERNAME/|g' examples/*.yaml
```

## Resource Cleanup

The test runner script (`./scripts/run-all-tests.sh`) automatically cleans up all test resources before and after running. For manual cleanup:

### Quick Cleanup Commands

**Kubernetes**:
```bash
# Delete all test pods
kubectl delete pod -l app=gpu-cluster-test --ignore-not-found=true

# Delete StatefulSet and Service
kubectl delete statefulset gpu-cluster-test-ddp --ignore-not-found=true
kubectl delete service gpu-cluster-test-ddp --ignore-not-found=true

# Or use the cleanup script
./scripts/cleanup-k8s-tests.sh
```

**See**: [docs/CLEANUP_GUIDE.md](docs/CLEANUP_GUIDE.md) for complete cleanup documentation.

---

## Troubleshooting

### Common Issues

1. **NCCL Timeout Errors**
   - Check network connectivity between nodes
   - Verify `NCCL_SOCKET_IFNAME` matches your high-speed network interface
   - Increase `NCCL_TIMEOUT_MS` environment variable

2. **Connection Refused**
   - Verify `MASTER_ADDR` is reachable from all nodes
   - Check firewall rules allow traffic on `MASTER_PORT`

3. **Out of Memory**
   - Reduce `--batch-size`
   - Use `transformer` model for memory pressure testing

4. **Slow Performance**
   - Check GPU utilization with `nvidia-smi`
   - Review NCCL logs for communication bottlenecks
   - Verify GPUs are connected via NVLink (not PCIe)

5. **Pods Not Scheduling on GPU Nodes (Mixed Kubernetes Clusters)**
   - **Symptom**: Pods stuck in `Pending` state or scheduled on non-GPU nodes
   - **Solutions**:
     ```bash
     # Check if GPU nodes are available
     kubectl get nodes -o json | jq '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | .metadata.name'
     
     # Verify NVIDIA device plugin is running
     kubectl get pods -n kube-system | grep nvidia
     
     # Check node labels
     kubectl get nodes --show-labels | grep gpu
     
     # Add nodeSelector to your PyTorchJob if using custom labels
     # See Kubernetes deployment example above
     ```
   
   - **If GPU nodes are tainted**, add tolerations to your pod spec:
     ```yaml
     tolerations:
     - key: nvidia.com/gpu
       operator: Exists
       effect: NoSchedule
     ```

For detailed troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) (common failures) and
[docs/INFINIBAND_CONFIGURATION.md](docs/INFINIBAND_CONFIGURATION.md) (NCCL/InfiniBand tuning).

## Architecture

```
gpu_cluster_testing/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet18.py       # ResNet-18 implementation
│   │   ├── resnet.py          # ResNet-50 implementation
│   │   └── transformer.py     # Transformer implementation
│   ├── data_utils.py           # Synthetic data generation
│   ├── dataset_loaders.py       # Optional real dataset loaders
│   └── train.py                # Main training script
├── scripts/
│   └── entrypoint.sh           # Universal environment detection
│   └── verify-k8s-gpu-cluster.sh # Optional cluster verification helper
├── examples/                    # Kubernetes examples
├── Dockerfile                  # Container definition
└── .github/
    └── workflows/
        └── ci.yml              # CI/CD pipeline
```

## Contributing

Contributions are welcome! Please see the [Implementation Plan](docs/Exercise%202%20Implementation%20Plan.md) for details.

## Documentation

### Core Guides
- **[How It Works](docs/HOW_IT_WORKS.md)** - Architecture, data flow, and testing methodology
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions (UCX/UCC, NCCL, OOM, etc.)
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Codebase structure and organization

### Advanced Topics
- **[InfiniBand Configuration](docs/INFINIBAND_CONFIGURATION.md)** - NCCL setup for InfiniBand/RoCE clusters
- **[NCCL Testing](docs/NCCL_TESTING.md)** - Direct NCCL bandwidth/latency measurement
- **[Cleanup Guide](docs/CLEANUP_GUIDE.md)** - Resource cleanup strategies for all platforms

## Examples

### Quick Start (Single GPU, No Operator Required)
```bash
### Run All Tests (Recommended)
```bash
./scripts/run-all-tests.sh --timeout 900
```
Runs Single GPU → Multi-GPU → Multi-Node DDP with automatic cleanup.

### Single GPU Test
```bash
kubectl apply -f examples/kubernetes-pod-single-gpu.yaml
kubectl logs -f pod/gpu-cluster-test-single-gpu
```
Simple single-GPU test—works on any Kubernetes cluster with GPUs.

### Multi-GPU Single Node
```bash
kubectl apply -f examples/kubernetes-pod-multi-gpu-single-node.yaml
kubectl logs -f pod/gpu-cluster-test-multi-gpu-single-node
```
Single-node torchrun DDP test with NCCL.

### Multi-Node DDP (InfiniBand)
```bash
kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml
kubectl logs -f pod/gpu-cluster-test-ddp-0
```
StatefulSet with anti-affinity for true multi-node RDMA/InfiniBand testing.

See `examples/` directory for all Kubernetes manifests.

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- **GitHub Issues**: Report bugs or request features
- **Documentation**: See [docs/](docs/) directory for detailed guides
