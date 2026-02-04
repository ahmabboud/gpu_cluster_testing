# GPU Cluster Acceptance Testing Tool

**Zero-Dependency Distributed Training for Infrastructure Validation**

A portable, scale-agnostic tool for validating GPU cluster health, performance, and interconnect stability. Built for Nebius Infrastructure Engineers to run acceptance tests on new clusters regardless of size or orchestration layer.

**Internal Nebius Solution Library** | **Production-Ready** | **Nebius-Optimized**

## Features

- **🚀 Zero Dependencies (Default)**: Uses synthetic data generation - no external datasets required
- **📦 Real Dataset Support**: Optional FashionMNIST (Nebius-proven), CIFAR-10, CIFAR-100, or ImageNet subset
- **🎯 Multiple Models**:
  - **ResNet18** - 11M parameters, matches Nebius KubeRay production pattern (faster validation)
  - **ResNet50** - 25M parameters, comprehensive testing
  - **Transformer** - Configurable size, bandwidth testing
- **📊 Comprehensive Testing**: Tests GPU compute, memory, and NCCL communication
- **🔧 Universal Compatibility**: Works on Kubernetes, Slurm, and bare metal
- **📈 Performance Profiling**: Detailed metrics including throughput, step time, and NCCL overhead
- **🔍 Diagnostic Rich**: Verbose NCCL logging for troubleshooting
- **⚡ Two Test Modes**:
  - **Full Training Tests**: Realistic workload with ResNet or Transformer models
  - **NCCL Bandwidth Tests**: Direct NCCL performance measurement (see [NCCL Testing Guide](docs/NCCL_TESTING.md))
- **🎯 Flexible Configuration**:
  - Configurable GPU count (1-8+ GPUs per worker)
  - Automatic resource scaling recommendations
  - No hardcoded assumptions
- **🏭 Production Patterns**: Based on proven Nebius production deployments
  - ResNet18 + FashionMNIST (matches KubeRay test)
  - Multi-GPU worker support
  - Shared memory configuration for DataLoader workers
  - Init containers for ulimit configuration
  - InfiniBand/RDMA optimization

## Quick Start

### Pull the Container

```bash
# From GitHub Container Registry (public)
docker pull ghcr.io/ahmabboud/gpu_cluster_testing:latest

# Or build locally (for AMD64 GPU servers)
docker build --platform linux/amd64 -t gpu_cluster_testing:latest .
```

### Single GPU Test

```bash
docker run --gpus all --rm \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest \
  --model resnet18 \
  --batch-size 128 \
  --data-mode fashion_mnist
```

### Quick Test (Matching Nebius KubeRay Pattern)

```bash
# ResNet18 + FashionMNIST - proven in production
docker run --gpus all --rm --ipc=host \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest \
  --model resnet18 \
  --batch-size 128 \
  --data-mode fashion_mnist
```

### Multi-GPU Test (Single Node)

```bash
# Using torchrun (recommended)
docker run --gpus all --rm --ipc=host \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  bash -c "cd /workspace/src && torchrun --nproc_per_node=8 train.py --model resnet50 --batch-size 32"
```

### Multi-Node Test Configuration

For multi-node testing, proper NCCL configuration is critical:

**With InfiniBand:**
```bash
docker run --gpus all --rm --ipc=host --network=host \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=mlx5_0 \
  -e NCCL_DEBUG=INFO \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --batch-size 64
```

**With Ethernet:**
```bash
docker run --gpus all --rm --ipc=host --network=host \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_DEBUG=INFO \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --batch-size 64
```

## Deployment Examples

### Kubernetes (with PyTorch Operator)

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
            image: cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
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
            image: cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
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
kubectl logs -f pytorch-job-master-0
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

### Slurm

Create a Slurm batch script `acceptance_test.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=gpu-acceptance
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=acceptance-%j.out

# Load container runtime
module load singularity

# Convert Docker image to Singularity (if needed)
# singularity pull docker://cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest

# Run with Slurm
srun singularity exec --nv \
  gpu_cluster_testing_latest.sif \
  /workspace/scripts/entrypoint.sh \
  --model resnet50 \
  --batch-size 64 \
  --active-iterations 100
```

Submit:
```bash
sbatch acceptance_test.sh
```

### Bare Metal (Multi-Node)

On the master node:

```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
export WORLD_SIZE=16  # Total number of GPUs
export NCCL_SOCKET_IFNAME=eth0  # Adjust to your network interface

# Start rank 0-7
for i in {0..7}; do
  RANK=$i LOCAL_RANK=$i \
    docker run --gpus device=$i --rm --network=host \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e RANK=$i \
    -e WORLD_SIZE=$WORLD_SIZE \
    -e LOCAL_RANK=$i \
    cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
    --model resnet50 --batch-size 32 &
done
```

On worker nodes (adjust RANK values accordingly):

```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500
export WORLD_SIZE=16

# Start rank 8-15 on second node
for i in {0..7}; do
  RANK=$((i+8)) LOCAL_RANK=$i \
    docker run --gpus device=$i --rm --network=host \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e RANK=$((i+8)) \
    -e WORLD_SIZE=$WORLD_SIZE \
    -e LOCAL_RANK=$i \
    cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
    --model resnet50 --batch-size 32 &
done
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `resnet50` | Model type: `resnet50` or `transformer` |
| `--data-mode` | `synthetic` | Data source: `synthetic`, `cifar10`, `cifar100`, or `imagenet` |
| `--data-dir` | `./data` | Directory for dataset storage (if using real datasets) |
| `--batch-size` | `32` | Batch size per GPU |
| `--num-classes` | `1000` | Number of classes (vocab size for transformer) |
| `--learning-rate` | `0.01` | Learning rate |
| `--warmup-iterations` | `50` | Number of warmup iterations |
| `--active-iterations` | `100` | Number of measurement iterations |

### Data Modes

**Synthetic (Default, Recommended for Acceptance Testing):**
```bash
docker run --gpus all --rm cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --data-mode synthetic
```

**CIFAR-10 (Lightweight, 170MB, Auto-download):**
```bash
docker run --gpus all --rm -v $(pwd)/data:/workspace/data \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --data-mode cifar10 --num-classes 10
```

**CIFAR-100 (Lightweight, 170MB, Auto-download):**
```bash
docker run --gpus all --rm -v $(pwd)/data:/workspace/data \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --data-mode cifar100 --num-classes 100
```

**ImageNet Subset (Requires pre-downloaded dataset):**
```bash
# First, download ImageNet validation set to ./data/imagenet/
docker run --gpus all --rm -v $(pwd)/data:/workspace/data \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  --model resnet50 --data-mode imagenet --num-classes 1000
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

## Expected Performance (Baseline)

See the [Acceptance Playbook](docs/ACCEPTANCE_PLAYBOOK.md) for detailed H100 benchmarks and troubleshooting guidance.

### Quick Reference (H100 80GB, NVLink)

| Configuration | Model | Batch/GPU | Expected Throughput |
|---------------|-------|-----------|---------------------|
| 8x H100 | ResNet-50 | 64 | ~14,000 samples/sec |
| 32x H100 | ResNet-50 | 64 | ~50,000 samples/sec |
| 8x H100 | Transformer | 32 | ~8,000 samples/sec |

## NCCL Bandwidth Testing

**NEW**: The container now includes official NVIDIA NCCL test binaries for direct bandwidth/latency measurements.

**See [NCCL Testing Guide](docs/NCCL_TESTING.md) for complete documentation.**

### Quick NCCL Test (Single Node, 8 GPUs)

Test NVLink bandwidth:
```bash
docker run --gpus all --rm --ipc=host \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  bash -c "cd /workspace/nccl-tests && mpirun --allow-run-as-root -np 8 ./build/all_reduce_perf -b 8K -e 8G -f 2 -g 1"
```

Expected result: **400-450 GB/s** for H100 with NVLink

### Test InfiniBand (Force IB Usage)

```bash
docker run --gpus all --rm --ipc=host --network=host \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest \
  bash -c "
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_ALGO=Ring
    export NCCL_DEBUG=INFO
    cd /workspace/nccl-tests
    mpirun --allow-run-as-root -np 8 ./build/all_reduce_perf -b 8K -e 8G -f 2 -g 1
  "
```

Expected result: **200-240 GB/s** for HDR InfiniBand

### Multi-Node NCCL Test (Slurm)

Use the provided script:
```bash
sbatch examples/slurm-nccl-test.sh
tail -f results/nccl_bandwidth_*.out
```

Tests performed:
1. NVLink performance (single node)
2. InfiniBand performance (forced IB)
3. Multi-node all-reduce
4. Latency measurement

**When to use NCCL tests vs Full Training:**
- **NCCL tests** (2-5 min): Quick infrastructure validation, network debugging
- **Full training** (10-30 min): Realistic acceptance testing, end-to-end validation

Both approaches are complementary. See [NCCL Testing Guide](docs/NCCL_TESTING.md) for detailed comparison.

## Environment Variables

### Detected Automatically

- **Slurm**: `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`, `SLURM_NODELIST`
- **Kubernetes**: Typically set by PyTorch operators

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

## Building from Source

```bash
git clone https://github.com/YOUR_USERNAME/gpu_cluster_testing.git
cd gpu_cluster_testing

# Build container (for AMD64 GPU servers)
docker build --platform linux/amd64 -t gpu-cluster-testing:local .

# Run local build
docker run --gpus all --rm gpu-cluster-testing:local
```

## Resource Cleanup

⚠️ **Important**: By default, test resources remain in the cluster after completion for log inspection.

### Cleanup Behavior by Platform

| Platform | Behavior | Recommendation |
|----------|----------|----------------|
| **Docker** | ✅ Auto-cleanup (all examples use `--rm`) | No action needed |
| **Kubernetes** | ❌ Jobs/pods remain after completion | Configure TTL or manual cleanup |
| **Slurm** | ⚠️ Jobs terminate, logs remain | Implement log rotation |

### Quick Cleanup Commands

**Kubernetes**:
```bash
# Delete specific test
kubectl delete pytorchjob gpu-cluster-acceptance-test

# Delete all completed tests
kubectl delete pytorchjob -l app=gpu-acceptance-test

# Automated cleanup script
./scripts/cleanup-k8s-tests.sh default 24  # Clean up tests older than 24h
```

**Slurm**:
```bash
# Clean up old logs
./scripts/cleanup-slurm-logs.sh ./results 7  # Keep last 7 days
```

### Automatic Cleanup (Kubernetes)

Use the example with TTL for automatic cleanup:
```bash
kubectl apply -f examples/kubernetes-with-auto-cleanup.yaml
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

For detailed troubleshooting, see the [Acceptance Playbook](docs/ACCEPTANCE_PLAYBOOK.md).

## Architecture

```
gpu_cluster_testing/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py          # ResNet-50 implementation
│   │   └── transformer.py     # Transformer implementation
│   ├── data_utils.py           # Synthetic data generation
│   └── train.py                # Main training script
├── scripts/
│   └── entrypoint.sh           # Universal environment detection
├── Dockerfile                  # Container definition
└── .github/
    └── workflows/
        └── ci.yml              # CI/CD pipeline
```

## Contributing

Contributions are welcome! Please see the [Implementation Plan](docs/Exercise%202%20Implementation%20Plan.md) for details.

## Documentation

### Getting Started
- **[How It Works](docs/HOW_IT_WORKS.md)** - Complete explanation of architecture, data flow, and testing methodology
- **[Nebius Registry Guide](docs/NEBIUS_REGISTRY_GUIDE.md)** - How to push/pull images to Nebius Container Registry

### Configuration Guides
- **[InfiniBand Configuration](docs/INFINIBAND_CONFIGURATION.md)** - NCCL setup for InfiniBand/RoCE clusters
- **[Learnings from Nebius](docs/LEARNINGS_FROM_NEBIUS.md)** - Best practices from production deployments

### Testing & Performance
- **[NCCL Testing Guide](docs/NCCL_TESTING.md)** - Direct NCCL bandwidth/latency testing
- **[Acceptance Playbook](docs/ACCEPTANCE_PLAYBOOK.md)** - Benchmarks and troubleshooting guide

### Operations
- **[Cleanup Guide](docs/CLEANUP_GUIDE.md)** - Resource cleanup strategies for all platforms
- **[Complete Summary](docs/COMPLETE_SUMMARY.md)** - Overview of all features and enhancements

### Reference
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Detailed codebase structure
- **[Implementation Plan](docs/Exercise%202%20Implementation%20Plan.md)** - Development roadmap

## Examples

### Quick Start (Nebius Pattern - ResNet18 + FashionMNIST)
```bash
kubectl apply -f examples/kubernetes-flexible-nebius-pattern.yaml
```
Matches Nebius KubeRay production test - configurable GPU count.

### Production (Multi-GPU, Nebius-Optimized)
```bash
kubectl apply -f examples/kubernetes-multi-gpu-nebius-optimized.yaml
```
Fixed 8-GPU configuration for H100 clusters.

### Standard (Any Kubernetes Cluster)
```bash
kubectl apply -f examples/kubernetes-pytorch-multi-node.yaml
```
Simple 1-GPU per worker configuration.

See `examples/` directory for:
- **kubernetes-flexible-nebius-pattern.yaml** - Configurable GPU count, matches KubeRay (ResNet18 + FashionMNIST)
- **kubernetes-multi-gpu-nebius-optimized.yaml** - Fixed 8-GPU H100 configuration
- **kubernetes-pytorch-multi-node.yaml** - Standard Kubernetes (any cluster)
- **kubernetes-with-auto-cleanup.yaml** - Automated cleanup configuration
- **slurm-acceptance-test.sh** - Slurm job script

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- **Nebius Infrastructure Team** (internal Slack or contact)
- Documentation: [docs/ACCEPTANCE_PLAYBOOK.md](docs/ACCEPTANCE_PLAYBOOK.md)
