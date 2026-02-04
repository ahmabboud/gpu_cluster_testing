# NCCL Testing Guide

This guide explains how to perform focused NCCL testing alongside the full training acceptance tests.

## Overview

Our tool provides **two complementary approaches** to GPU cluster acceptance testing:

1. **Full Training Tests** (main tool): Realistic workload with actual model training
2. **NCCL Bandwidth Tests** (this guide): Direct NCCL performance measurement

Both are valuable and serve different purposes.

## When to Use Each Approach

### Use Full Training Tests When:
- ✅ Validating end-to-end ML workload performance
- ✅ Testing real application behavior (data loading, training loops, checkpointing)
- ✅ Acceptance testing for production ML workloads
- ✅ Testing framework-level integrations (PyTorch DDP, datasets, etc.)
- ✅ You want portable tests that work anywhere (no dependencies on NCCL test binaries)

### Use NCCL Bandwidth Tests When:
- ✅ Isolating pure communication performance
- ✅ Debugging network issues (InfiniBand vs NVLink)
- ✅ Measuring raw bandwidth and latency
- ✅ Validating NCCL configuration
- ✅ Quick infrastructure validation (< 5 minutes)
- ✅ Comparing against reference benchmarks

## Approach Comparison

| Aspect | Full Training Tests | NCCL Bandwidth Tests |
|--------|---------------------|----------------------|
| **Runtime** | 10-30 minutes | 2-5 minutes |
| **Dependencies** | PyTorch container | NCCL test binaries or container |
| **Realism** | High (actual workload) | Medium (synthetic) |
| **Debugging** | Application + network | Network only |
| **Portability** | Excellent | Good (needs nccl-tests) |
| **Acceptance** | Production readiness | Infrastructure readiness |

## Option 1: Using NCCL Test Binaries (Slurm)

### Prerequisites

Install NCCL test suite:
```bash
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/path/to/mpi
export PATH=$PWD/build:$PATH
```

### Running Tests

Use the provided Slurm script:
```bash
# Submit the job
sbatch examples/slurm-nccl-test.sh

# Monitor output
tail -f results/nccl_bandwidth_*.out
```

The script tests:
1. **NVLink Performance**: GPU-to-GPU on single node
2. **InfiniBand Performance**: Forced IB by disabling P2P/SHM
3. **Multi-Node All-Reduce**: Cross-node communication
4. **Latency**: Small message performance

### Expected Results

**H100 PCIe with NVLink:**
```
Size      Count    Type   Time    AlgBw   BusBw
8K        1000     float  26.23   0.00    0.00
16K       1000     float  26.89   0.00    0.00
...
8G        1000     float  21.44   373.46  653.56  <-- Target: >350 GB/s
```

**InfiniBand HDR (200 Gbps):**
```
8G        1000     float  35.12   227.89  398.81  <-- Target: >200 GB/s
```

**Multi-Node Latency:**
```
8         10000    float  48.21   0.00    0.00    <-- Target: <50 μs
```

## Option 2: Using Our Container (Kubernetes/Slurm)

Our container includes NCCL tests in `/workspace/nccl-tests/`.

### Quick Test on Kubernetes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nccl-bandwidth-test
spec:
  restartPolicy: Never
  containers:
  - name: nccl-test
    image: cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest
    command: ["/bin/bash", "-c"]
    args:
      - |
        cd /workspace/nccl-tests
        mpirun --allow-run-as-root -np 8 --bind-to none \
          ./build/all_reduce_perf -b 8K -e 8G -f 2 -g 1
    resources:
      limits:
        nvidia.com/gpu: 8
  nodeSelector:
    nvidia.com/gpu.product: NVIDIA-H100-PCIe
```

### Quick Test on Slurm

```bash
# Single node test
srun --nodes=1 --gpus-per-node=8 --container-image=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest \
  bash -c 'cd /workspace/nccl-tests && mpirun --allow-run-as-root -np 8 ./build/all_reduce_perf -b 8K -e 8G -f 2 -g 1'

# Multi-node test
srun --nodes=2 --ntasks-per-node=8 --gpus-per-node=8 --mpi=pmix \
  --container-image=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest \
  bash -c 'cd /workspace/nccl-tests && ./build/all_reduce_perf_mpi -b 8K -e 8G -f 2 -g 1'
```

## Option 3: Testing Specific Network Modes

### Test NVLink Only (Single Node)

```bash
# Default behavior - uses NVLink between GPUs on same node
srun --nodes=1 --gpus-per-node=8 \
  --container-image=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest \
  bash -c 'cd /workspace/nccl-tests && mpirun --allow-run-as-root -np 8 ./build/all_reduce_perf -b 512M -e 8G -f 2 -g 1'
```

**Expected**: 400-450 GB/s for H100 with NVLink

### Test InfiniBand Only (Force IB)

```bash
# Disable P2P and SHM to force InfiniBand usage
srun --nodes=1 --gpus-per-node=8 \
  --container-image=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest \
  bash -c '
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_ALGO=Ring
    export NCCL_DEBUG=INFO
    cd /workspace/nccl-tests
    mpirun --allow-run-as-root -np 8 ./build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
  '
```

**Expected**: 200-240 GB/s for HDR InfiniBand (200 Gbps per port)

### Test Multi-Node InfiniBand

```bash
srun --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 --mpi=pmix \
  --container-image=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu-cluster-test:latest \
  bash -c '
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=INFO
    cd /workspace/nccl-tests
    ./build/all_reduce_perf_mpi -b 512M -e 8G -f 2 -g 1
  '
```

**Expected**: Scales with number of nodes while maintaining >200 GB/s per node

## Interpreting Results

### Understanding nccl-tests Output

```
# out-of-place | in-place
       size         count      type   redop     time   algbw   busbw  error
        (B)    (elements)                       (us)  (GB/s)  (GB/s)       
     524288         131072     float     sum   1521.9   0.34   0.64  0e+00
    1048576         262144     float     sum   1845.3   0.57   1.06  0e+00
    ...
  8589934592     2147483648     float     sum  21443.2  400.47  700.82  0e+00
```

- **size**: Message size in bytes
- **time**: Time in microseconds
- **algbw**: Algorithm bandwidth (GB/s) - raw communication
- **busbw**: Bus bandwidth (GB/s) - effective bandwidth accounting for bidirectional traffic
- **error**: Numerical error (should be 0e+00)

### Key Metrics

1. **Large Message Bandwidth** (8G): Should match hardware specs
   - H100 NVLink: >400 GB/s
   - InfiniBand HDR: >200 GB/s
   - Ethernet 100GbE: >10 GB/s

2. **Small Message Latency** (8K-64K): Should be low
   - NVLink: <30 μs
   - InfiniBand: <50 μs
   - Ethernet: <100 μs

3. **Scaling**: As you increase nodes, per-node bandwidth should stay consistent

### Red Flags

❌ **Low Bandwidth**: < 50% of expected → Check network configuration
❌ **High Latency**: > 200 μs → Check for congestion or misconfig
❌ **Poor Scaling**: Bandwidth drops with more nodes → Check network topology
❌ **NCCL Errors**: Any error messages → Check InfiniBand/RDMA setup

## Troubleshooting

### If NCCL Tests Show Low Bandwidth

1. **Check InfiniBand Status**:
   ```bash
   ibstat
   ibstatus
   ```

2. **Verify RDMA Modules**:
   ```bash
   lsmod | grep -E 'ib|rdma'
   ```

3. **Check NCCL Configuration**:
   ```bash
   export NCCL_DEBUG=INFO
   # Re-run test and look for which transport is used
   ```

4. **Verify GPU Topology**:
   ```bash
   nvidia-smi topo -m
   ```

### If Multi-Node Tests Fail

1. **Check MPI Setup**:
   ```bash
   srun --nodes=2 hostname  # Should show both nodes
   ```

2. **Verify InfiniBand Connectivity**:
   ```bash
   srun --nodes=2 ibstatus  # Should show active ports
   ```

3. **Test Basic NCCL**:
   ```bash
   # Start with 2 GPUs, 2 nodes
   srun --nodes=2 --ntasks=2 --gpus-per-task=1 --mpi=pmix \
     all_reduce_perf_mpi -b 1M -e 1M -g 1
   ```

## Integration with Full Training Tests

**Recommended Workflow**:

1. **Quick NCCL Check** (5 min):
   ```bash
   sbatch examples/slurm-nccl-test.sh
   ```
   ➜ Validates raw network performance

2. **Full Training Test** (20 min):
   ```bash
   sbatch examples/slurm-multi-node.sbatch
   ```
   ➜ Validates complete ML stack

3. **Compare Results**:
   - NCCL bandwidth should be close to hardware limits
   - Training throughput should scale with NCCL performance
   - If NCCL is good but training is slow → application issue
   - If both are slow → infrastructure issue

## Reference Benchmarks

### Expected NCCL All-Reduce Performance

| Hardware | NVLink BW | InfiniBand BW | Latency |
|----------|-----------|---------------|---------|
| **H100 SXM** | 450-490 GB/s | 220-240 GB/s | 20-30 μs |
| **H100 PCIe** | 400-450 GB/s | 200-220 GB/s | 25-35 μs |
| **A100 SXM** | 300-330 GB/s | 180-200 GB/s | 25-40 μs |
| **A100 PCIe** | 280-310 GB/s | 180-200 GB/s | 30-45 μs |

### Multi-Node Scaling

| Nodes | H100 8xGPU | Expected Behavior |
|-------|------------|-------------------|
| 1 | 400-450 GB/s | Baseline (NVLink) |
| 2 | 380-420 GB/s | 5-10% overhead |
| 4 | 360-400 GB/s | 10-15% overhead |
| 8 | 340-380 GB/s | 15-20% overhead |

## Summary

- **Use NCCL tests** for quick infrastructure validation and network debugging
- **Use full training tests** for realistic acceptance testing
- **Both approaches** are complementary and recommended for comprehensive cluster validation
- **Start with NCCL** to verify network, then run full training to verify complete stack

## Further Reading

- [NVIDIA NCCL Tests GitHub](https://github.com/NVIDIA/nccl-tests)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [InfiniBand Performance Tuning](https://docs.nvidia.com/networking/display/MLNXOFEDv461000/Performance+Tuning)
- [Our Main Acceptance Playbook](ACCEPTANCE_PLAYBOOK.md)
