# 🚀 InfiniBand and NCCL Configuration Guide

## Overview

This guide covers NCCL configuration for high-performance GPU clusters, based on learnings from Nebius production deployments and the official NCCL documentation.

---

## Quick Reference: NCCL Environment Variables

### Essential Variables

```bash
# Network Interface Selection
NCCL_SOCKET_IFNAME=eth0              # Ethernet (default)
NCCL_SOCKET_IFNAME=ib0               # InfiniBand
NCCL_SOCKET_IFNAME=^lo,docker0       # Exclude interfaces

# InfiniBand/RDMA Configuration
NCCL_IB_DISABLE=0                    # Enable InfiniBand (default: auto-detect)
NCCL_IB_HCA=mlx5                     # Mellanox HCA prefix
NCCL_IB_TIMEOUT=23                   # Timeout (default: 18)
NCCL_IB_RETRY_CNT=7                  # Retry count (default: 7)

# Debugging
NCCL_DEBUG=INFO                      # WARN (default), INFO, TRACE
NCCL_DEBUG_SUBSYS=ALL                # ALL, INIT, COLL, P2P, SHM, NET, etc.

# Performance Tuning
NCCL_MIN_NCHANNELS=4                 # Minimum channels (default: auto)
NCCL_MAX_NCHANNELS=16                # Maximum channels (default: auto)
NCCL_BUFFSIZE=2097152                # Buffer size in bytes (default: 4MB)
```

---

## Network Topology Patterns

### Pattern 1: Single Node with NVLink (8 GPUs)

**Hardware**:
- 8× NVIDIA H100/A100 GPUs
- NVLink/NVSwitch for GPU-to-GPU communication
- No InfiniBand needed

**Configuration**:
```yaml
env:
  - name: NCCL_DEBUG
    value: "WARN"  # INFO for debugging
  # No special configuration needed - NVLink auto-detected
```

**Expected Performance**:
- **H100 NVLink**: ~400-450 GB/s All-Reduce bandwidth
- **A100 NVLink**: ~300-350 GB/s All-Reduce bandwidth

**Verification**:
```bash
# Check NVLink status
nvidia-smi nvlink --status

# Expected output: All links "UP"
```

---

### Pattern 2: Multi-Node with InfiniBand (Nebius-style)

**Hardware**:
- Multiple nodes, each with 8 GPUs
- 8× InfiniBand adapters per node (1 per GPU)
- Mellanox ConnectX-6/7 adapters
- HDR InfiniBand (200 Gb/s per port)

**Configuration**:
```yaml
env:
  # Network Interface
  - name: NCCL_SOCKET_IFNAME
    value: "ib0"  # or eth0 if IB runs over IP
  
  # InfiniBand Settings
  - name: NCCL_IB_DISABLE
    value: "0"  # Explicitly enable
  
  - name: NCCL_IB_HCA
    value: "mlx5"  # Mellanox HCA prefix
  
  # UCX Configuration (Advanced)
  - name: UCX_NET_DEVICES
    value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
  
  - name: UCX_TLS
    value: "rc_x,ud_x,sm,cuda_copy,cuda_ipc"
  
  # Performance Tuning
  - name: NCCL_IB_TIMEOUT
    value: "23"
  
  - name: NCCL_IB_RETRY_CNT
    value: "7"
  
  # Debugging (set to WARN in production)
  - name: NCCL_DEBUG
    value: "INFO"
  
  - name: NCCL_DEBUG_SUBSYS
    value: "INIT,NET"

# Optional: Privileged mode for RDMA
securityContext:
  privileged: true
  capabilities:
    add:
    - IPC_LOCK
    - SYS_ADMIN
```

**Expected Performance**:
- **Single node**: ~400 GB/s (NVLink dominates)
- **Multi-node**: ~150-200 GB/s (InfiniBand bottleneck)

**Device Mapping**:
```
GPU 0 ↔ mlx5_0 (IB port 1)
GPU 1 ↔ mlx5_1 (IB port 1)
GPU 2 ↔ mlx5_2 (IB port 1)
GPU 3 ↔ mlx5_3 (IB port 1)
GPU 4 ↔ mlx5_4 (IB port 1)
GPU 5 ↔ mlx5_5 (IB port 1)
GPU 6 ↔ mlx5_6 (IB port 1)
GPU 7 ↔ mlx5_7 (IB port 1)
```

---

### Pattern 3: Multi-Node with RoCE (RDMA over Converged Ethernet)

**Hardware**:
- Standard Ethernet NICs with RoCE support
- 100 Gb/s or 200 Gb/s Ethernet
- No dedicated InfiniBand fabric

**Configuration**:
```yaml
env:
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"  # or specific RoCE interface
  
  - name: NCCL_IB_DISABLE
    value: "0"  # RoCE uses IB protocol
  
  # RoCE-specific
  - name: NCCL_IB_GID_INDEX
    value: "3"  # RoCE v2 (check with `show_gids`)
  
  - name: NCCL_IB_TC
    value: "106"  # Traffic class for PFC
  
  # Network configuration
  - name: NCCL_NET_GDR_LEVEL
    value: "5"  # GPUDirect RDMA level
  
  - name: NCCL_DEBUG
    value: "INFO"
```

**Expected Performance**:
- **100 Gb/s RoCE**: ~10-12 GB/s per NIC
- **200 Gb/s RoCE**: ~20-24 GB/s per NIC

---

### Pattern 4: Cloud Ethernet (AWS, Azure, GCP)

**Hardware**:
- No InfiniBand or RoCE
- Standard cloud networking (25-100 Gb/s)
- EFA (AWS) or similar cloud-native RDMA

**Configuration**:
```yaml
env:
  - name: NCCL_SOCKET_IFNAME
    value: "eth0"
  
  # Disable InfiniBand
  - name: NCCL_IB_DISABLE
    value: "1"
  
  # Use sockets for inter-node
  - name: NCCL_NET
    value: "Socket"
  
  # AWS EFA-specific (if available)
  - name: FI_PROVIDER
    value: "efa"
  
  - name: FI_EFA_USE_DEVICE_RDMA
    value: "1"
  
  - name: NCCL_DEBUG
    value: "WARN"
```

**Expected Performance**:
- **25 Gb/s Ethernet**: ~2.5 GB/s
- **100 Gb/s Ethernet**: ~10 GB/s
- **AWS EFA**: ~15-20 GB/s (with RDMA)

---

## Nebius-Specific Configuration

### H100 Cluster (8 GPUs per node, InfiniBand)

Based on `nebius-solutions-library/modules/nccl-test/`:

```yaml
apiVersion: kubeflow.org/v2beta1
kind: PyTorchJob
metadata:
  name: gpu-test-nebius-h100
spec:
  pytorchReplicaSpec:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: ghcr.io/ahmabboud/gpu_cluster_testing:latest
            resources:
              limits:
                nvidia.com/gpu: 8
                cpu: 112
                memory: 1200Gi
            env:
            # Nebius InfiniBand Configuration
            - name: NCCL_DEBUG
              value: "INFO"
            
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"  # Check with: ip a
            
            - name: NCCL_IB_HCA
              value: "mlx5"
            
            - name: UCX_NET_DEVICES
              value: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
            
            - name: SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING
              value: "1"
            
            - name: NCCL_COLLNET_ENABLE
              value: "0"  # Disable SHARP if not available
            
            securityContext:
              privileged: true
            
            volumeMounts:
            - name: dshm
              mountPath: /dev/shm
          
          volumes:
          - name: dshm
            emptyDir:
              medium: Memory
```

---

## Diagnostic Commands

### 1. Check Network Interfaces

```bash
# List all network interfaces
ip a

# Expected output for InfiniBand:
# 4: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP>
#     inet 10.0.0.1/24

# Check InfiniBand status
ibstat

# Expected output:
# CA 'mlx5_0'
#     Port 1:
#         State: Active
#         Physical state: LinkUp
#         Rate: 200 (HDR)
```

### 2. Check RDMA Devices

```bash
# List RDMA devices
ibv_devices

# Expected output:
#     device                 node GUID
#     ------              ----------------
#     mlx5_0              506b4b0300abcd00
#     mlx5_1              506b4b0300abcd01
#     ...

# Device capabilities
ibv_devinfo -d mlx5_0
```

### 3. Check GPU-to-InfiniBand Mapping

```bash
# Show GPU topology
nvidia-smi topo -m

# Expected output shows GPU affinity to NUMA nodes and NICs
#         GPU0    GPU1    GPU2    GPU3    mlx5_0  mlx5_1
# GPU0     X      SYS     SYS     SYS     NODE    SYS
# GPU1    SYS      X      SYS     SYS     SYS     NODE
# ...
# 
# Legend:
#   X    = Self
#   SYS  = Connection traversing PCIe as well as NUMA
#   NODE = Connection traversing same NUMA node
#   PHB  = Connection traversing same PCIe bridge
```

### 4. Test NCCL Directly

```bash
# Single node test (8 GPUs)
/workspace/nccl-tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1

# Multi-node test with MPI
mpirun -np 16 \
  -H node1:8,node2:8 \
  -bind-to none \
  -x LD_LIBRARY_PATH \
  -x NCCL_DEBUG=INFO \
  -x NCCL_SOCKET_IFNAME=ib0 \
  -x NCCL_IB_HCA=mlx5 \
  /workspace/nccl-tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1
```

### 5. Monitor During Training

```bash
# Watch NCCL logs
kubectl logs -f pytorchjob-master-0 | grep NCCL

# Expected output (good):
# NCCL INFO Bootstrap : Using ib0:10.0.0.1<0>
# NCCL INFO NET/Plugin: Using network plugin: libncclnc.so
# NCCL INFO Using network NCCL_IB
# NCCL INFO Channel 00/04 :    0   1   2   3   4   5   6   7

# Bad signs:
# NCCL WARN Net : No GPU affinity set, could lead to poor performance
# NCCL WARN Failed to open mlx5_0
```

---

## Performance Troubleshooting

### Issue: Low Bandwidth (< 100 GB/s for H100)

**Possible Causes**:
1. Not using InfiniBand → Check `NCCL_SOCKET_IFNAME`
2. InfiniBand not detected → Check `ibstat`, verify `NCCL_IB_HCA`
3. Wrong GPU-to-NIC mapping → Check `nvidia-smi topo -m`
4. Network congestion → Check switch/fabric

**Solution**:
```yaml
# Enable verbose NCCL logging
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_DEBUG_SUBSYS
  value: "INIT,NET,GRAPH"

# Check logs for actual network used:
# "NCCL INFO Using network NCCL_IB" = Good (InfiniBand)
# "NCCL INFO Using network Socket" = Bad (TCP/IP)
```

---

### Issue: NCCL Timeouts

**Symptoms**:
```
NCCL WARN Timeout waiting for remote GPU
NCCL ERROR Timeout
```

**Solutions**:
```yaml
# Increase timeouts
- name: NCCL_IB_TIMEOUT
  value: "23"  # Default: 18

- name: NCCL_TIMEOUT
  value: "1800"  # 30 minutes in seconds

# Increase retries
- name: NCCL_IB_RETRY_CNT
  value: "10"  # Default: 7
```

---

### Issue: "No space left on device" Errors

**Cause**: `/dev/shm` too small for shared memory operations

**Solution**:
```yaml
volumeMounts:
- name: dshm
  mountPath: /dev/shm

volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 32Gi  # Increase as needed
```

---

### Issue: Poor GPU Affinity

**Symptoms**:
```
NCCL WARN No GPU affinity set, could lead to poor performance
```

**Solution**:
```yaml
# Let Kubernetes handle affinity
nodeSelector:
  node.kubernetes.io/instance-type: gpu-h100-8  # or similar

# Ensure processes bind to correct NUMA nodes (usually automatic)
```

---

## Validation Checklist

Before running production workloads:

- [ ] **Network connectivity**: `ping` between all nodes works
- [ ] **InfiniBand status**: `ibstat` shows "State: Active"
- [ ] **RDMA devices**: `ibv_devices` lists all expected adapters
- [ ] **GPU topology**: `nvidia-smi topo -m` shows correct affinity
- [ ] **NCCL test**: Single-node bandwidth ≥ 350 GB/s (H100)
- [ ] **NCCL test**: Multi-node bandwidth ≥ 150 GB/s (HDR IB)
- [ ] **Shared memory**: `/dev/shm` mounted with sufficient size
- [ ] **Resource limits**: No "Out of Memory" errors during training
- [ ] **Logs clean**: No NCCL WARN/ERROR messages

---

## Configuration Matrix

| Cluster Type | NCCL_SOCKET_IFNAME | NCCL_IB_DISABLE | NCCL_IB_HCA | Expected BW |
|--------------|-------------------|-----------------|-------------|-------------|
| **Single Node (NVLink)** | (any) | (any) | - | ~400 GB/s |
| **Nebius InfiniBand** | eth0 or ib0 | 0 | mlx5 | ~200 GB/s |
| **RoCE Cluster** | eth0 | 0 | - | ~100 GB/s |
| **Cloud (EFA/ENA)** | eth0 | 1 | - | ~20 GB/s |
| **Standard Ethernet** | eth0 | 1 | - | ~10 GB/s |

---

## References

1. **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
2. **Nebius Solutions Library**: `nebius-solutions-library/modules/nccl-test/`
3. **InfiniBand Debugging**: https://docs.nvidia.com/networking/display/MLNXOFEDv494180/Viewing+the+InfiniBand+Subnet
4. **PyTorch Distributed**: https://pytorch.org/docs/stable/distributed.html

---

## Quick Start Examples

### Nebius Production Config
```yaml
# Copy from: examples/kubernetes-multi-gpu-nebius-optimized.yaml
# Pre-configured for H100 with InfiniBand
```

### Generic InfiniBand
```yaml
# Copy from: examples/kubernetes-infiniband.yaml
# Template for custom InfiniBand clusters
```

### Standard Ethernet
```yaml
# Copy from: examples/kubernetes-pytorch-multi-node.yaml
# No InfiniBand, works on any Kubernetes
```

---

## Summary

**Key Takeaways**:
1. **NVLink** (single node): Zero config, ~400 GB/s
2. **InfiniBand** (multi-node): Needs `NCCL_IB_HCA`, ~200 GB/s
3. **Ethernet** (cloud): Set `NCCL_IB_DISABLE=1`, ~10 GB/s
4. **Always mount** `/dev/shm` for shared memory
5. **Always use** init container for ulimit
6. **Always check** logs for NCCL network selection

**When in Doubt**:
```yaml
# Start with this minimal config:
env:
  - name: NCCL_DEBUG
    value: "INFO"  # Check logs to see what NCCL detects

# Then tune based on what you see in logs
```
