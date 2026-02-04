# GPU Cluster Acceptance Playbook

**For Infrastructure Engineers**

This playbook provides guidance on interpreting test results, expected performance benchmarks, and troubleshooting common issues when validating new GPU clusters.

## Design Philosophy

This tool uses **synthetic data generation** instead of real datasets, providing:
- **Zero external dependencies** - No dataset downloads or storage requirements
- **Instant startup** - Tests run immediately without preprocessing
- **Hardware-agnostic** - Works on any NVIDIA GPU (V100, A100, H100, RTX series, etc.)
- **Consistent results** - Eliminates dataset variability from benchmarks

The models (ResNet-50, Transformer) are standard architectures implemented from scratch, ensuring complete portability and transparency.

## Table of Contents

1. [Baseline Performance Benchmarks](#baseline-performance-benchmarks)
2. [Interpreting Results](#interpreting-results)
3. [NCCL Troubleshooting Guide](#nccl-troubleshooting-guide)
4. [Hardware-Specific Considerations](#hardware-specific-considerations)
5. [Acceptance Criteria](#acceptance-criteria)

---

## Baseline Performance Benchmarks

### NVIDIA H100 80GB (NVLink 4.0)

#### ResNet-50 Performance

| Configuration | Batch/GPU | Global Batch | Throughput (samples/sec) | Step Time (ms) |
|---------------|-----------|--------------|-------------------------|----------------|
| 1x H100 | 64 | 64 | ~1,750 | ~37 |
| 8x H100 | 64 | 512 | ~14,000 | ~37 |
| 16x H100 | 64 | 1,024 | ~27,500 | ~37 |
| 32x H100 | 64 | 2,048 | ~53,000 | ~39 |
| 64x H100 | 64 | 4,096 | ~100,000 | ~41 |

**Expected Scaling Efficiency:**
- 8 GPUs: 98-100% (ideal)
- 16 GPUs: 95-98%
- 32 GPUs: 92-96%
- 64 GPUs: 88-93%

#### Transformer Performance

| Configuration | Batch/GPU | Seq Length | Throughput (samples/sec) | Step Time (ms) |
|---------------|-----------|------------|-------------------------|----------------|
| 1x H100 | 32 | 512 | ~1,000 | ~32 |
| 8x H100 | 32 | 512 | ~7,800 | ~33 |
| 16x H100 | 32 | 512 | ~15,200 | ~34 |
| 32x H100 | 32 | 512 | ~29,500 | ~35 |

**Expected Scaling Efficiency:**
- 8 GPUs: 97-99%
- 16 GPUs: 94-97%
- 32 GPUs: 90-94%

### NVIDIA A100 80GB (NVLink 3.0)

#### ResNet-50 Performance

| Configuration | Batch/GPU | Global Batch | Throughput (samples/sec) | Step Time (ms) |
|---------------|-----------|--------------|-------------------------|----------------|
| 1x A100 | 64 | 64 | ~1,200 | ~53 |
| 8x A100 | 64 | 512 | ~9,400 | ~54 |
| 16x A100 | 64 | 1,024 | ~18,000 | ~57 |
| 32x A100 | 64 | 2,048 | ~34,000 | ~60 |

**Expected Scaling Efficiency:**
- 8 GPUs: 97-99%
- 16 GPUs: 94-97%
- 32 GPUs: 88-93%

### NVIDIA A100 40GB (PCIe 4.0)

#### ResNet-50 Performance

| Configuration | Batch/GPU | Global Batch | Throughput (samples/sec) | Step Time (ms) |
|---------------|-----------|--------------|-------------------------|----------------|
| 1x A100 | 64 | 64 | ~1,150 | ~56 |
| 8x A100 | 64 | 512 | ~8,200 | ~62 |

**Note:** PCIe-based systems show reduced scaling efficiency compared to NVLink systems due to limited inter-GPU bandwidth.

### Other NVIDIA GPUs

This tool works on **any CUDA-capable NVIDIA GPU**. For GPUs not listed above:

| GPU Type | Relative Performance | Recommended Batch Size | Notes |
|----------|---------------------|----------------------|--------|
| V100 32GB | ~40% of H100 | 32-64 | Good baseline for older clusters |
| RTX 4090 24GB | ~50% of H100 | 16-32 | Excellent price/performance |
| RTX 3090 24GB | ~35% of H100 | 16-32 | Consumer-grade testing |
| T4 16GB | ~15% of H100 | 8-16 | Inference-focused, limited training |
| L40 48GB | ~55% of H100 | 32-64 | Good for mixed workloads |

**To establish baselines for other GPUs:**
1. Run single-GPU test to establish baseline
2. Run 8-GPU test to measure scaling efficiency
3. Compare step time and throughput against your expectations
4. Document results for future acceptance tests

---

## Interpreting Results

### Key Performance Indicators

#### 1. **Throughput (samples/sec)**

This is the primary metric for cluster acceptance.

**Good Performance:**
- Scales linearly up to 8 GPUs (95%+ efficiency)
- Maintains >90% scaling efficiency up to 32 GPUs
- No significant degradation after warmup

**Red Flags:**
- <80% scaling efficiency at 8 GPUs
- Throughput decreases with more GPUs
- Wide variance between runs (>10%)

#### 2. **Average Step Time**

Should remain relatively constant as you scale.

**Expected Behavior:**
- Single GPU: Baseline time
- 8 GPUs: +5-10% overhead (NCCL communication)
- 32+ GPUs: +10-20% overhead

**Red Flags:**
- Step time increases by >30% at scale
- High variance between steps
- Step time continues to increase over time (throttling)

#### 3. **Scaling Efficiency**

```
Efficiency = (Throughput_N_GPUs / N) / Throughput_1_GPU
```

**Acceptance Thresholds:**
- 8 GPUs: >95%
- 16 GPUs: >90%
- 32 GPUs: >85%
- 64 GPUs: >80%

### Sample Results Analysis

#### Example 1: Healthy Cluster

```json
{
  "model": "resnet50",
  "world_size": 32,
  "batch_size_per_gpu": 64,
  "global_batch_size": 2048,
  "total_time_seconds": 45.2,
  "average_step_time_seconds": 0.039,
  "throughput_samples_per_second": 52800,
  "device": "cuda:0",
  "backend": "nccl"
}
```

**Analysis:**
- ✅ Throughput: 52,800 samples/sec (99% of expected 53,000)
- ✅ Step time: 39ms (within expected range)
- ✅ Scaling efficiency: ~94% (excellent for 32 GPUs)
- **Verdict: PASS**

#### Example 2: Network Issues

```json
{
  "model": "resnet50",
  "world_size": 32,
  "batch_size_per_gpu": 64,
  "global_batch_size": 2048,
  "total_time_seconds": 120.5,
  "average_step_time_seconds": 0.105,
  "throughput_samples_per_second": 19800,
  "device": "cuda:0",
  "backend": "nccl"
}
```

**Analysis:**
- ❌ Throughput: 19,800 samples/sec (37% of expected)
- ❌ Step time: 105ms (2.7x expected)
- ❌ Scaling efficiency: ~35%
- **Verdict: FAIL - Investigate network/NCCL**

---

## NCCL Troubleshooting Guide

### Common Error Codes

#### 1. **Connection Timeout**

**Error Message:**
```
[Rank 0] NCCL WARN Connect to <host> failed : Connection timed out
```

**Causes:**
- Firewall blocking NCCL traffic
- Wrong network interface selected
- Network misconfiguration

**Diagnosis:**
```bash
# Check connectivity
ping <master_node_ip>

# Verify NCCL interface
echo $NCCL_SOCKET_IFNAME

# List network interfaces
ip addr show
```

**Solutions:**
```bash
# Set correct network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0 for InfiniBand

# Increase timeout
export NCCL_TIMEOUT_MS=600000  # 10 minutes

# Allow NCCL traffic through firewall
sudo ufw allow 29500/tcp  # MASTER_PORT
```

#### 2. **Connection Refused**

**Error Message:**
```
[Rank 1] NCCL WARN Connect to <master>:29500 failed : Connection refused
```

**Causes:**
- Master process not started yet
- Wrong MASTER_ADDR or MASTER_PORT
- Port already in use

**Solutions:**
```bash
# Verify master is listening
netstat -tulpn | grep 29500

# Use different port if in use
export MASTER_PORT=29501

# Add startup delay for workers
sleep 10  # Before starting worker ranks
```

#### 3. **Memory Allocation Failure**

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Causes:**
- Batch size too large
- Model too large for GPU memory
- Memory fragmentation

**Solutions:**
```bash
# Reduce batch size
--batch-size 32  # Instead of 64

# Enable gradient checkpointing (for transformer)
# This is a model modification, not a runtime flag

# Clear cache between runs
nvidia-smi --gpu-reset
```

#### 4. **NCCL Initialization Hang**

**Symptoms:**
- Process hangs at "Initializing process group"
- No error messages
- Timeout after several minutes

**Diagnosis:**
```bash
# Enable verbose NCCL debugging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,NET

# Check if all ranks can reach master
# On each node:
telnet $MASTER_ADDR $MASTER_PORT
```

**Solutions:**
```bash
# Ensure consistent environment across all ranks
# Check: WORLD_SIZE, MASTER_ADDR, MASTER_PORT

# Verify network routing
traceroute <master_node_ip>

# Try alternate NCCL transport
export NCCL_IB_DISABLE=1  # Disable InfiniBand
export NCCL_P2P_DISABLE=1  # Disable P2P
```

#### 5. **Network Bottleneck**

**Symptoms:**
- Scaling efficiency <50%
- High step time variance
- NCCL logs show long wait times

**Diagnosis:**
```bash
# Monitor network utilization
iftop -i eth0

# Check for network errors
netstat -i
cat /proc/net/dev

# NCCL transport detection
export NCCL_DEBUG=INFO
# Look for "Using network" in logs
```

**Solutions:**
```bash
# Use high-speed network interface
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand

# Enable RDMA (if available)
export NCCL_IB_HCA=mlx5_0

# Tune NCCL parameters
export NCCL_IB_TIMEOUT=23
export NCCL_IB_GID_INDEX=3
```

### NCCL Performance Tuning

#### For InfiniBand Networks

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1  # Use multiple HCAs
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=23
```

#### For RoCE Networks

```bash
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_NET_CHUNKSIZE=131072
```

#### For NVSwitch Systems

```bash
# NVSwitch typically auto-detected
# Verify with:
nvidia-smi nvlink --status
```

---

## Hardware-Specific Considerations

### H100 Systems

**Optimal Configurations:**
- Batch size: 64-128 per GPU for ResNet-50
- Batch size: 32-64 per GPU for Transformer
- Enable Transformer Engine (FP8) for maximum performance
- Use NVLink switches for multi-node setups

**Known Issues:**
- Temperature throttling at sustained >95% utilization
- ECC errors under extreme memory bandwidth workloads

### A100 Systems

**Optimal Configurations:**
- Batch size: 64 per GPU for ResNet-50
- Batch size: 16-32 per GPU for Transformer
- NVLink: Essential for good scaling
- PCIe: Expect 70-80% scaling efficiency at 8 GPUs

**Known Issues:**
- PCIe Gen3 systems show significant bottleneck beyond 4 GPUs
- MIG mode: Not recommended for this test

### Network Requirements

#### Minimum Specifications

| Scale | Minimum Network | Recommended Network |
|-------|----------------|---------------------|
| 1-8 GPUs (single node) | NVLink | NVLink |
| 8-32 GPUs | 100 GbE | 200 GbE or HDR IB |
| 32-128 GPUs | 200 GbE | 400 GbE or NDR IB |
| 128+ GPUs | 400 GbE | NDR IB + RoCE |

---

## Acceptance Criteria

### Pass Criteria (All Must Pass)

1. **No NCCL Errors**: Test completes without connection errors or timeouts
2. **Scaling Efficiency**: Meets minimum thresholds for cluster size
3. **Stability**: 3 consecutive runs with <5% variance in throughput
4. **GPU Utilization**: All GPUs show >90% utilization during active phase
5. **Temperature**: No thermal throttling (GPU temp <85°C for H100, <80°C for A100)

### Warning Criteria (Investigate But May Pass)

1. **Scaling Efficiency**: 5-10% below expected baseline
2. **Step Time Variance**: 5-10% variance between runs
3. **Single GPU Underperformance**: One GPU showing <90% utilization

### Fail Criteria (Must Fix Before Production)

1. **Scaling Efficiency**: >10% below expected baseline
2. **NCCL Errors**: Any connection or timeout errors
3. **Step Time Degradation**: Step time increases over time (>20%)
4. **Result Variance**: >10% variance between runs
5. **Hardware Errors**: ECC errors, XID errors, or PCIe errors in logs

### Test Matrix (Recommended)

| Test # | Model | Nodes | GPUs/Node | Duration | Purpose |
|--------|-------|-------|-----------|----------|---------|
| 1 | ResNet-50 | 1 | 1 | 5 min | Baseline single GPU |
| 2 | ResNet-50 | 1 | 8 | 10 min | NVLink validation |
| 3 | ResNet-50 | 4 | 8 | 15 min | Network validation |
| 4 | ResNet-50 | 8 | 8 | 20 min | Scale validation |
| 5 | Transformer | 1 | 8 | 15 min | High bandwidth test |
| 6 | Transformer | 4 | 8 | 20 min | Large model simulation |

### Stress Test (Optional)

For long-term stability testing:

```bash
# Run for 24 hours
--warmup-iterations 100 \
--active-iterations 10000
```

Monitor for:
- Memory leaks (increasing GPU memory over time)
- Thermal throttling
- Network degradation
- ECC errors

---

## Quick Reference: Decision Tree

```
Test Fails
├── Throughput < 50% expected?
│   ├── Yes → Check NCCL logs for network errors
│   │   ├── Timeout errors → Network connectivity issue
│   │   ├── Connection refused → Master not reachable
│   │   └── No errors → Check GPU utilization
│   └── No → Continue
├── Throughput 50-80% expected?
│   ├── Yes → Check scaling efficiency
│   │   ├── Single GPU slow → Check GPU clocks/power
│   │   └── Multi-GPU slow → Network bottleneck
│   └── No → Continue
├── Throughput 80-95% expected?
│   ├── Yes → Check for thermal throttling
│   │   ├── GPU temp > 85°C → Cooling issue
│   │   └── Temp normal → Minor tuning needed
│   └── No → Continue
└── Throughput >95% expected?
    └── PASS → Cluster ready for production
```

---

## Appendix: Log Analysis

### Healthy NCCL Logs

```
[Rank 0] NCCL INFO Bootstrap : Using eth0:192.168.1.100<0>
[Rank 0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB
[Rank 0] NCCL INFO Using network Socket
[Rank 0] NCCL INFO comm 0x7f8a4c000000 rank 0 nranks 32 cudaDev 0 busId 1000 - Init COMPLETE
```

### Problematic NCCL Logs

```
[Rank 1] NCCL WARN Connect to 192.168.1.100<12345> failed : Connection timed out
[Rank 1] NCCL WARN Connect to 192.168.1.100<12345> failed : Connection timed out
[Rank 1] NCCL WARN Connect to 192.168.1.100<12345> failed : Connection timed out
[Rank 1] NCCL ERROR Network connection timeout
```

**Action Required:** Network/firewall issue between nodes.

---

For additional support, contact the Nebius Infrastructure Engineering team.
