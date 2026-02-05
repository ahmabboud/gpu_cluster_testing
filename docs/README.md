# Documentation Index

Welcome to the GPU Cluster Acceptance Testing Tool documentation.

## 🚀 Quick Start

**New to this tool?** Start here:
1. Read the main [README.md](../README.md) - Quick start commands
2. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md) - Architecture and data flow
3. Try the examples in `../examples/`

---

## 📚 Documentation

### Core Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | Architecture, execution flow, why synthetic data works | Developers |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues: UCX/UCC, NCCL, OOM, platform errors | All users |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Repository layout and file inventory | All users |

### Testing Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| [NCCL_TESTING.md](NCCL_TESTING.md) | NCCL bandwidth/latency testing, interpreting results | Infra engineers |
| [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) | Decision trees, recommended test sequence | Test operators |
| [INFINIBAND_CONFIGURATION.md](INFINIBAND_CONFIGURATION.md) | NCCL/InfiniBand setup, multi-node config | Infra engineers |

### Operations

| Document | Purpose | Audience |
|----------|---------|----------|
| [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md) | Resource cleanup, TTL config, manual cleanup | Ops teams |

---

## 🎯 Common Tasks

### "I want to validate a new GPU cluster"
1. [Main README](../README.md) - Quick Start commands
2. [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) - Which tests to run

### "Network seems slow"
1. [NCCL_TESTING.md](NCCL_TESTING.md) - Run bandwidth tests
2. [INFINIBAND_CONFIGURATION.md](INFINIBAND_CONFIGURATION.md) - Check IB config

### "How do I clean up test resources?"
→ [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md)

### "Should I use NCCL tests or training tests?"
→ [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) - Decision Tree

### "What performance should I expect?"
→ [Main README](../README.md) - Expected Performance section

---

## 🔍 Search by Keyword

| Topic | Document |
|-------|----------|
| Performance benchmarks | Main README, HOW_IT_WORKS.md |
| NCCL configuration | NCCL_TESTING.md, INFINIBAND_CONFIGURATION.md |
| InfiniBand setup | INFINIBAND_CONFIGURATION.md |
| Kubernetes examples | Main README, examples/ directory |
| Network debugging | NCCL_TESTING.md, TROUBLESHOOTING.md |
| Container build | Main README - Building from Source |

---

## 📦 Example Files

All examples are in the `examples/` directory:

| File | Purpose | GPUs |
|------|---------|------|
| `kubernetes-pod-single-gpu.yaml` | Single GPU smoke test | 1 |
| `kubernetes-pod-multi-gpu-single-node.yaml` | Multi-GPU DDP test | 2+ |
| `kubernetes-statefulset-multi-node-ddp.yaml` | Multi-node InfiniBand DDP | 2+ nodes |
| `kubernetes-multi-gpu-nebius-optimized.yaml` | Nebius-optimized multi-GPU | 8 |
| `kubernetes-flexible-nebius-pattern.yaml` | Flexible Nebius deployment | Variable |
| `kubernetes-with-auto-cleanup.yaml` | Auto-cleanup with TTL | - |
| `kubernetes-mixed-cluster.yaml` | Mixed GPU/non-GPU clusters | - |

---

## 🚀 Quick Commands

```bash
# Run all tests (recommended)
./scripts/run-all-tests.sh

# Or run individual tests
kubectl apply -f examples/kubernetes-pod-single-gpu.yaml
kubectl logs -f pod/gpu-cluster-test-single-gpu

# Multi-GPU test
kubectl apply -f examples/kubernetes-pod-multi-gpu-single-node.yaml
kubectl logs -f pod/gpu-cluster-test-multi-gpu-single-node

# Multi-node DDP test
kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml
kubectl logs -f pod/gpu-cluster-test-ddp-0
```

---

**Repository**: `ghcr.io/ahmabboud/gpu_cluster_testing:latest`
