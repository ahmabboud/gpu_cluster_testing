# Troubleshooting Guide

## Common Issues and Solutions

### PyTorch Import Failures

#### Symptom
```
ERROR: PyTorch not found
ImportError: /opt/hpcx/ucc/lib/libucc.so.1: undefined symbol: ucs_config_doc_nop
```

#### Cause
UCX/UCC library conflict between container's HPC-X libraries and system libraries on the host.

#### Solution
**Already Fixed in v1.0+**: The entrypoint automatically prepends HPC-X library paths:
```bash
export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}"
```

If you're building custom images based on this tool, ensure this is set **before** any Python imports.

#### Manual Workaround (if needed)
Add environment variable to your pod spec:
```yaml
env:
- name: LD_LIBRARY_PATH
  value: "/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib"
```

---

### NCCL Communication Failures

#### Symptom
```
NCCL WARN Connect to <host> failed : Connection refused
NCCL INFO Bootstrap : Using eth0:<ip>
```

#### Cause
Incorrect network interface detection or firewall blocking NCCL ports.

#### Solution

**For InfiniBand clusters**, add to pod spec:
```yaml
env:
- name: NCCL_IB_DISABLE
  value: "0"
- name: NCCL_IB_HCA
  value: "mlx5_0"
- name: NCCL_DEBUG
  value: "INFO"
```

**For Ethernet clusters**, add to pod spec:
```yaml
env:
- name: NCCL_SOCKET_IFNAME
  value: "eth0"
- name: NCCL_DEBUG
  value: "INFO"
```

**Debug network interface detection:**
```bash
kubectl exec <pod-name> -- ip addr show
kubectl exec <pod-name> -- ibstat  # For InfiniBand
```

---

### Out of Memory (OOM) Errors

#### Symptom
```
RuntimeError: CUDA out of memory
```

#### Solution
Reduce batch size in your pod spec args:
```yaml
args:
  - "--batch-size"
  - "64"  # Reduce from default
```

Or use a smaller model:
```yaml
args:
  - "--model"
  - "resnet18"  # 11M params instead of ResNet50's 25M params
```

---

### Shared Memory Issues

#### Symptom
```
ERROR: Unexpected bus error encountered in worker
```

#### Cause
Insufficient shared memory for DataLoader workers with real datasets.

#### Solution
Add shared memory volume to your pod spec (already included in provided examples):
```yaml
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: "8Gi"
volumeMounts:
- name: dshm
  mountPath: /dev/shm
```

---

### Multi-GPU Training Not Using All GPUs

#### Symptom
Only 1 GPU showing activity when multiple GPUs allocated.

#### Cause
DDP not initialized properly or CUDA_VISIBLE_DEVICES not set correctly.

#### Solution
The tool automatically detects available GPUs via `torch.cuda.device_count()`.

**Verify GPU allocation:**
```bash
kubectl exec <pod-name> -- nvidia-smi
```

**Check CUDA_VISIBLE_DEVICES:**
```bash
kubectl exec <pod-name> -- bash -c 'echo $CUDA_VISIBLE_DEVICES'
```

---

### Platform Architecture Mismatch

#### Symptom
```
no matching manifest for linux/arm64/v8
```

#### Cause
Trying to pull AMD64 image on ARM64 system (e.g., Apple Silicon Mac) or misconfigured Kubernetes node.

#### Solution
The container image is built for AMD64 GPU servers. Ensure your Kubernetes GPU nodes are AMD64 architecture.

**When building for your registry:**
```bash
docker build --platform linux/amd64 -t your-registry/gpu_cluster_testing:latest .
```

Note: ARM64 is not supported as NVIDIA data center GPUs require AMD64.

---

## Debugging Steps

### 1. Check Pod/Container Status
```bash
kubectl get pod <pod-name>
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### 2. Interactive Debugging
```bash
kubectl run debug --image=ghcr.io/ahmabboud/gpu_cluster_testing:latest \
  --restart=Never --rm -it -- bash
```

Inside the pod:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

### 3. Check NCCL Configuration
```bash
kubectl exec <pod-name> -- bash -c 'env | grep NCCL'
```

### 4. Verify InfiniBand/RDMA
```bash
kubectl exec <pod-name> -- ibstat
kubectl exec <pod-name> -- ls -la /dev/infiniband/
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check NCCL logs**: Set `NCCL_DEBUG=INFO` for detailed output
2. **Verify cluster health**: Run single-GPU test first
3. **Review examples**: See [examples/](../examples/) directory
4. **Contact**: Open an issue on GitHub with:
   - Error message
   - Pod/container logs
   - Cluster configuration (GPU count, network type)
   - Command used to launch the test
