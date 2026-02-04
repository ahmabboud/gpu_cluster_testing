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
```bash
docker run --gpus all --rm \
  -e LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH}" \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest
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

**For InfiniBand clusters:**
```bash
docker run --gpus all --rm --network=host \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=mlx5_0 \
  -e NCCL_DEBUG=INFO \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest
```

**For Ethernet clusters:**
```bash
docker run --gpus all --rm --network=host \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e NCCL_DEBUG=INFO \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest
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
Reduce batch size:
```bash
docker run --gpus all --rm \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest \
  --batch-size 64  # Reduce from default 128
```

Or use a smaller model:
```bash
# ResNet18 (11M params) instead of ResNet50 (25M params)
--model resnet18
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
**Docker:**
```bash
docker run --gpus all --rm --ipc=host \
  ghcr.io/ahmabboud/gpu_cluster_testing:latest
```

**Kubernetes:**
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
Trying to pull AMD64 image on ARM64 system (e.g., Apple Silicon Mac).

#### Solution
**For local testing on ARM Mac:**
```bash
docker pull --platform linux/amd64 ghcr.io/ahmabboud/gpu_cluster_testing:latest
```

**For building locally:**
```bash
docker build --platform linux/amd64 -t gpu_cluster_testing:local .
```

Note: The image is built for AMD64 GPU servers. ARM64 is not supported as NVIDIA GPUs require AMD64.

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
