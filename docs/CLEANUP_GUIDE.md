# Test Cleanup Guide

## Overview

By default, test resources remain in the cluster after completion to allow for log inspection and debugging. This guide explains the cleanup behavior for each platform and how to configure automatic cleanup.

---

## Cleanup Behavior by Platform

### 🐳 Docker (Bare Metal)

**Default Behavior**: Containers stop but remain unless `--rm` flag is used.

**Current Examples**: ✅ All examples use `--rm` flag for automatic cleanup

```bash
# Automatic cleanup (recommended - already in all examples)
docker run --gpus all --rm \
  cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing:latest
```

**Manual Cleanup** (if you omit `--rm`):
```bash
# List stopped containers
docker ps -a --filter "ancestor=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing"

# Remove specific container
docker rm <container-id>

# Remove all stopped containers
docker container prune
```

---

### ☸️ Kubernetes (PyTorchJob)

**Default Behavior**: ⚠️ Pods and PyTorchJob remain after completion

**Current Examples**: Jobs remain for log inspection

#### Automatic Cleanup Options

**Option 1: TTL Controller (Recommended)**

Add TTL to your PyTorchJob:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: gpu-cluster-acceptance-test
spec:
  # Automatically delete 1 hour after completion
  ttlSecondsAfterFinished: 3600
  
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      # ... rest of spec
```

**Option 2: Job Lifecycle Policy**

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: gpu-cluster-acceptance-test
spec:
  # Clean up immediately after success, keep failures for debugging
  runPolicy:
    cleanPodPolicy: Running  # Options: All, Running, None
  
  pytorchReplicaSpecs:
    # ... spec
```

**Option 3: Manual Cleanup**

```bash
# View completed jobs
kubectl get pytorchjobs

# Delete specific job (and its pods)
kubectl delete pytorchjob gpu-cluster-acceptance-test

# View pods
kubectl get pods -l app=gpu-acceptance-test

# Delete all completed test pods
kubectl delete pods -l app=gpu-acceptance-test --field-selector=status.phase=Succeeded

# Clean up all completed pods in namespace
kubectl delete pods --field-selector=status.phase=Succeeded
kubectl delete pods --field-selector=status.phase=Failed  # Optional
```

**Option 4: Automated Cleanup Script**

```bash
#!/bin/bash
# cleanup-k8s-tests.sh

NAMESPACE=${1:-default}
AGE_THRESHOLD=${2:-1h}  # Delete jobs older than 1 hour

echo "Cleaning up PyTorchJobs older than ${AGE_THRESHOLD} in namespace ${NAMESPACE}"

# Get jobs older than threshold
kubectl get pytorchjobs -n $NAMESPACE -o json | \
  jq -r ".items[] | select(.status.completionTime != null) | 
    select((now - (.status.completionTime | fromdateiso8601)) > (\"${AGE_THRESHOLD}\" | sub(\"h\"; \"\") | tonumber * 3600)) | 
    .metadata.name" | \
  xargs -I {} kubectl delete pytorchjob {} -n $NAMESPACE

echo "Cleanup complete"
```

---

### 🖥️ Slurm

**Default Behavior**: Jobs terminate, logs remain in `results/` directory

**Current Examples**: Logs persist for later analysis

```bash
# Job terminates automatically after completion
sbatch examples/slurm-nccl-test.sh

# Logs remain at:
results/nccl_bandwidth_<jobid>.out
```

#### Manual Cleanup

```bash
# View job history
sacct -S today

# Cancel running jobs
scancel <jobid>

# Clean up old log files
find results/ -name "*.out" -mtime +7 -delete  # Delete logs older than 7 days
```

#### Automatic Log Cleanup (Optional)

Add to your sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=nccl-test

# ... test commands ...

# Clean up old logs at the end
find results/ -name "*.out" -mtime +7 -delete
```

---

## Recommended Cleanup Strategies

### For Development/Testing

**Keep resources** for debugging:
- Kubernetes: Don't set TTL, manually delete after inspection
- Docker: Omit `--rm` flag if you need to inspect container
- Slurm: Keep logs indefinitely

### For CI/CD Pipelines

**Automatic cleanup** to prevent resource accumulation:

**Kubernetes**:
```yaml
spec:
  ttlSecondsAfterFinished: 600  # Delete after 10 minutes
  runPolicy:
    cleanPodPolicy: Running  # Delete running pods, keep failed for debugging
```

**Docker** (already configured):
```bash
docker run --rm ...  # Auto-cleanup on exit
```

**Slurm**:
```bash
# Add to end of sbatch script
trap "find results/ -name '*_${SLURM_JOB_ID}.out' -delete" EXIT
```

### For Production Acceptance Testing

**Balanced approach**:

**Kubernetes**:
```yaml
spec:
  ttlSecondsAfterFinished: 86400  # Keep for 24 hours
  runPolicy:
    cleanPodPolicy: None  # Keep all pods for forensics
```

**Slurm**:
```bash
# Keep logs for 30 days
find results/ -name "*.out" -mtime +30 -delete
```

---

## Storage Considerations

### Disk Usage

**Logs per test**:
- NCCL tests: ~1-5 MB
- Training tests: ~10-50 MB

**Kubernetes pod logs**:
- Check with: `kubectl get pods -o json | jq '[.items[] | {name: .metadata.name, logs: .status.containerStatuses[].state}]'`

**Slurm results directory**:
```bash
# Check disk usage
du -sh results/

# Count log files
ls -1 results/*.out | wc -l
```

### Cleanup Thresholds

**Recommended cleanup policies**:

| Environment | Successful Jobs | Failed Jobs | Rationale |
|-------------|----------------|-------------|-----------|
| **Development** | Keep 1 day | Keep 7 days | Debug recent issues |
| **CI/CD** | Delete immediately | Keep 1 day | Fast iteration |
| **Production** | Keep 7 days | Keep 30 days | Compliance, auditing |

---

## Automated Cleanup Examples

### Kubernetes CronJob for Cleanup

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-gpu-tests
spec:
  schedule: "0 2 * * *"  # Run daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: cleanup-sa  # Needs permissions
          containers:
          - name: cleanup
            image: bitnami/kubectl:latest
            command:
            - /bin/bash
            - -c
            - |
              # Delete completed PyTorchJobs older than 24 hours
              kubectl get pytorchjobs -o json | \
                jq -r '.items[] | select(.status.completionTime != null) | 
                  select((now - (.status.completionTime | fromdateiso8601)) > 86400) | 
                  .metadata.name' | \
                xargs -r kubectl delete pytorchjob
              
              # Delete completed pods older than 24 hours
              kubectl delete pods --field-selector=status.phase=Succeeded \
                --all-namespaces --ignore-not-found
          restartPolicy: OnFailure
```

### Slurm Cleanup Script

```bash
#!/bin/bash
# /usr/local/bin/cleanup-slurm-tests.sh
# Add to cron: 0 2 * * * /usr/local/bin/cleanup-slurm-tests.sh

RESULTS_DIR="/opt/slurm-test/quickcheck/results"
RETENTION_DAYS=7

echo "$(date): Cleaning up test results older than ${RETENTION_DAYS} days"

# Delete old NCCL test logs
find "${RESULTS_DIR}" -name "nccl_*.out" -mtime +${RETENTION_DAYS} -delete

# Delete old training test logs
find "${RESULTS_DIR}" -name "training_*.out" -mtime +${RETENTION_DAYS} -delete

# Report disk usage
du -sh "${RESULTS_DIR}"
```

Add to crontab:
```bash
0 2 * * * /usr/local/bin/cleanup-slurm-tests.sh >> /var/log/cleanup-tests.log 2>&1
```

---

## Best Practices

### ✅ DO

1. **Set TTL for Kubernetes jobs** in production (24-48 hours)
2. **Use `--rm` for Docker** in examples and CI/CD
3. **Implement log rotation** for Slurm results
4. **Keep failed jobs longer** than successful ones for debugging
5. **Monitor disk usage** in results directories
6. **Document cleanup policy** in your runbooks

### ❌ DON'T

1. **Delete immediately** without log retention policy
2. **Forget to clean up** in CI/CD pipelines
3. **Use same retention** for dev and production
4. **Ignore disk usage alerts**
5. **Delete logs** without backup in production

---

## Quick Cleanup Commands

### Kubernetes

```bash
# Delete all completed test jobs
kubectl delete pytorchjob -l app=gpu-acceptance-test

# Delete all test pods
kubectl delete pods -l app=gpu-acceptance-test

# Clean up all completed pods (any workload)
kubectl delete pods --all-namespaces --field-selector=status.phase=Succeeded

# Clean up all failed pods older than 1 day
kubectl get pods --all-namespaces --field-selector=status.phase=Failed \
  -o json | jq -r '.items[] | 
    select((now - (.status.startTime | fromdateiso8601)) > 86400) | 
    "\(.metadata.namespace) \(.metadata.name)"' | \
  xargs -n2 kubectl delete pod -n
```

### Docker

```bash
# Remove all stopped test containers
docker ps -a --filter "ancestor=cr.eu-north1.nebius.cloud/e00tnz9wpyxva2s992/gpu_cluster_testing" -q | xargs docker rm

# Remove all stopped containers (any image)
docker container prune -f

# Remove old images
docker image prune -a --filter "until=24h"
```

### Slurm

```bash
# Cancel all queued/running test jobs
squeue -u $USER -n gpu-test -h -o "%i" | xargs scancel

# Clean up logs older than 7 days
find results/ -name "*.out" -mtime +7 -delete

# Archive old logs
tar -czf results-archive-$(date +%Y%m%d).tar.gz results/*.out
mv results-archive-*.tar.gz /path/to/archive/
```

---

## Monitoring Cleanup

### Check Resource Usage

**Kubernetes**:
```bash
# Count test jobs by status
kubectl get pytorchjobs -o json | jq '[.items[] | .status.conditions[0].type] | group_by(.) | map({status: .[0], count: length})'

# Total disk used by logs
kubectl get pods -l app=gpu-acceptance-test -o json | \
  jq -r '.items[] | .metadata.name' | \
  xargs -I {} sh -c 'kubectl logs {} | wc -c' | \
  awk '{sum+=$1} END {print "Total: " sum/1024/1024 " MB"}'
```

**Docker**:
```bash
# List containers with disk usage
docker ps -a --format "table {{.Names}}\t{{.Size}}\t{{.Status}}"

# Total size of test images
docker images --filter=reference='*gpu_cluster_testing*' --format "{{.Size}}"
```

**Slurm**:
```bash
# Disk usage of results
du -sh results/

# Count log files by age
find results/ -name "*.out" -mtime -1 | wc -l  # < 1 day
find results/ -name "*.out" -mtime +7 | wc -l  # > 7 days
```

---

## Summary

### Current State

| Platform | Auto-Cleanup? | Recommendation |
|----------|---------------|----------------|
| Docker | ✅ Yes (`--rm` in all examples) | No changes needed |
| Kubernetes | ❌ No | Add TTL for production |
| Slurm | ⚠️ Partial (jobs end, logs stay) | Add log rotation |

### Action Items

1. **Kubernetes users**: Add `ttlSecondsAfterFinished` to PyTorchJob specs
2. **Slurm users**: Implement log rotation script
3. **All users**: Monitor disk usage in results directories
4. **CI/CD**: Ensure automatic cleanup is configured

### Files Updated

- [examples/kubernetes-mixed-cluster-with-ttl.yaml](../examples/kubernetes-mixed-cluster-with-ttl.yaml) - NEW: Example with automatic cleanup
- [scripts/cleanup-k8s-tests.sh](../scripts/cleanup-k8s-tests.sh) - NEW: Automated K8s cleanup
- [scripts/cleanup-slurm-logs.sh](../scripts/cleanup-slurm-logs.sh) - NEW: Automated Slurm log cleanup

See the examples directory for reference implementations.
