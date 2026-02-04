#!/bin/bash
#
# GPU Cluster Verification Script for Mixed Kubernetes Clusters
#
# This script helps verify that your Kubernetes cluster is properly configured
# for GPU workloads before running acceptance tests.
#

set -e

echo "========================================="
echo "Kubernetes GPU Cluster Verification"
echo "========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}✗ kubectl not found${NC}"
    echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi
echo -e "${GREEN}✓ kubectl is available${NC}"

# Check if jq is available (optional but helpful)
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠ jq not found (optional, but recommended for better output)${NC}"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    HAS_JQ=false
else
    echo -e "${GREEN}✓ jq is available${NC}"
    HAS_JQ=true
fi

echo ""
echo "========================================="
echo "1. Checking Cluster Nodes"
echo "========================================="

# Count total nodes
TOTAL_NODES=$(kubectl get nodes --no-headers | wc -l | tr -d ' ')
echo "Total nodes in cluster: $TOTAL_NODES"

# Check for GPU nodes
if [ "$HAS_JQ" = true ]; then
    GPU_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | .metadata.name')
    GPU_NODE_COUNT=$(echo "$GPU_NODES" | grep -c . || echo "0")
else
    GPU_NODES=$(kubectl get nodes -o wide | grep -i gpu || echo "")
    GPU_NODE_COUNT=$(echo "$GPU_NODES" | grep -c . || echo "0")
fi

if [ "$GPU_NODE_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ No GPU nodes found in cluster${NC}"
    echo ""
    echo "Your cluster doesn't have GPU nodes or they're not properly labeled."
    echo "Please ensure:"
    echo "  1. GPU nodes exist in your cluster"
    echo "  2. NVIDIA device plugin is installed"
    echo "  3. GPU capacity is properly advertised"
    exit 1
else
    echo -e "${GREEN}✓ Found $GPU_NODE_COUNT GPU node(s)${NC}"
    echo ""
    echo "GPU Nodes:"
    if [ "$HAS_JQ" = true ]; then
        kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | "  - \(.metadata.name): \(.status.capacity."nvidia.com/gpu") GPUs"'
    else
        echo "$GPU_NODES"
    fi
fi

NON_GPU_NODES=$((TOTAL_NODES - GPU_NODE_COUNT))
if [ "$NON_GPU_NODES" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ Found $NON_GPU_NODES non-GPU node(s) in cluster${NC}"
    echo "This is a mixed GPU/non-GPU cluster - GPU workloads will be scheduled only on GPU nodes."
fi

echo ""
echo "========================================="
echo "2. Checking NVIDIA Device Plugin"
echo "========================================="

# Check for NVIDIA device plugin
NVIDIA_PLUGIN=$(kubectl get pods -n kube-system -o wide | grep nvidia-device-plugin || echo "")
if [ -z "$NVIDIA_PLUGIN" ]; then
    echo -e "${RED}✗ NVIDIA device plugin not found${NC}"
    echo ""
    echo "The NVIDIA device plugin is required for GPU support."
    echo "Install it with:"
    echo "  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml"
    exit 1
else
    PLUGIN_COUNT=$(echo "$NVIDIA_PLUGIN" | wc -l | tr -d ' ')
    echo -e "${GREEN}✓ NVIDIA device plugin is running ($PLUGIN_COUNT instance(s))${NC}"
    echo ""
    kubectl get pods -n kube-system | grep nvidia-device-plugin
fi

echo ""
echo "========================================="
echo "3. Checking GPU Node Labels"
echo "========================================="

echo "Common GPU-related labels found on nodes:"
kubectl get nodes -o json | jq -r '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | "Node: \(.metadata.name)\n  Labels:" + (.metadata.labels | to_entries | map("    \(.key)=\(.value)") | join("\n"))' 2>/dev/null || {
    echo "Using basic label check..."
    for node in $GPU_NODES; do
        echo "Node: $node"
        kubectl get node "$node" --show-labels | grep -o 'accelerator=[^,]*\|gpu=[^,]*\|nvidia.com/gpu=[^,]*' || echo "  No standard GPU labels found"
    done
}

echo ""
echo "========================================="
echo "4. Checking GPU Node Taints"
echo "========================================="

TAINTED_GPU_NODES=""
for node in $GPU_NODES; do
    TAINTS=$(kubectl get node "$node" -o jsonpath='{.spec.taints[*].key}' 2>/dev/null || echo "")
    if [ -n "$TAINTS" ]; then
        echo -e "${YELLOW}⚠ Node $node has taints: $TAINTS${NC}"
        TAINTED_GPU_NODES="$TAINTED_GPU_NODES $node"
    fi
done

if [ -z "$TAINTED_GPU_NODES" ]; then
    echo -e "${GREEN}✓ No GPU nodes are tainted${NC}"
    echo "Your PyTorchJob does not need tolerations."
else
    echo ""
    echo "Your GPU nodes are tainted. You MUST add tolerations to your PyTorchJob:"
    echo "  tolerations:"
    echo "  - key: nvidia.com/gpu"
    echo "    operator: Exists"
    echo "    effect: NoSchedule"
fi

echo ""
echo "========================================="
echo "5. Testing GPU Resource Allocation"
echo "========================================="

echo "Creating a test pod to verify GPU scheduling..."

# Create a test pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-pod
spec:
  restartPolicy: Never
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Wait for pod to complete
echo "Waiting for test pod to complete..."
kubectl wait --for=condition=Ready pod/gpu-test-pod --timeout=60s 2>/dev/null || {
    echo -e "${YELLOW}⚠ Pod did not become ready (may complete before ready)${NC}"
}

sleep 5

# Check pod status
POD_STATUS=$(kubectl get pod gpu-test-pod -o jsonpath='{.status.phase}')
POD_NODE=$(kubectl get pod gpu-test-pod -o jsonpath='{.spec.nodeName}')

if [ "$POD_STATUS" == "Succeeded" ] || kubectl logs gpu-test-pod 2>/dev/null | grep -q "NVIDIA-SMI"; then
    echo -e "${GREEN}✓ Test pod successfully scheduled on GPU node: $POD_NODE${NC}"
    echo ""
    echo "GPU detected by test pod:"
    kubectl logs gpu-test-pod 2>/dev/null | head -n 20 || echo "Could not retrieve logs"
else
    echo -e "${RED}✗ Test pod failed or could not access GPU${NC}"
    echo "Pod status: $POD_STATUS"
    echo "Pod node: $POD_NODE"
    echo ""
    echo "Pod events:"
    kubectl describe pod gpu-test-pod | tail -n 20
fi

# Cleanup
echo ""
echo "Cleaning up test pod..."
kubectl delete pod gpu-test-pod --ignore-not-found=true

echo ""
echo "========================================="
echo "6. Checking Network Configuration"
echo "========================================="

# Check for InfiniBand devices on GPU nodes
echo "Checking for InfiniBand/RDMA support on GPU nodes..."
HAS_IB=false

for node in $GPU_NODES; do
    echo ""
    echo "Node: $node"
    
    # Create a debug pod to check network devices
    cat <<EOF | kubectl apply -f - > /dev/null 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: network-check-$node
spec:
  nodeName: $node
  hostNetwork: true
  containers:
  - name: network-check
    image: ubuntu:22.04
    command: ["sleep", "30"]
    securityContext:
      privileged: true
EOF
    
    # Wait for pod to be ready
    kubectl wait --for=condition=Ready pod/network-check-$node --timeout=30s > /dev/null 2>&1 || {
        echo -e "${YELLOW}  ⚠ Could not create network check pod${NC}"
        kubectl delete pod network-check-$node --ignore-not-found=true > /dev/null 2>&1
        continue
    }
    
    # Check for InfiniBand devices
    IB_DEVICES=$(kubectl exec network-check-$node -- sh -c "ls -la /sys/class/infiniband/ 2>/dev/null | grep -v total | wc -l" 2>/dev/null || echo "0")
    
    if [ "$IB_DEVICES" -gt 0 ]; then
        echo -e "${GREEN}  ✓ InfiniBand devices found: $IB_DEVICES adapter(s)${NC}"
        HAS_IB=true
        
        # Get IB device names
        IB_NAMES=$(kubectl exec network-check-$node -- sh -c "ls /sys/class/infiniband/" 2>/dev/null || echo "")
        if [ -n "$IB_NAMES" ]; then
            echo "    Adapters: $IB_NAMES"
        fi
        
        # Check if RDMA modules are loaded
        RDMA_LOADED=$(kubectl exec network-check-$node -- sh -c "lsmod 2>/dev/null | grep -E 'rdma|ib_' | wc -l" 2>/dev/null || echo "0")
        if [ "$RDMA_LOADED" -gt 0 ]; then
            echo -e "${GREEN}    ✓ RDMA kernel modules loaded${NC}"
        else
            echo -e "${YELLOW}    ⚠ RDMA kernel modules not detected${NC}"
        fi
    else
        echo -e "${YELLOW}  ⚠ No InfiniBand devices found${NC}"
        echo "    Using Ethernet for NCCL communication"
    fi
    
    # Check network interfaces
    echo "  Network interfaces:"
    kubectl exec network-check-$node -- sh -c "ip -br link show 2>/dev/null | grep -E 'UP|UNKNOWN' | head -5" 2>/dev/null || echo "    Could not retrieve interface list"
    
    # Cleanup
    kubectl delete pod network-check-$node --ignore-not-found=true > /dev/null 2>&1
done

echo ""
if [ "$HAS_IB" = true ]; then
    echo -e "${GREEN}✓ High-speed interconnect (InfiniBand) detected${NC}"
    echo ""
    echo "NCCL Recommendations for InfiniBand:"
    echo "  Add these environment variables to your PyTorchJob containers:"
    echo "    - NCCL_IB_DISABLE=0"
    echo "    - NCCL_IB_HCA=mlx5_0,mlx5_1  # Adjust based on your adapters"
    echo "    - NCCL_IB_GID_INDEX=3"
    echo "    - NCCL_IB_TC=136"
else
    echo -e "${YELLOW}⚠ No InfiniBand detected - using Ethernet${NC}"
    echo ""
    echo "NCCL Recommendations for Ethernet:"
    echo "  Add these environment variables to your PyTorchJob containers:"
    echo "    - NCCL_SOCKET_IFNAME=eth0  # Or your high-speed interface"
    echo "    - NCCL_IB_DISABLE=1        # Disable IB if not available"
fi

echo ""
echo "========================================="
echo "7. Checking NCCL Configuration"
echo "========================================="

# Check if any existing pods have NCCL environment variables set
EXISTING_PODS=$(kubectl get pods -A -o json | jq -r '.items[] | select(.spec.containers[].image | contains("gpu") or contains("pytorch") or contains("cuda")) | .metadata.namespace + "/" + .metadata.name' 2>/dev/null | head -5)

if [ -n "$EXISTING_PODS" ]; then
    echo "Checking NCCL configuration in existing GPU pods:"
    echo ""
    
    for pod in $EXISTING_PODS; do
        NAMESPACE=$(echo $pod | cut -d'/' -f1)
        POD_NAME=$(echo $pod | cut -d'/' -f2)
        
        echo "Pod: $pod"
        
        # Check for NCCL environment variables
        NCCL_VARS=$(kubectl get pod -n $NAMESPACE $POD_NAME -o json 2>/dev/null | jq -r '.spec.containers[].env[]? | select(.name | startswith("NCCL")) | "  " + .name + "=" + .value' 2>/dev/null)
        
        if [ -n "$NCCL_VARS" ]; then
            echo -e "${GREEN}  ✓ NCCL environment variables configured:${NC}"
            echo "$NCCL_VARS"
        else
            echo -e "${YELLOW}  ⚠ No NCCL environment variables found${NC}"
            echo "    Consider adding NCCL_DEBUG=INFO for troubleshooting"
        fi
        echo ""
    done
else
    echo "No existing GPU pods found to check NCCL configuration."
    echo ""
    echo "Recommended NCCL environment variables for your PyTorchJob:"
    echo "  env:"
    echo "    - name: NCCL_DEBUG"
    echo "      value: \"INFO\""
    echo "    - name: NCCL_DEBUG_SUBSYS"
    echo "      value: \"ALL\""
    if [ "$HAS_IB" = true ]; then
        echo "    - name: NCCL_IB_DISABLE"
        echo "      value: \"0\""
        echo "    - name: NCCL_IB_HCA"
        echo "      value: \"mlx5_0\"  # Check your adapter name"
    else
        echo "    - name: NCCL_SOCKET_IFNAME"
        echo "      value: \"eth0\"  # Or your primary interface"
    fi
fi

echo ""
echo "========================================="
echo "Summary & Recommendations"
echo "========================================="
echo ""

if [ "$GPU_NODE_COUNT" -gt 0 ] && [ -n "$NVIDIA_PLUGIN" ]; then
    echo -e "${GREEN}✓ Your cluster is ready for GPU workloads!${NC}"
    echo ""
    echo "Deployment checklist:"
    echo "  ✓ GPU nodes are available ($GPU_NODE_COUNT node(s))"
    echo "  ✓ NVIDIA device plugin is running"
    
    if [ -n "$TAINTED_GPU_NODES" ]; then
        echo "  ⚠ Add tolerations to your PyTorchJob (GPU nodes are tainted)"
    else
        echo "  ✓ No tolerations needed (GPU nodes not tainted)"
    fi
    
    if [ "$NON_GPU_NODES" -gt 0 ]; then
        echo "  ⚠ Mixed cluster detected - ensure resource requests include nvidia.com/gpu"
    else
        echo "  ✓ Pure GPU cluster - all nodes have GPUs"
    fi
    
    if [ "$HAS_IB" = true ]; then
        echo "  ✓ InfiniBand detected - configure NCCL for IB (see recommendations above)"
    else
        echo "  ⚠ Using Ethernet - configure NCCL_SOCKET_IFNAME (see recommendations above)"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Review the example: examples/kubernetes-mixed-cluster.yaml"
    echo "  2. Update NCCL environment variables based on your network setup"
    echo "  3. Deploy your acceptance test:"
    echo "       kubectl apply -f pytorchjob.yaml"
    echo "  4. Monitor the test:"
    echo "       kubectl logs -f pytorch-job-master-0"
    echo "  5. Check NCCL logs for network performance:"
    echo "       kubectl logs pytorch-job-master-0 | grep NCCL"
else
    echo -e "${RED}✗ Your cluster is NOT ready for GPU workloads${NC}"
    echo ""
    echo "Please address the issues identified above before deploying GPU workloads."
fi

echo ""
echo "========================================="
