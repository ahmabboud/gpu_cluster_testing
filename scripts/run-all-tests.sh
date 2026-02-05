#!/usr/bin/env bash
#
# GPU Cluster Acceptance Test Runner
#
# Runs all validation tests in sequence and reports results.
#
# Usage:
#   ./scripts/run-all-tests.sh [--skip-multinode] [--timeout 300] [--test N]
#

set -e

# Ensure bash 4+ for associative arrays (macOS ships with bash 3)
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    # Fallback: use simple arrays instead
    USE_SIMPLE_ARRAYS=true
else
    USE_SIMPLE_ARRAYS=false
fi

# Configuration
TIMEOUT=${TIMEOUT:-300}  # 5 minutes default per test
SKIP_MULTINODE=false
NAMESPACE=${NAMESPACE:-default}
RUN_TEST=""  # Empty = run all tests

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-multinode)
            SKIP_MULTINODE=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --test)
            RUN_TEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Results tracking (simple arrays for bash 3 compatibility)
RESULT_NAMES=()
RESULT_STATUS=()
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}       GPU Cluster Acceptance Test Suite${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Namespace: $NAMESPACE"
echo "Timeout per test: ${TIMEOUT}s"
echo "Skip multi-node: $SKIP_MULTINODE"
echo ""

# ═══════════════════════════════════════════════════════════════
# INITIAL CLEANUP - Remove any leftover resources from previous runs
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}Cleaning up any previous test resources...${NC}"
kubectl delete pod gpu-cluster-test-single-gpu -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete pod gpu-cluster-test-multi-gpu-single-node -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete statefulset gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete service gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete pod -l app=gpu-cluster-test -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
sleep 3
echo ""

# Helper: Wait for pod to complete
wait_for_pod() {
    local pod_name=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    echo "  Waiting for pod/$pod_name to complete (timeout: ${timeout}s)..."
    
    while true; do
        local status=$(kubectl get pod "$pod_name" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
        local elapsed=$(($(date +%s) - start_time))
        
        case $status in
            Succeeded)
                echo -e "  ${GREEN}✓ Pod completed successfully${NC}"
                return 0
                ;;
            Failed)
                echo -e "  ${RED}✗ Pod failed${NC}"
                return 1
                ;;
            NotFound)
                if [ $elapsed -gt 30 ]; then
                    echo -e "  ${RED}✗ Pod not found${NC}"
                    return 1
                fi
                ;;
            Running|Pending)
                if [ $elapsed -gt $timeout ]; then
                    echo -e "  ${YELLOW}⚠ Timeout waiting for pod${NC}"
                    return 1
                fi
                ;;
        esac
        sleep 5
    done
}

# Helper: Run a test
run_test() {
    local test_name=$1
    local yaml_file=$2
    local pod_name=$3
    local cleanup_cmd=$4
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Test $TESTS_RUN: $test_name${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    
    # Cleanup any existing resources
    eval "$cleanup_cmd" 2>/dev/null || true
    sleep 2
    
    # Deploy
    echo "  Deploying: $yaml_file"
    if ! kubectl apply -f "$yaml_file" -n "$NAMESPACE"; then
        echo -e "  ${RED}✗ Failed to deploy${NC}"
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("FAILED (deploy)")
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
    
    # Wait for completion
    if wait_for_pod "$pod_name" "$TIMEOUT"; then
        # Get logs
        echo ""
        echo "  Output (last 20 lines):"
        echo "  ─────────────────────────"
        kubectl logs "$pod_name" -n "$NAMESPACE" --tail=20 2>/dev/null | sed 's/^/  /'
        
        # Check for throughput in logs
        local throughput=$(kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null | grep -i "throughput" | tail -1)
        if [ -n "$throughput" ]; then
            echo ""
            echo -e "  ${GREEN}Result: $throughput${NC}"
        fi
        
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("PASSED")
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo ""
        echo "  Last logs:"
        kubectl logs "$pod_name" -n "$NAMESPACE" --tail=10 2>/dev/null | sed 's/^/  /' || echo "  (no logs)"
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("FAILED")
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Cleanup
    echo ""
    echo "  Cleaning up..."
    eval "$cleanup_cmd" 2>/dev/null || true
}

# Helper: Run inline test (heredoc)
run_inline_test() {
    local test_name=$1
    local pod_name=$2
    local yaml_content=$3
    
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Test $TESTS_RUN: $test_name${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    
    # Cleanup
    kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
    sleep 2
    
    # Deploy
    echo "  Deploying inline pod..."
    if ! echo "$yaml_content" | kubectl apply -f - -n "$NAMESPACE"; then
        echo -e "  ${RED}✗ Failed to deploy${NC}"
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("FAILED (deploy)")
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
    
    # Wait
    if wait_for_pod "$pod_name" "$TIMEOUT"; then
        echo ""
        echo "  Output:"
        echo "  ─────────────────────────"
        kubectl logs "$pod_name" -n "$NAMESPACE" 2>/dev/null | sed 's/^/  /'
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("PASSED")
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        RESULT_NAMES+=("$test_name")
        RESULT_STATUS+=("FAILED")
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Cleanup
    kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
}

# ═══════════════════════════════════════════════════════════════
# TEST 1: Single GPU Test
# ═══════════════════════════════════════════════════════════════

if [ -z "$RUN_TEST" ] || [ "$RUN_TEST" = "1" ]; then
    run_test "Single GPU Training" \
        "examples/kubernetes-pod-single-gpu.yaml" \
        "gpu-cluster-test-single-gpu" \
        "kubectl delete pod gpu-cluster-test-single-gpu -n $NAMESPACE --ignore-not-found=true"
fi

# ═══════════════════════════════════════════════════════════════
# TEST 2: Multi-GPU Test (Single Node)
# ═══════════════════════════════════════════════════════════════

if [ -z "$RUN_TEST" ] || [ "$RUN_TEST" = "2" ]; then
    run_test "Multi-GPU Training (NCCL)" \
        "examples/kubernetes-pod-multi-gpu-single-node.yaml" \
        "gpu-cluster-test-multi-gpu-single-node" \
        "kubectl delete pod gpu-cluster-test-multi-gpu-single-node -n $NAMESPACE --ignore-not-found=true"
fi

# ═══════════════════════════════════════════════════════════════
# TEST 3: Multi-Node DDP Test (InfiniBand/NCCL)
# ═══════════════════════════════════════════════════════════════

if ([ -z "$RUN_TEST" ] || [ "$RUN_TEST" = "3" ]) && [ "$SKIP_MULTINODE" = false ]; then
    TESTS_RUN=$((TESTS_RUN + 1))
    
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Test $TESTS_RUN: Multi-Node DDP Training${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────${NC}"
    
    # Cleanup
    kubectl delete statefulset gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
    kubectl delete service gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
    sleep 5
    
    # Deploy
    echo "  Deploying StatefulSet..."
    if kubectl apply -f examples/kubernetes-statefulset-multi-node-ddp.yaml -n "$NAMESPACE"; then
        
        # Wait for both pods to run and complete training
        # StatefulSets don't go to "Succeeded" so we check logs for completion
        echo "  Waiting for DDP training to complete (timeout: ${TIMEOUT}s)..."
        start_time=$(date +%s)
        completed=false
        
        while true; do
            elapsed=$(($(date +%s) - start_time))
            
            # Check if pod-0 logs contain throughput (training completed)
            if kubectl logs gpu-cluster-test-ddp-0 -n "$NAMESPACE" 2>/dev/null | grep -q "Throughput:"; then
                completed=true
                break
            fi
            
            if [ $elapsed -gt $TIMEOUT ]; then
                echo -e "  ${YELLOW}⚠ Timeout waiting for training${NC}"
                break
            fi
            
            # Check pod status
            pod0_status=$(kubectl get pod gpu-cluster-test-ddp-0 -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
            pod1_status=$(kubectl get pod gpu-cluster-test-ddp-1 -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
            
            if [ "$pod0_status" = "Failed" ] || [ "$pod1_status" = "Failed" ]; then
                echo -e "  ${RED}✗ Pod failed${NC}"
                break
            fi
            
            sleep 10
        done
        
        if [ "$completed" = true ]; then
            echo -e "  ${GREEN}✓ Training completed successfully${NC}"
            echo ""
            echo "  Output from rank 0 (last 20 lines):"
            echo "  ─────────────────────────"
            kubectl logs gpu-cluster-test-ddp-0 -n "$NAMESPACE" --tail=20 2>/dev/null | sed 's/^/  /'
            
            throughput=$(kubectl logs gpu-cluster-test-ddp-0 -n "$NAMESPACE" 2>/dev/null | grep -i "throughput" | tail -1)
            if [ -n "$throughput" ]; then
                echo ""
                echo -e "  ${GREEN}Result: $throughput${NC}"
            fi
            
            RESULT_NAMES+=("Multi-Node DDP")
            RESULT_STATUS+=("PASSED")
            TESTS_PASSED=$((TESTS_PASSED + 1))
        else
            echo ""
            echo "  Last logs from pod-0:"
            kubectl logs gpu-cluster-test-ddp-0 -n "$NAMESPACE" --tail=10 2>/dev/null | sed 's/^/  /' || echo "  (no logs)"
            RESULT_NAMES+=("Multi-Node DDP")
            RESULT_STATUS+=("FAILED")
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        RESULT_NAMES+=("Multi-Node DDP")
        RESULT_STATUS+=("FAILED (deploy)")
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    
    # Cleanup
    echo ""
    echo "  Cleaning up..."
    kubectl delete statefulset gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
    kubectl delete service gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
else
    echo ""
    echo -e "${YELLOW}Skipping Multi-Node DDP test (--skip-multinode)${NC}"
fi

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

echo ""
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    TEST SUMMARY${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

for i in "${!RESULT_NAMES[@]}"; do
    test_name="${RESULT_NAMES[$i]}"
    result="${RESULT_STATUS[$i]}"
    if [[ $result == "PASSED" ]]; then
        echo -e "  ${GREEN}✓${NC} $test_name: ${GREEN}$result${NC}"
    else
        echo -e "  ${RED}✗${NC} $test_name: ${RED}$result${NC}"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────────"
echo -e "  Total:  $TESTS_RUN tests"
echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
echo "────────────────────────────────────────────────────────────"
echo ""

# ═══════════════════════════════════════════════════════════════
# FINAL CLEANUP - Remove all test resources
# ═══════════════════════════════════════════════════════════════
echo -e "${YELLOW}Final cleanup...${NC}"
kubectl delete pod gpu-cluster-test-single-gpu -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete pod gpu-cluster-test-multi-gpu-single-node -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete statefulset gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete service gpu-cluster-test-ddp -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
kubectl delete pod -l app=gpu-cluster-test -n "$NAMESPACE" --ignore-not-found=true 2>/dev/null
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Cluster is healthy.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Review output above.${NC}"
    exit 1
fi
