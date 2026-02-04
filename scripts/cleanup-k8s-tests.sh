#!/bin/bash
#
# Kubernetes Test Cleanup Script
#
# Automatically cleans up completed GPU acceptance tests to prevent resource accumulation.
# Can be run manually or via CronJob.
#
# Usage:
#   ./cleanup-k8s-tests.sh [namespace] [age-threshold-hours]
#
# Examples:
#   ./cleanup-k8s-tests.sh default 24    # Clean up tests older than 24 hours
#   ./cleanup-k8s-tests.sh gpu-tests 1   # Clean up tests older than 1 hour
#

set -e

NAMESPACE=${1:-default}
AGE_HOURS=${2:-24}
AGE_SECONDS=$((AGE_HOURS * 3600))
DRY_RUN=${DRY_RUN:-false}

echo "=========================================="
echo "GPU Test Cleanup Script"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Age threshold: $AGE_HOURS hours"
echo "Dry run: $DRY_RUN"
echo "=========================================="
echo ""

# Function to get completion time and calculate age
get_old_pytorchjobs() {
    kubectl get pytorchjobs -n "$NAMESPACE" -o json 2>/dev/null | \
    jq -r --arg age "$AGE_SECONDS" '.items[] | 
        select(.status.completionTime != null) | 
        select((now - (.status.completionTime | fromdateiso8601)) > ($age | tonumber)) | 
        "\(.metadata.name) \(.status.completionTime)"'
}

# Clean up PyTorchJobs
echo "1. Checking for completed PyTorchJobs..."
OLD_JOBS=$(get_old_pytorchjobs)

if [ -z "$OLD_JOBS" ]; then
    echo "   ✓ No old PyTorchJobs found"
else
    echo "   Found PyTorchJobs to clean up:"
    echo "$OLD_JOBS" | while read -r job_name completion_time; do
        echo "   - $job_name (completed: $completion_time)"
        if [ "$DRY_RUN" = "false" ]; then
            kubectl delete pytorchjob "$job_name" -n "$NAMESPACE" --ignore-not-found
            echo "     ✓ Deleted"
        else
            echo "     (dry run - would delete)"
        fi
    done
fi

echo ""

# Clean up completed pods
echo "2. Checking for completed test pods..."
OLD_PODS=$(kubectl get pods -n "$NAMESPACE" \
    -l app=gpu-acceptance-test \
    --field-selector=status.phase=Succeeded \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)

if [ -z "$OLD_PODS" ]; then
    echo "   ✓ No completed pods found"
else
    echo "   Found completed pods to clean up:"
    echo "$OLD_PODS" | while read -r pod_name; do
        if [ -n "$pod_name" ]; then
            echo "   - $pod_name"
            if [ "$DRY_RUN" = "false" ]; then
                kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found
                echo "     ✓ Deleted"
            else
                echo "     (dry run - would delete)"
            fi
        fi
    done
fi

echo ""

# Clean up failed pods older than threshold (optional - keeps recent failures)
echo "3. Checking for old failed test pods..."
FAILED_PODS=$(kubectl get pods -n "$NAMESPACE" \
    -l app=gpu-acceptance-test \
    --field-selector=status.phase=Failed \
    -o json 2>/dev/null | \
    jq -r --arg age "$AGE_SECONDS" '.items[] | 
        select((now - (.status.startTime | fromdateiso8601)) > ($age | tonumber)) | 
        .metadata.name' || true)

if [ -z "$FAILED_PODS" ]; then
    echo "   ✓ No old failed pods found"
else
    echo "   Found old failed pods to clean up:"
    echo "$FAILED_PODS" | while read -r pod_name; do
        if [ -n "$pod_name" ]; then
            echo "   - $pod_name"
            if [ "$DRY_RUN" = "false" ]; then
                kubectl delete pod "$pod_name" -n "$NAMESPACE" --ignore-not-found
                echo "     ✓ Deleted"
            else
                echo "     (dry run - would delete)"
            fi
        fi
    done
fi

echo ""

# Report remaining resources
echo "=========================================="
echo "Summary"
echo "=========================================="

REMAINING_JOBS=$(kubectl get pytorchjobs -n "$NAMESPACE" -l app=gpu-acceptance-test 2>/dev/null | tail -n +2 | wc -l || echo "0")
REMAINING_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=gpu-acceptance-test 2>/dev/null | tail -n +2 | wc -l || echo "0")

echo "Remaining in namespace '$NAMESPACE':"
echo "  - PyTorchJobs: $REMAINING_JOBS"
echo "  - Pods: $REMAINING_PODS"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "This was a dry run. No resources were deleted."
    echo "Run without DRY_RUN=true to perform actual cleanup."
fi

echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
