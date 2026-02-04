#!/bin/bash
#
# Universal Entrypoint for GPU Cluster Acceptance Testing
#
# This script auto-detects the orchestration environment (Slurm, Kubernetes, or Bare Metal)
# and maps the appropriate environment variables to PyTorch distributed standards.
#
# Supported Environments:
# - Slurm: Uses SLURM_* variables
# - Kubernetes: Uses Kubernetes pod-specific variables
# - Bare Metal: Uses manually set RANK, WORLD_SIZE, etc.
#

set -e

# Enable verbose NCCL debugging for infrastructure engineers
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ALL}

# Detect and configure distributed environment
detect_and_configure_env() {
    echo "========================================"
    echo "Environment Detection"
    echo "========================================"
    
    # Check if running in Slurm
    if [ -n "$SLURM_PROCID" ]; then
        echo "Detected: SLURM environment"
        
        # Map Slurm variables to PyTorch distributed variables
        export RANK=${SLURM_PROCID}
        export WORLD_SIZE=${SLURM_NTASKS:-1}
        export LOCAL_RANK=${SLURM_LOCALID:-0}
        
        # Slurm master address configuration
        if [ -n "$SLURM_NODELIST" ]; then
            # Get the first node as master
            MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
            export MASTER_ADDR=${MASTER_ADDR:-$MASTER_NODE}
        fi
        export MASTER_PORT=${MASTER_PORT:-29500}
        
        echo "SLURM Configuration:"
        echo "  Job ID: $SLURM_JOB_ID"
        echo "  Nodes: $SLURM_NODELIST"
        echo "  Tasks per Node: $SLURM_TASKS_PER_NODE"
        
    # Check if running in Kubernetes
    elif [ -n "$KUBERNETES_SERVICE_HOST" ] || [ -n "$K8S_MASTER_ADDR" ]; then
        echo "Detected: Kubernetes environment"
        
        # Kubernetes variables (typically set by PyTorch operators or custom scripts)
        export RANK=${RANK:-0}
        export WORLD_SIZE=${WORLD_SIZE:-1}
        export LOCAL_RANK=${LOCAL_RANK:-0}
        export MASTER_ADDR=${MASTER_ADDR:-${K8S_MASTER_ADDR:-localhost}}
        export MASTER_PORT=${MASTER_PORT:-29500}
        
        echo "Kubernetes Configuration:"
        echo "  Pod Name: ${HOSTNAME}"
        echo "  Namespace: ${POD_NAMESPACE:-default}"
        
    # Bare metal or manual configuration
    else
        echo "Detected: Bare Metal / Manual Configuration"
        
        # Use environment variables as-is, with defaults for single-GPU testing
        export RANK=${RANK:-0}
        export WORLD_SIZE=${WORLD_SIZE:-1}
        export LOCAL_RANK=${LOCAL_RANK:-0}
        export MASTER_ADDR=${MASTER_ADDR:-localhost}
        export MASTER_PORT=${MASTER_PORT:-29500}
        
        echo "Manual Configuration Mode"
    fi
    
    # Determine backend
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        export BACKEND=${BACKEND:-nccl}
        echo "GPU detected: Using NCCL backend"
        
        # Print GPU info
        echo ""
        echo "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    else
        export BACKEND=${BACKEND:-gloo}
        echo "No GPU detected: Using Gloo backend"
    fi
    
    echo ""
    echo "PyTorch Distributed Configuration:"
    echo "  RANK: $RANK"
    echo "  WORLD_SIZE: $WORLD_SIZE"
    echo "  LOCAL_RANK: $LOCAL_RANK"
    echo "  MASTER_ADDR: $MASTER_ADDR"
    echo "  MASTER_PORT: $MASTER_PORT"
    echo "  BACKEND: $BACKEND"
    echo "========================================"
    echo ""
}

# System diagnostics
print_system_info() {
    echo "========================================"
    echo "System Diagnostics"
    echo "========================================"
    
    # CPU info
    echo "CPU Cores: $(nproc)"
    
    # Memory info
    if command -v free &> /dev/null; then
        echo "Memory:"
        free -h | grep -E "Mem:|Swap:"
    fi
    
    # Network interfaces
    echo ""
    echo "Network Interfaces:"
    ip addr show | grep -E "^[0-9]+:|inet " | head -n 10
    
    # NCCL environment variables
    echo ""
    echo "NCCL Configuration:"
    env | grep NCCL | sort || echo "  No NCCL variables set"
    
    echo "========================================"
    echo ""
}

# Health check before starting
health_check() {
    echo "Running pre-flight health check..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python3 not found"
        exit 1
    fi
    
    # Check if required Python packages are available
    python3 -c "import torch" 2>/dev/null || {
        echo "ERROR: PyTorch not found"
        exit 1
    }
    
    # Check CUDA availability if GPUs are present
    if [ "$BACKEND" = "nccl" ]; then
        python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
            echo "ERROR: CUDA is not available"
            exit 1
        }
    fi
    
    echo "Health check passed ✓"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "###############################################"
    echo "#   GPU Cluster Acceptance Testing Tool      #"
    echo "#   Nebius Infrastructure Engineering        #"
    echo "###############################################"
    echo ""
    
    # Detect environment and configure variables
    detect_and_configure_env
    
    # Print system information
    print_system_info
    
    # Run health check
    health_check
    
    # Change to the application directory
    cd /workspace/src
    
    # Execute the training script with all passed arguments
    echo "Starting training script..."
    echo "Command: python3 train.py $@"
    echo ""
    
    exec python3 train.py "$@"
}

# Run main function
main "$@"
