#!/bin/bash
#
# NCCL Bandwidth & Latency Test Script for Slurm
#
# This script runs official NCCL tests (nccl-tests suite) to measure:
# - All-reduce bandwidth across GPUs
# - Latency characteristics
# - Both NVLink and InfiniBand performance
#
# Based on Nebius Soperator test patterns
#

#SBATCH --job-name=nccl-bandwidth-test
#SBATCH --output=results/nccl_bandwidth_%j.out
#SBATCH --error=results/nccl_bandwidth_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00

# Create results directory
mkdir -p results

echo "=========================================="
echo "NCCL Bandwidth and Latency Testing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))"
echo "=========================================="
echo ""

# Check if nccl-tests are available
if ! command -v all_reduce_perf &> /dev/null; then
    echo "ERROR: nccl-tests not found!"
    echo ""
    echo "Please install nccl-tests:"
    echo "  git clone https://github.com/NVIDIA/nccl-tests.git"
    echo "  cd nccl-tests"
    echo "  make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi"
    echo ""
    echo "Or use the containerized version in our main test tool."
    exit 1
fi

echo "=========================================="
echo "Test 1: Single-Node NVLink Performance"
echo "=========================================="
echo "Testing GPU-to-GPU communication via NVLink..."
echo ""

# Single-node test (NVLink)
srun --nodes=1 --ntasks=1 --gpus-per-node=8 bash -c "
    echo 'Running on: \$(hostname)'
    echo 'GPUs: \$SLURM_GPUS_ON_NODE'
    echo ''
    all_reduce_perf -b 8K -e 8G -f 2 -g \$SLURM_GPUS_ON_NODE
"

echo ""
echo "=========================================="
echo "Test 2: Single-Node InfiniBand Performance"
echo "=========================================="
echo "Testing GPU-to-GPU communication via InfiniBand..."
echo "(Disabling P2P and SHM to force IB usage)"
echo ""

# Single-node test forcing InfiniBand
# NOTE: These settings are for testing only, not production
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_ALGO=Ring
export NCCL_DEBUG=INFO

srun --nodes=1 --ntasks=1 --gpus-per-node=8 bash -c "
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_ALGO=Ring
    export NCCL_DEBUG=INFO
    echo 'Running on: \$(hostname)'
    echo 'GPUs: \$SLURM_GPUS_ON_NODE'
    echo ''
    all_reduce_perf -b 8K -e 8G -f 2 -g \$SLURM_GPUS_ON_NODE
"

# Reset for multi-node tests
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_ALGO

echo ""
echo "=========================================="
echo "Test 3: Multi-Node All-Reduce (MPI)"
echo "=========================================="
echo "Testing multi-node GPU communication..."
echo "Nodes: $SLURM_NNODES"
echo ""

# Multi-node MPI test
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

srun --mpi=pmix bash -c "
    echo 'Node: \$(hostname) | Rank: \$SLURM_PROCID | Local Rank: \$SLURM_LOCALID'
    all_reduce_perf_mpi -b 8K -e 8G -f 2 -g 1
"

echo ""
echo "=========================================="
echo "Test 4: Latency Test"
echo "=========================================="
echo "Testing communication latency..."
echo ""

# Latency test with small messages
srun --mpi=pmix bash -c "
    all_reduce_perf_mpi -b 8 -e 1K -f 2 -g 1
"

echo ""
echo "=========================================="
echo "NCCL Bandwidth Test Complete"
echo "=========================================="
echo "Results saved to: results/nccl_bandwidth_${SLURM_JOB_ID}.out"
echo ""
echo "Expected Performance (Reference):"
echo "  NVLink (H100): ~400-450 GB/s for all_reduce on 8 GPUs"
echo "  InfiniBand (HDR): ~200-240 GB/s for all_reduce on 8 GPUs"
echo "  Latency: < 50 microseconds for small messages"
echo ""
echo "Review the output above to verify:"
echo "  1. Bandwidth matches expectations for your hardware"
echo "  2. No NCCL errors or warnings"
echo "  3. All ranks complete successfully"
echo "=========================================="
