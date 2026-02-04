#!/usr/bin/env python3
"""
Distributed Training Script for GPU Cluster Acceptance Testing

This script performs distributed training with synthetic data to validate:
1. GPU compute performance
2. Inter-GPU/Inter-node communication (NCCL)
3. Cluster stability under sustained load

Zero dependencies on external datasets or storage.
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Import our models and utilities
from models.resnet import ResNet50, get_model_info as get_resnet_info
from models.resnet18 import resnet18
from models.transformer import TransformerLanguageModel, get_model_info as get_transformer_info
from data_utils import SyntheticDataLoader
from dataset_loaders import get_dataloader, get_fashion_mnist_loaders


def setup_distributed():
    """
    Initialize distributed training environment.
    Supports multiple backends (NCCL, Gloo, MPI).
    """
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is not available")
    
    # Get distributed environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Determine backend
    backend = os.environ.get("BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    
    # Initialize process group
    if world_size > 1:
        # torchrun sets MASTER_ADDR/MASTER_PORT automatically; use env:// to let PyTorch read them
        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )
    
    return rank, world_size, local_rank, backend


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int) -> torch.device:
    """
    Get the appropriate device for this process.
    
    Args:
        local_rank: Local rank of the process
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")


def create_model(model_type: str, num_classes: int, device: torch.device, 
                 in_channels: int = 3) -> nn.Module:
    """
    Create and initialize the model.
    
    Args:
        model_type: Type of model ("resnet50", "resnet18", or "transformer")
        num_classes: Number of output classes (or vocab size for transformer)
        device: Target device
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Model instance
    """
    if model_type == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif model_type == "resnet18":
        # ResNet18 - matching Nebius KubeRay pattern
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    elif model_type == "transformer":
        model = TransformerLanguageModel(
            vocab_size=num_classes,
            d_model=1024,
            nhead=16,
            num_layers=12,
            dim_feedforward=4096,
            max_seq_length=512
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_step(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    model_type: str
) -> float:
    """
    Execute a single training step.
    
    Args:
        model: The model to train
        data: Input data
        labels: Target labels
        optimizer: Optimizer
        criterion: Loss function
        model_type: Type of model for handling different forward signatures
    
    Returns:
        Loss value
    """
    optimizer.zero_grad()
    
    if model_type == "transformer":
        # For transformer, we need to prepare src and tgt
        # Use a simple autoregressive setup
        src = data[:, :-1].transpose(0, 1)  # [seq_len-1, batch_size]
        tgt = data[:, 1:].transpose(0, 1)   # [seq_len-1, batch_size]
        output = model(src)
        # Flatten for loss computation
        output = output.reshape(-1, output.size(-1))
        tgt = tgt.reshape(-1)
    else:
        # ResNet forward
        output = model(data)
        tgt = labels
    
    loss = criterion(output, tgt)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def run_training(
    rank: int,
    world_size: int,
    local_rank: int,
    args: argparse.Namespace
):
    """
    Main training loop with performance profiling.
    
    Args:
        rank: Global rank of the process
        world_size: Total number of processes
        local_rank: Local rank on the node
        args: Command-line arguments
    """
    device = get_device(local_rank)
    
    # Print configuration (rank 0 only)
    if rank == 0:
        print("=" * 80)
        print("GPU CLUSTER ACCEPTANCE TEST")
        print("=" * 80)
        print(f"World Size: {world_size}")
        print(f"Model Type: {args.model}")
        print(f"Batch Size (per GPU): {args.batch_size}")
        print(f"Global Batch Size: {args.batch_size * world_size}")
        print(f"Device: {device}")
        print(f"Backend: {os.environ.get('BACKEND', 'auto-detected')}")
        print("=" * 80)
        sys.stdout.flush()
    
    # Create model - determine input channels based on dataset
    in_channels = 3  # Default RGB
    if args.data_mode == "fashion_mnist":
        in_channels = 1  # Grayscale
    
    model = create_model(args.model, args.num_classes, device, in_channels=in_channels)
    
    # Wrap model with DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    # Print model info (rank 0 only)
    if rank == 0:
        if args.model in ["resnet50", "resnet18"]:
            info = get_resnet_info(model.module if world_size > 1 else model)
        else:
            info = get_transformer_info(model.module if world_size > 1 else model)
        print(f"Model Parameters: {info['total_params']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print("=" * 80)
        sys.stdout.flush()
    
    # Create optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loader
    if args.data_mode == "synthetic":
        # Synthetic data generation (default, zero dependencies)
        if args.model in ["resnet50", "resnet18"]:
            data_shape = (3, 224, 224)  # ImageNet-like for ResNet models
            data_type = "image"
        else:
            data_shape = (512,)  # Sequence length for Transformer
            data_type = "sequence"
        
        train_loader = SyntheticDataLoader(
            num_batches=args.warmup_iterations + args.active_iterations,
            batch_size=args.batch_size,
            data_shape=data_shape,
            num_classes=args.num_classes,
            device=device,
            data_type=data_type
        )
    else:
        # Real dataset loader
        train_loader, _ = get_dataloader(
            data_mode=args.data_mode,
            batch_size=args.batch_size,
            num_workers=4,
            data_dir=args.data_dir
        )
        
        if train_loader is None:
            if rank == 0:
                print(f"Failed to load {args.data_mode} dataset, falling back to synthetic data")
                sys.stdout.flush()
            # Fallback to synthetic
            if args.model in ["resnet50", "resnet18"]:
                data_shape = (3, 224, 224)
                data_type = "image"
            else:
                data_shape = (512,)
                data_type = "sequence"
            
            train_loader = SyntheticDataLoader(
                num_batches=args.warmup_iterations + args.active_iterations,
                batch_size=args.batch_size,
                data_shape=data_shape,
                num_classes=args.num_classes,
                device=device,
                data_type=data_type
            )
    
    # Training loop with profiling
    model.train()
    
    # Handle different data loader types
    is_real_dataset = args.data_mode != "synthetic"
    total_batches = args.warmup_iterations + args.active_iterations
    batch_count = 0
    epoch = 0
    
    if rank == 0:
        print(f"Data Mode: {args.data_mode}")
        sys.stdout.flush()
    
    # Warmup phase
    if rank == 0:
        print(f"Starting warmup phase ({args.warmup_iterations} iterations)...")
        sys.stdout.flush()
    
    while batch_count < args.warmup_iterations:
        # Handle real datasets with multiple epochs
        if is_real_dataset:
            for data, labels in train_loader:
                if batch_count >= args.warmup_iterations:
                    break
                data, labels = data.to(device), labels.to(device)
                train_step(model, data, labels, optimizer, criterion, args.model)
                batch_count += 1
            epoch += 1
        else:
            # Synthetic data - direct iteration
            for i, (data, labels) in enumerate(train_loader):
                if i >= args.warmup_iterations:
                    break
                train_step(model, data, labels, optimizer, criterion, args.model)
            batch_count = args.warmup_iterations
            break
    
    # Synchronize before measurement phase
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    
    # Active measurement phase
    if rank == 0:
        print(f"Starting measurement phase ({args.active_iterations} iterations)...")
        sys.stdout.flush()
    
    start_time = time.time()
    total_samples = 0
    step_times = []
    
    batch_count = 0
    while batch_count < args.active_iterations:
        # Handle real datasets with multiple epochs
        if is_real_dataset:
            for data, labels in train_loader:
                if batch_count >= args.active_iterations:
                    break
                
                data, labels = data.to(device), labels.to(device)
                step_start = time.time()
                loss = train_step(model, data, labels, optimizer, criterion, args.model)
                
                # Synchronize to measure actual step time including NCCL
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                total_samples += args.batch_size
                
                if rank == 0 and batch_count % 10 == 0:
                    print(f"  Iteration {batch_count}/{args.active_iterations}, "
                          f"Loss: {loss:.4f}, Step Time: {step_time:.4f}s")
                    sys.stdout.flush()
                
                batch_count += 1
            epoch += 1
        else:
            # Synthetic data - direct iteration
            for i, (data, labels) in enumerate(train_loader):
                if i < args.warmup_iterations:
                    continue
                if i >= args.warmup_iterations + args.active_iterations:
                    break
                
                step_start = time.time()
                loss = train_step(model, data, labels, optimizer, criterion, args.model)
                
                # Synchronize to measure actual step time including NCCL
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                total_samples += args.batch_size
                
                if rank == 0 and (i - args.warmup_iterations) % 10 == 0:
                    print(f"  Iteration {i - args.warmup_iterations}/{args.active_iterations}, "
                          f"Loss: {loss:.4f}, Step Time: {step_time:.4f}s")
                    sys.stdout.flush()
            
            batch_count = args.active_iterations
            break
    
    # Final synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    throughput = (total_samples * world_size) / total_time if total_time > 0 else 0
    
    # Gather metrics from all ranks
    if world_size > 1:
        # Convert to tensors for gathering
        metrics_tensor = torch.tensor([avg_step_time, throughput], device=device)
        gathered_metrics = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_metrics, metrics_tensor)
        
        if rank == 0:
            gathered_metrics = torch.stack(gathered_metrics).cpu().numpy()
            global_avg_step_time = gathered_metrics[:, 0].mean()
            global_throughput = gathered_metrics[:, 1].sum()
        else:
            global_avg_step_time = 0
            global_throughput = 0
    else:
        global_avg_step_time = avg_step_time
        global_throughput = throughput
    
    # Print results (rank 0 only)
    if rank == 0:
        print("=" * 80)
        print("PERFORMANCE RESULTS")
        print("=" * 80)
        print(f"Total Training Time: {total_time:.2f}s")
        print(f"Average Step Time: {global_avg_step_time:.4f}s")
        print(f"Throughput: {global_throughput:.2f} samples/sec")
        print(f"Samples Processed: {total_samples * world_size:,}")
        print("=" * 80)
        
        # Save results to JSON
        results = {
            "model": args.model,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "total_time_seconds": total_time,
            "average_step_time_seconds": global_avg_step_time,
            "throughput_samples_per_second": global_throughput,
            "total_samples": total_samples * world_size,
            "device": str(device),
            "backend": os.environ.get("BACKEND", "auto-detected")
        }
        
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to results.json")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="GPU Cluster Acceptance Testing")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "transformer"],
        help="Model type to use for testing (resnet18 matches Nebius KubeRay pattern)"
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "fashion_mnist", "cifar10", "cifar100", "imagenet"],
        help="Data source: synthetic (default), fashion_mnist (Nebius pattern), cifar10, cifar100, or imagenet"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for dataset storage"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (or vocab size for transformer)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=50,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--active-iterations",
        type=int,
        default=100,
        help="Number of active measurement iterations"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup distributed training
        rank, world_size, local_rank, backend = setup_distributed()
        
        # Run training
        run_training(rank, world_size, local_rank, args)
        
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        raise
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()
