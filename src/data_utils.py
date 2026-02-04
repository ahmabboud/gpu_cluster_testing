"""
Synthetic Data Generation Utilities for Zero-Dependency Testing

This module provides functions to generate synthetic training data directly in VRAM,
eliminating the need for external datasets or storage mounting.
"""

import torch
from typing import Tuple


def generate_synthetic_batch(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    num_classes: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic batch of image data and labels in VRAM.
    
    Args:
        batch_size: Number of samples in the batch
        num_channels: Number of image channels (e.g., 3 for RGB)
        height: Image height in pixels
        width: Image width in pixels
        num_classes: Number of classification classes
        device: Target device (GPU or CPU)
    
    Returns:
        Tuple of (images, labels) tensors on the specified device
    """
    # Generate random image data directly on device
    images = torch.randn(
        batch_size, num_channels, height, width,
        device=device,
        dtype=torch.float32
    )
    
    # Generate random labels
    labels = torch.randint(
        0, num_classes,
        (batch_size,),
        device=device,
        dtype=torch.long
    )
    
    return images, labels


def generate_synthetic_sequence_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a synthetic batch of sequence data for transformer models.
    
    Args:
        batch_size: Number of sequences in the batch
        seq_length: Length of each sequence
        vocab_size: Size of the vocabulary
        device: Target device (GPU or CPU)
    
    Returns:
        Tuple of (input_ids, labels) tensors on the specified device
    """
    # Generate random token IDs
    input_ids = torch.randint(
        0, vocab_size,
        (batch_size, seq_length),
        device=device,
        dtype=torch.long
    )
    
    # For language modeling, labels are typically shifted input_ids
    labels = input_ids.clone()
    
    return input_ids, labels


class SyntheticDataLoader:
    """
    Iterator that generates synthetic data on-the-fly.
    Mimics PyTorch DataLoader interface without requiring actual data.
    """
    
    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        data_shape: Tuple[int, ...],
        num_classes: int,
        device: torch.device,
        data_type: str = "image"
    ):
        """
        Args:
            num_batches: Number of batches to generate per epoch
            batch_size: Samples per batch
            data_shape: Shape of input data (channels, height, width) for images
                       or (seq_length,) for sequences
            num_classes: Number of output classes (or vocab_size for sequences)
            device: Target device
            data_type: Type of data ("image" or "sequence")
        """
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.device = device
        self.data_type = data_type
        self.current_batch = 0
    
    def __iter__(self):
        self.current_batch = 0
        return self
    
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        self.current_batch += 1
        
        if self.data_type == "image":
            return generate_synthetic_batch(
                self.batch_size,
                self.data_shape[0],  # channels
                self.data_shape[1],  # height
                self.data_shape[2],  # width
                self.num_classes,
                self.device
            )
        elif self.data_type == "sequence":
            return generate_synthetic_sequence_batch(
                self.batch_size,
                self.data_shape[0],  # seq_length
                self.num_classes,    # vocab_size
                self.device
            )
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")
    
    def __len__(self):
        return self.num_batches
