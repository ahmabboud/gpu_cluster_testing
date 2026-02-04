"""
Real Dataset Loaders for GPU Cluster Testing

Provides lightweight, open-source dataset options alongside synthetic data.
All datasets are automatically downloaded and cached.

Includes FashionMNIST - matching Nebius KubeRay test pattern.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, DistributedSampler
from typing import Tuple, Optional


def get_cifar10_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 dataset loaders (lightweight, 170MB).
    
    CIFAR-10: 60,000 32x32 color images in 10 classes
    - 50,000 training images
    - 10,000 test images
    
    Args:
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        data_dir: Directory to store dataset
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # ImageNet normalization (commonly used)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar100_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 dataset loaders (lightweight, 170MB).
    
    CIFAR-100: 60,000 32x32 color images in 100 classes
    More challenging than CIFAR-10 with finer-grained categories.
    
    Args:
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        data_dir: Directory to store dataset
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    normalize = transforms.Normalize(
        mean=[0.507, 0.487, 0.441],
        std=[0.267, 0.256, 0.276]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_fashion_mnist_loaders(
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    """
    Get FashionMNIST dataset loaders - matching Nebius KubeRay test pattern.
    
    FashionMNIST: 70,000 28x28 grayscale images in 10 classes
    - 60,000 training images
    - 10,000 test images
    - Only 30MB download size
    - Proven to work in Nebius production
    
    Args:
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        data_dir: Directory to store dataset
        rank: Distributed rank (default: 0)
        world_size: Total number of processes (default: 1)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # FashionMNIST normalization (calculated from dataset)
    normalize = transforms.Normalize(
        mean=(0.28604,),  # Grayscale - single channel
        std=(0.32025,)
    )
    
    # Training transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    # Use DistributedSampler if multi-GPU
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_limited_imagenet_loader(
    batch_size: int,
    num_samples: int = 5000,
    num_workers: int = 4,
    data_dir: str = "./data",
    resolution: int = 224
) -> Optional[DataLoader]:
    """
    Get a limited subset of ImageNet for acceptance testing.
    
    Uses torchvision's ImageNet with a subset of samples.
    Note: Requires ImageNet dataset to be pre-downloaded at data_dir/imagenet
    
    If dataset is not available, returns None (graceful degradation to synthetic).
    
    Args:
        batch_size: Batch size per GPU
        num_samples: Number of samples to use (default: 5000)
        num_workers: Number of data loading workers
        data_dir: Directory containing ImageNet dataset
        resolution: Image resolution (default: 224)
    
    Returns:
        DataLoader or None if dataset not available
    """
    imagenet_path = os.path.join(data_dir, "imagenet")
    
    # Check if ImageNet is available
    if not os.path.exists(imagenet_path):
        print(f"ImageNet dataset not found at {imagenet_path}")
        print("Falling back to synthetic data mode")
        return None
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        normalize,
    ])
    
    try:
        # Try to load validation set (smaller than train)
        dataset = torchvision.datasets.ImageNet(
            root=imagenet_path,
            split='val',
            transform=transform
        )
        
        # Limit to specified number of samples
        if num_samples > 0 and num_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_samples]
            dataset = Subset(dataset, indices)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Loaded ImageNet subset: {len(dataset)} samples")
        return loader
        
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Falling back to synthetic data mode")
        return None


def get_wikitext_dataset(
    batch_size: int,
    seq_length: int = 512,
    num_workers: int = 4,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get WikiText-2 dataset for language modeling (lightweight, ~4MB).
    
    WikiText-2: Language modeling dataset from Wikipedia articles.
    Suitable for transformer model testing.
    
    Args:
        batch_size: Batch size per GPU
        seq_length: Sequence length for language modeling
        num_workers: Number of data loading workers
        data_dir: Directory to store dataset
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    try:
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator
    except ImportError:
        print("torchtext not available. Install with: pip install torchtext")
        print("Falling back to synthetic data mode")
        return None, None
    
    # Load datasets
    train_iter = WikiText2(root=data_dir, split='train')
    test_iter = WikiText2(root=data_dir, split='test')
    
    # Tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter),
        specials=['<unk>', '<pad>', '<bos>', '<eos>'],
        min_freq=3
    )
    vocab.set_default_index(vocab['<unk>'])
    
    # Simplified: Return placeholder loaders
    # For full implementation, would need proper batching and padding
    print("WikiText-2 dataset prepared (simplified mode)")
    print("Note: For production use, implement proper sequence batching")
    
    return None, None


def get_dataloader(
    data_mode: str,
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = "./data",
    **kwargs
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Universal data loader factory.
    
    Args:
        data_mode: One of 'cifar10', 'cifar100', 'imagenet', 'wikitext'
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        data_dir: Directory for datasets
        **kwargs: Additional arguments for specific loaders
    
    Returns:
        Tuple of (train_loader, test_loader) or (None, None) if unavailable
    """
    data_mode = data_mode.lower()
    
    if data_mode == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers, data_dir)
    elif data_mode == 'cifar100':
        return get_cifar100_loaders(batch_size, num_workers, data_dir)
    elif data_mode == 'imagenet':
        loader = get_limited_imagenet_loader(
            batch_size,
            num_samples=kwargs.get('num_samples', 5000),
            num_workers=num_workers,
            data_dir=data_dir
        )
        return loader, None
    elif data_mode == 'wikitext':
        return get_wikitext_dataset(batch_size, num_workers=num_workers, data_dir=data_dir)
    else:
        print(f"Unknown data mode: {data_mode}")
        return None, None
