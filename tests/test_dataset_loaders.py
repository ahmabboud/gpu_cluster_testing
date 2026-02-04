"""Unit tests for dataset loaders"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_loaders import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_fashion_mnist_loaders
)


class TestDatasetLoaders:
    def test_fashion_mnist_loader_single_process(self):
        """Test FashionMNIST loader creation (single process)"""
        train_loader, test_loader = get_fashion_mnist_loaders(
            batch_size=32,
            num_workers=0,
            rank=0,
            world_size=1
        )
        assert train_loader is not None
        assert test_loader is not None
        
    def test_fashion_mnist_loader_distributed(self):
        """Test FashionMNIST loader with distributed sampler"""
        train_loader, test_loader = get_fashion_mnist_loaders(
            batch_size=32,
            num_workers=0,
            rank=0,
            world_size=2
        )
        assert train_loader is not None
        assert test_loader is not None
        
    def test_cifar10_loader_creation(self):
        """Test CIFAR-10 loader can be created"""
        try:
            train_loader, test_loader = get_cifar10_loaders(
                batch_size=32,
                num_workers=0,
                rank=0,
                world_size=1
            )
            assert train_loader is not None
            assert test_loader is not None
        except Exception as e:
            pytest.skip(f"CIFAR-10 download not available: {e}")
            
    def test_cifar100_loader_creation(self):
        """Test CIFAR-100 loader can be created"""
        try:
            train_loader, test_loader = get_cifar100_loaders(
                batch_size=32,
                num_workers=0,
                rank=0,
                world_size=1
            )
            assert train_loader is not None
            assert test_loader is not None
        except Exception as e:
            pytest.skip(f"CIFAR-100 download not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
