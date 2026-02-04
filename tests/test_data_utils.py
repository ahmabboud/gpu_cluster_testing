"""Unit tests for data utilities"""
import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import (
    generate_synthetic_batch,
    generate_synthetic_sequence_batch,
    SyntheticDataLoader,
)


class TestSyntheticImageData:
    def test_generate_image_data_rgb(self):
        """Test RGB image generation"""
        device = torch.device("cpu")
        images, labels = generate_synthetic_batch(
            batch_size=100,
            num_channels=3,
            height=224,
            width=224,
            num_classes=1000,
            device=device,
        )
        assert images.shape == (100, 3, 224, 224)
        assert labels.shape == (100,)
        assert labels.max() < 1000
        
    def test_generate_image_data_grayscale(self):
        """Test grayscale image generation"""
        device = torch.device("cpu")
        images, labels = generate_synthetic_batch(
            batch_size=100,
            num_channels=1,
            height=28,
            width=28,
            num_classes=10,
            device=device,
        )
        assert images.shape == (100, 1, 28, 28)
        assert labels.shape == (100,)
        assert labels.max() < 10


class TestSyntheticTextData:
    def test_generate_text_data(self):
        """Test text sequence generation"""
        device = torch.device("cpu")
        sequences, labels = generate_synthetic_sequence_batch(
            batch_size=100,
            seq_length=128,
            vocab_size=10000,
            device=device,
        )
        assert sequences.shape == (100, 128)
        assert labels.shape == (100, 128)
        assert sequences.max() < 10000
        assert labels.max() < 10000


class TestSyntheticDataLoader:
    def test_dataloader_creation(self):
        """Test synthetic dataloader creation"""
        dataloader = SyntheticDataLoader(
            num_batches=4,
            batch_size=32,
            data_shape=(3, 224, 224),
            num_classes=1000,
            device=torch.device("cpu"),
            data_type="image",
        )
        assert dataloader is not None
        
    def test_dataloader_iteration(self):
        """Test iterating through dataloader"""
        dataloader = SyntheticDataLoader(
            num_batches=4,
            batch_size=32,
            data_shape=(3, 224, 224),
            num_classes=1000,
            device=torch.device("cpu"),
            data_type="image",
        )
        images, labels = next(iter(dataloader))
        assert images.shape[0] <= 32  # batch size
        assert images.shape[1:] == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
