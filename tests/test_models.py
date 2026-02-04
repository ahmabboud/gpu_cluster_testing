"""Unit tests for model architectures"""
import pytest
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import resnet18, resnet50, transformer


class TestResNet18:
    def test_resnet18_creation(self):
        """Test ResNet18 model can be created"""
        model = resnet18(num_classes=10, in_channels=1)
        assert model is not None
        
    def test_resnet18_forward_grayscale(self):
        """Test ResNet18 forward pass with grayscale images"""
        model = resnet18(num_classes=10, in_channels=1)
        batch = torch.randn(2, 1, 28, 28)
        output = model(batch)
        assert output.shape == (2, 10)
    
    def test_resnet18_forward_rgb(self):
        """Test ResNet18 forward pass with RGB images"""
        model = resnet18(num_classes=100, in_channels=3)
        batch = torch.randn(2, 3, 32, 32)
        output = model(batch)
        assert output.shape == (2, 100)


class TestResNet50:
    def test_resnet50_creation(self):
        """Test ResNet50 model can be created"""
        model = resnet50(num_classes=1000)
        assert model is not None
        
    def test_resnet50_forward(self):
        """Test ResNet50 forward pass"""
        model = resnet50(num_classes=1000)
        batch = torch.randn(2, 3, 224, 224)
        output = model(batch)
        assert output.shape == (2, 1000)


class TestTransformer:
    def test_transformer_creation(self):
        """Test Transformer model can be created"""
        model = transformer(vocab_size=10000, d_model=512, nhead=8, 
                          num_layers=6, dim_feedforward=2048)
        assert model is not None
        
    def test_transformer_forward(self):
        """Test Transformer forward pass"""
        model = transformer(vocab_size=10000, d_model=512, nhead=8, 
                          num_layers=6, dim_feedforward=2048)
        batch = torch.randint(0, 10000, (2, 32))
        output = model(batch)
        assert output.shape == (2, 32, 10000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
