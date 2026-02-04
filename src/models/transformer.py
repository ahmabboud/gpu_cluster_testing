"""
Transformer Model Implementation for GPU Cluster Testing

Optimized for testing high-bandwidth throughput and large-packet synchronization.
This implementation emphasizes gradient patterns typical of large language model workloads.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for cluster acceptance testing.
    
    Characteristics for testing:
    - Large attention matrices → High memory bandwidth requirements
    - Large gradient tensors → Tests high-throughput NCCL operations
    - Self-attention → Quadratic complexity stress test
    
    This model is designed to stress-test:
    1. GPU-GPU interconnect bandwidth (NVLink/InfiniBand)
    2. Large tensor all-reduce operations
    3. Memory subsystem throughput
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        nhead: int = 16,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence to prevent attention to future positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            src: Source sequence [seq_len, batch_size]
            tgt: Target sequence [seq_len, batch_size]
            src_mask: Source attention mask
            tgt_mask: Target attention mask
        
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        # Embed tokens and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Generate masks if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Encode source
        memory = self.transformer_encoder(src, src_mask)
        
        # Decode target
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        
        return output


class TransformerLanguageModel(nn.Module):
    """
    Simplified Transformer for language modeling (encoder-only).
    Useful for quick acceptance tests without encoder-decoder complexity.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 24,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super(TransformerLanguageModel, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output = nn.Linear(d_model, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive modeling."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the language model.
        
        Args:
            src: Input sequence [seq_len, batch_size]
            src_mask: Attention mask
        
        Returns:
            Output logits [seq_len, batch_size, vocab_size]
        """
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(src.device)
        
        # Transform
        output = self.transformer(src, src_mask)
        
        # Project to vocabulary
        output = self.output(output)
        
        return output


def get_model_info(model: nn.Module) -> dict:
    """
    Get model statistics for logging.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model parameters count and size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": param_size_mb
    }
