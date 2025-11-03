"""
model.py - Simple transformer model for microbiome data
"""

import torch
import torch.nn as nn
from typing import Dict



class MicrobiomeTransformer(nn.Module):
    """
    Simple transformer model for microbiome OTU embeddings
    Handles two types of embeddings with separate input projections
    Returns per-embedding predictions with variable length output
    """
    
    def __init__(
        self,
        input_dim_type1: int = 384,
        input_dim_type2: int = 1536,  
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_output_activation: bool = True
    ):
        super().__init__()
        
        # Store activation flag
        self.use_output_activation = use_output_activation
        
        # Separate input projections for each embedding type
        self.input_projection_type1 = nn.Linear(input_dim_type1, d_model)
        self.input_projection_type2 = nn.Linear(input_dim_type2, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers - per position
        self.output_projection = nn.Linear(d_model, 1)
        self.output_activation = nn.Tanh()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: Dict with:
                - 'embeddings_type1': (batch_size, seq_len1, input_dim_type1)
                - 'embeddings_type2': (batch_size, seq_len2, input_dim_type2)
                - 'mask': (batch_size, seq_len1 + seq_len2) - combined mask
                - 'type_indicators': (batch_size, seq_len1 + seq_len2) - which type each position is
                
        Returns:
            torch.Tensor: (batch_size, seq_len1 + seq_len2) - value per embedding position
        """
        embeddings_type1 = batch['embeddings_type1']  # (batch_size, seq_len1, input_dim_type1)
        embeddings_type2 = batch['embeddings_type2']  # (batch_size, seq_len2, input_dim_type2)
        mask = batch['mask']                          # (batch_size, total_seq_len)
        type_indicators = batch['type_indicators']    # (batch_size, total_seq_len) - 0 for type1, 1 for type2
        
        # Project each type separately
        x1 = self.input_projection_type1(embeddings_type1)  # (batch_size, seq_len1, d_model)
        x2 = self.input_projection_type2(embeddings_type2)  # (batch_size, seq_len2, d_model)
        
        # Concatenate along sequence dimension
        x = torch.cat([x1, x2], dim=1)  # (batch_size, total_seq_len, d_model)
        
        # Transformer (mask padded tokens)
        x = self.transformer(x, src_key_padding_mask=~mask)  # (batch_size, total_seq_len, d_model)
        
        # Output projection per position
        output = self.output_projection(x)  # (batch_size, total_seq_len, 1)
        
        # Apply activation only if specified
        if self.use_output_activation:
            output = self.output_activation(output)  # (batch_size, total_seq_len, 1)
            
        output = output.squeeze(-1)  # (batch_size, total_seq_len)
        
        # Mask out padded positions
        output = output * mask.float()
        
        return output


# Example usage
if __name__ == "__main__":
    model = MicrobiomeTransformer(
        input_dim_type1=384,
        input_dim_type2=256,
        d_model=512,
        nhead=8,
        num_layers=6
    )
    
    # Test with dummy data
    batch_size = 4
    seq_len1 = 60  # Type 1 embeddings
    seq_len2 = 40  # Type 2 embeddings
    total_len = seq_len1 + seq_len2
    
    batch = {
        'embeddings_type1': torch.randn(batch_size, seq_len1, 384),
        'embeddings_type2': torch.randn(batch_size, seq_len2, 256),
        'mask': torch.ones(batch_size, total_len, dtype=torch.bool),
        'type_indicators': torch.cat([
            torch.zeros(batch_size, seq_len1, dtype=torch.long),  # Type 1
            torch.ones(batch_size, seq_len2, dtype=torch.long)    # Type 2
        ], dim=1)
    }
    
    # Add some padding
    batch['mask'][:, 80:] = False
    
    output = model(batch)
    print(f"Output shape: {output.shape}")  # Should be (4, 100)
    print(f"Type 1 output shape: {output[:, :seq_len1].shape}")  # (4, 60)
    print(f"Type 2 output shape: {output[:, seq_len1:seq_len1+seq_len2].shape}")  # (4, 40)
    
    # Check that padded positions are zeroed
    print(f"Padded positions sum: {output[:, 80:].sum().item()}")  # Should be 0
    
    # Check active positions
    active_output = output[:, :80]
    print(f"Active output range: {active_output.min().item():.3f} to {active_output.max().item():.3f}")