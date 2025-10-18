import torch
import torch.nn as nn
import torch.nn.functional as F
from .TCN_Module import Stacked_TCN


class Encoder(nn.Module):
    """Encoder with TCN only"""
    def __init__(self, 
                 input_dim, 
                 d_model, 
                 dropout=0.1,
                 window_size=5000,
                 stacked_tcn_channels=None,
                 stacked_tcn_ks_list=None,
                 stacked_tcn_activation='gelu'):
        """
        Args:
            input_dim: Input dimension
            d_model: Model dimension for output projection
            dropout: Dropout rate
            window_size: Window size for TCN
            stacked_tcn_channels: Channels for Stacked_TCN
            stacked_tcn_ks_list: Kernel sizes for Stacked_TCN
            stacked_tcn_activation: Activation for Stacked_TCN
        """
        super(Encoder, self).__init__()
        
        # Stacked_TCN Block
        if stacked_tcn_channels is None:
            stacked_tcn_channels = [d_model//2, d_model]
        if stacked_tcn_ks_list is None:
            stacked_tcn_ks_list = [2, 3]
        
        self.tcn_block = Stacked_TCN(
            n_dims=input_dim,
            num_channels=stacked_tcn_channels,
            activation=stacked_tcn_activation,
            window_size=window_size,
            ks_list=stacked_tcn_ks_list,
            dropout=dropout
        )
        
        # Output projection from TCN output (input_dim) to d_model
        self.output_projection = nn.Linear(input_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Transpose input for TCN: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x_transposed = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        
        # Apply TCN
        tcn_output = self.tcn_block(x_transposed)  # (batch_size, input_dim, seq_len)
        
        # Transpose back: (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        tcn_output_transposed = tcn_output.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        
        # Project to d_model
        output = self.output_projection(tcn_output_transposed)  # (batch_size, seq_len, d_model)
        
        return output