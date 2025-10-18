import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from src.utils.connv1d import TemporalConv1d

class LinearAugmentation(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.weights1 = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.weights1, mean=0.0, std=0.001)
        # bias theo kênh, broadcast theo T
        self.bias1 = nn.Parameter(torch.zeros(output_dim, 1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x_ft = x.transpose(1, 2)                       # (B,F_in,T)
        y = torch.matmul(self.weights1, x_ft) + self.bias1  # (B,F_out,T) broadcast
        y = self.drop(y).transpose(1, 2)               # (B,T,F_out)
        return y

class MLPAugmentation(nn.Module):
    """Per-timestep MLP: (B,T,F_in) -> (B,T,F_out)"""
    def __init__(self, T_fixed, input_dim, output_dim, dropout=0):
        super().__init__()
        in_flat  = T_fixed * input_dim
        out_flat = T_fixed * output_dim
        # h = hidden or max(in_flat // 2, 64)
        self.T, self.Fout = T_fixed, output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_flat),
            nn.Linear(in_flat, in_flat*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_flat*2, out_flat),
        )

    def forward(self, x):   # x: (B,T,F)
        B, T, F = x.shape
        y_flat = self.net(x.reshape(B, T*F))          # (B, T*F_in) -> (B, T*F_out)
        return y_flat.reshape(B, T, self.Fout)        # (B,T,F_out)

class CNNAugmentation(nn.Module):
    """
    1D CNN augmentation (giữ nguyên T)
    - Padding đều 2 bên (non-causal 'same' đối xứng)
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.1,
                 dilation=1):
        super().__init__()
        self.conv = TemporalConv1d(
            input_dim, output_dim, kernel_size,
            dilation=dilation, causal=False, pad_mode="zeros"
        )
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B,T,C) hoặc (T,C)
        single = (x.dim() == 2)
        if single: x = x.unsqueeze(0)
        x = x.transpose(1, 2)              # (B,C,T)
        x = self.conv(x)                   # giữ nguyên T
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)              # (B,T,C_out)
        return x.squeeze(0) if single else x

class Augmentation(nn.Module):
    """
    Main augmentation class that combines all nonlinear modules
    
    Supports various input dimensions:
    - gesture: 2 features (x, y coordinates)
    - pd: 1 feature (univariate time series)
    - ecg: 2 features (2-lead ECG)
    - psm: 25 features (multivariate sensors)
    - nab: 1 feature (univariate time series)
    - smap_msl: 25 features (multivariate sensors)
    - smd: 38 features (multivariate sensors)
    - ucr: 1 feature (univariate time series)
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, temperature=1.0, window_size=5000, **kwargs):
        super(Augmentation, self).__init__()
        
        # Temperature parameter for softmax (tau)
        self.temperature = temperature
        
        # Enforce identical input/output feature dimension for all modules
        desired_output_dim = input_dim
        
        # Initialize all augmentation modules
        self.linear_module = LinearAugmentation(input_dim, desired_output_dim, dropout)
        self.mlp_module = MLPAugmentation(window_size, input_dim, desired_output_dim, dropout)
        self.cnn_module = CNNAugmentation(input_dim, desired_output_dim, 
                                        kernel_size=kwargs.get('cnn_kernel_size', 3), 
                                        dropout=dropout)
        
        # Weight parameters for combining outputs
        self.alpha = nn.Parameter(torch.ones(3) / 3)  # 3 modules: Linear, MLP, CNN
        
    def forward(self, x):
        """
        Forward pass through all augmentation modules and combine results
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Combined output of shape (batch_size, seq_len, output_dim)
        """
        # Get outputs from all modules
        linear_out = self.linear_module(x)
        mlp_out = self.mlp_module(x)
        cnn_out = self.cnn_module(x)
        
        # Stack all outputs: (num_aug, batch, seq_len, feat)
        outputs = torch.stack([linear_out, mlp_out, cnn_out], dim=0)

        # # Learned probabilities for 3 augmentations (after temperature-scaled gumbel-softmax)
        # probs = F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False, dim=0)  # (3,)

        # # Flatten per augmentation, apply weights, reshape back, then sum over aug dimension
        # num_aug, bsz, seq_len, feat = outputs.shape
        # outputs_flat = outputs.reshape(num_aug, bsz * seq_len * feat)  # (3, N)
        # weighted_flat = torch.unsqueeze(probs, -1) * outputs_flat       # (3, N)
        # weighted = weighted_flat.reshape(num_aug, bsz, seq_len, feat)   # (3, B, T, D)
        # combined_output = torch.sum(weighted, dim=0)                    # (B, T, D)
        
        probs = F.gumbel_softmax(self.alpha, tau=self.temperature, hard=False, dim=0)  # (3,)
        weighted = outputs * probs.view(-1, 1, 1, 1)                      # broadcast
        combined_output = weighted.sum(dim=0)  
        
        return combined_output
    
