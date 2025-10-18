import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from ..modules.agf_encoder.agf_tcn import Agf_TCN
from src.modules.encoder import Encoder
from src.modules.decoder import Decoder
from src.modules.augmentation import Augmentation


class MLPProjection(nn.Module):
    """MLP for contrastive learning projection"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super(MLPProjection, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.mlp(x)


class ContrastiveModel(nn.Module):
    """Contrastive Learning Model with TCN + Transformer Encoder"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 dropout: float = 0.1,
                 window_size: int = 5000,
                 stacked_tcn_channels: Optional[List[int]] = None,
                 stacked_tcn_ks_list: Optional[List[int]] = None,
                 stacked_tcn_activation: str = 'gelu'):
        """
        Args:
            input_dim: Input dimension (number of features)
            d_model: Model dimension for output projection
            dropout: Dropout rate
            window_size: Window size for TCN
            stacked_tcn_channels: Channels for Stacked_TCN
            stacked_tcn_ks_list: Kernel sizes for Stacked_TCN
            stacked_tcn_activation: Activation for Stacked_TCN
        """
        super(ContrastiveModel, self).__init__()
        
        # Store all parameters for checkpoint saving
        self.input_dim = input_dim
        self.d_model = d_model
        self.dropout = dropout
        self.window_size = window_size
        self.stacked_tcn_channels = stacked_tcn_channels
        self.stacked_tcn_ks_list = stacked_tcn_ks_list
        self.stacked_tcn_activation = stacked_tcn_activation
        
        # Encoder for both original and augmented data
        self.encoder = Encoder(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
            window_size=window_size,
            stacked_tcn_channels=stacked_tcn_channels,
            stacked_tcn_ks_list=stacked_tcn_ks_list,
            stacked_tcn_activation=stacked_tcn_activation
        )
        
        # Decoder for reconstruction using CustomLinear
        self.decoder = Decoder(
            d_model=d_model,
            output_dim=input_dim,
            dropout=dropout
        )
        
        self.agf = Agf_TCN(
            num_inputs=input_dim,
            num_channels=stacked_tcn_channels,
            dropout=dropout,
            activation=stacked_tcn_activation,
            fuse_type=1,
            window_size=window_size,
        )
        
        
        # Augmentation module for data augmentation
        self.augmentation = Augmentation(
            input_dim=input_dim,
            output_dim=input_dim,
            dropout=dropout,
            window_size=window_size
        )
        
    def forward(self, original_data: torch.Tensor, augmented_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            original_data: Original data tensor of shape (batch_size, seq_len, input_dim)
            augmented_data: Augmented data tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Dictionary containing:
                - original_encoded: Encoded original data
                - augmented_encoded: Encoded augmented data
                - reconstructed: Reconstructed original data from augmented encoding
        """
        # Apply augmentation to the data
        augmented_data = self.augmentation(augmented_data)
        
        # Transpose input for AGF: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        augmented_data_transposed = augmented_data.transpose(1, 2)
        
        # Apply AGF
        augmented_afg = self.agf(augmented_data_transposed)
        
        # Transpose back: (batch_size, input_dim, seq_len) -> (batch_size, seq_len, input_dim)
        augmented_afg = augmented_afg.transpose(1, 2)
        
        # Reconstruct original data from augmented encoding
        # reconstructed = self.decoder(augmented_encoded)  # (batch_size, seq_len, input_dim)
        
        return {
            # 'original_encoded': original_encoded,
            # 'augmented_encoded': augmented_encoded,
            'augmented_afg': augmented_afg,
            # 'reconstructed': reconstructed
        }
    
    def compute_reconstruction_loss(self, 
                                   original_data: torch.Tensor, 
                                   reconstructed_data: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss (MSE)
        
        Args:
            original_data: Original data tensor
            reconstructed_data: Reconstructed data tensor
        
        Returns:
            Reconstruction loss
        """
        return F.mse_loss(reconstructed_data, original_data)
    
    def compute_total_loss(self, 
                          original_data: torch.Tensor, 
                          augmented_data: torch.Tensor,
                          reconstruction_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute total loss (reconstruction only)
        
        Args:
            original_data: Original data tensor
            augmented_data: Augmented data tensor
            reconstruction_weight: Weight for reconstruction loss
        
        Returns:
            Dictionary containing individual and total losses
        """
        # Forward pass
        outputs = self.forward(original_data, augmented_data)
        
        # Compute reconstruction loss
        reconstruction_loss = self.compute_reconstruction_loss(
            original_data,
            outputs['augmented_afg']
        )
        
        # Total loss
        total_loss = reconstruction_weight * reconstruction_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss
        }


class WindowSampler:
    """Window sampler for non-overlapping windows with random sampling"""
    
    def __init__(self, data: np.ndarray, window_size: int, stride: int = None):
        """
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of each window
            stride: Stride between windows (default: window_size for non-overlapping)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        # Calculate number of windows
        self.n_windows = (data.shape[1] - window_size) // self.stride + 1
        
        # Create window indices
        self.window_indices = []
        for i in range(self.n_windows):
            start_idx = i * self.stride
            end_idx = start_idx + window_size
            self.window_indices.append((start_idx, end_idx))
    
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of windows with random augmentation
        
        Args:
            batch_size: Number of windows to sample
        
        Returns:
            Tuple of (original_windows, augmented_windows)
        """
        # Randomly sample window indices
        rng = np.random.default_rng(42)
        sampled_indices = rng.choice(len(self.window_indices), batch_size, replace=True)
        
        original_windows = []
        augmented_windows = []
        
        for idx in sampled_indices:
            start_idx, end_idx = self.window_indices[idx]
            
            # Extract original window
            original_window = self.data[:, start_idx:end_idx]  # (features, window_size)
            
            # Apply random augmentation
            augmented_window = self._apply_random_augmentation(original_window)
            
            original_windows.append(original_window)
            augmented_windows.append(augmented_window)
        
        # Convert to numpy arrays and transpose to (batch_size, window_size, features)
        original_windows = np.array(original_windows).transpose(0, 2, 1)
        augmented_windows = np.array(augmented_windows).transpose(0, 2, 1)
        
        return original_windows, augmented_windows
    
    def _apply_random_augmentation(self, window: np.ndarray) -> np.ndarray:
        """Apply random augmentation to a window - augmentation handled by model"""
        # Augmentation is now handled by the contrastive model's augmentation module
        return window


class ContrastiveDataset(torch.utils.data.Dataset):
    """Dataset for contrastive learning with window sampling"""
    
    def __init__(self,
                 data: np.ndarray,
                 window_size: int,
                 stride: int = None,
                 mask_mode: str = 'none',
                 mask_ratio: float = 0.0,
                 mask_seed: int = None):
        """
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of each window
            stride: Stride between windows (default: window_size for non-overlapping)
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        # Masking configuration
        self.mask_mode = mask_mode  # 'none' | 'time' | 'feature'
        self.mask_ratio = float(mask_ratio) if mask_ratio is not None else 0.0
        self.mask_seed = mask_seed
        
        # Create window sampler
        self.sampler = WindowSampler(data, window_size, stride)
        
        # Calculate number of samples
        self.n_samples = self.sampler.n_windows
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Get window indices
        start_idx, end_idx = self.sampler.window_indices[idx]
        
        # Extract original window
        original_window = self.data[:, start_idx:end_idx]  # (features, window_size)
        
        # Apply masking to ORIGINAL window (masked window is treated as original data)
        if self.mask_mode != 'none' and self.mask_ratio > 0:
            ws = original_window.shape[1]
            feat = original_window.shape[0]
            rng = np.random.RandomState(self.mask_seed + idx) if self.mask_seed is not None else np.random
            if self.mask_mode == 'time':
                num_mask = int(max(1, round(ws * self.mask_ratio)))
                mask_idx = rng.choice(ws, size=min(num_mask, ws), replace=False)
                original_window[:, mask_idx] = 0.0
            elif self.mask_mode == 'feature':
                num_mask = int(max(1, round(feat * self.mask_ratio)))
                mask_idx = rng.choice(feat, size=min(num_mask, feat), replace=False)
                original_window[mask_idx, :] = 0.0

        # For contrastive forward, start augmented as a copy; model's augmentation will change it
        augmented_window = original_window.copy()
        
        # Transpose to (window_size, features)
        original_window = original_window.T
        augmented_window = augmented_window.T
        
        # Convert to tensors
        original_tensor = torch.FloatTensor(original_window)
        augmented_tensor = torch.FloatTensor(augmented_window)
        
        return original_tensor, augmented_tensor


