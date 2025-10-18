import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import time
from .contrastive_model import ContrastiveModel, ContrastiveDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..utils.dataloader import create_dataloaders


class ContrastiveTrainer:
    """Trainer for contrastive learning model"""
    
    def __init__(self,
                 model: ContrastiveModel,
                 train_dataloader: DataLoader,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 reconstruction_weight: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'checkpoints',
                 window_size: int = None):
        """
        Args:
            model: Contrastive learning model
            train_dataloader: Training dataloader
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            reconstruction_weight: Weight for reconstruction loss
            device: Device to run training on
            save_dir: Directory to save checkpoints
            window_size: Window size for training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.device = device
        self.save_dir = save_dir
        self.window_size = window_size
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss weights
        self.reconstruction_weight = reconstruction_weight
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.reconstruction_losses = []
        
        # Best loss tracking for checkpointing
        self.best_loss = float('inf')
        self.best_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(self.train_dataloader, desc="Epoch", leave=False, unit="batch")
        
        for batch_idx, (original_data, augmented_data) in enumerate(batch_pbar):
            # Move data to device
            original_data = original_data.to(self.device)
            augmented_data = augmented_data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.model.compute_total_loss(
                original_data=original_data,
                augmented_data=augmented_data,
                reconstruction_weight=self.reconstruction_weight
            )
            
            total_loss_batch = loss_dict['total_loss']
            reconstruction_loss_batch = loss_dict['reconstruction_loss']
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_reconstruction_loss += reconstruction_loss_batch.item()
            num_batches += 1
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.6f}',
                'Recon': f'{reconstruction_loss_batch.item():.6f}'
            })
        
        batch_pbar.close()
        
        # Average losses
        avg_total_loss = total_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        
        return {
            'total_loss': avg_total_loss,
            'reconstruction_loss': avg_reconstruction_loss
        }
    
    def train(self, num_epochs: int = 100, save_best: bool = True):
        """Train the model for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # Train one epoch
            epoch_losses = self.train_epoch()
            current_loss = epoch_losses['total_loss']
            
            # Store losses
            self.train_losses.append(current_loss)
            self.reconstruction_losses.append(epoch_losses['reconstruction_loss'])
            
            epoch_time = time.time() - epoch_start_time
            
            # Update progress bar with current losses
            epoch_pbar.set_postfix({
                'Total Loss': f'{current_loss:.6f}',
                'Recon Loss': f'{epoch_losses["reconstruction_loss"]:.6f}',
                'Best Loss': f'{self.best_loss:.6f}',
                'Time': f'{epoch_time:.2f}s'
            })
            
            # Save best model if loss improved
            if save_best and current_loss < self.best_loss:
                old_best_loss = self.best_loss
                self.best_loss = current_loss
                self.best_epoch = epoch + 1
                print(f"\n🎉 New best loss: {current_loss:.6f} (improvement from {old_best_loss:.6f})")
                self.save_checkpoint(epoch + 1, is_best=True)
        
        epoch_pbar.close()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
        
        # Save final checkpoint
        self.save_checkpoint(num_epochs, is_final=True)
    
    def save_checkpoint(self, epoch: int, is_final: bool = False, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'reconstruction_losses': self.reconstruction_losses,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'model_config': {
                'input_dim': self.model.input_dim,
                'd_model': self.model.d_model,
                'dropout': self.model.dropout,
                'window_size': self.model.window_size,
                'stacked_tcn_channels': self.model.stacked_tcn_channels,
                'stacked_tcn_ks_list': self.model.stacked_tcn_ks_list,
                'stacked_tcn_activation': self.model.stacked_tcn_activation,
            }
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
            print(f"🏆 Best model saved to {checkpoint_path} (loss: {self.best_loss:.6f})")
        elif is_final:
            checkpoint_path = os.path.join(self.save_dir, 'final_checkpoint.pth')
            print(f"Final checkpoint saved to {checkpoint_path}")
        else:
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
            print(f"Checkpoint saved to {checkpoint_path}")
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.reconstruction_losses = checkpoint.get('reconstruction_losses', [])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        # Check if we have data to plot
        if not self.train_losses and not self.reconstruction_losses:
            print("No training data available to plot.")
            return
        
        # Create subplots based on available data
        if self.train_losses and self.reconstruction_losses:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Total loss
            axes[0].plot(self.train_losses, label='Train', color='blue')
            axes[0].set_title('Total Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Reconstruction loss
            axes[1].plot(self.reconstruction_losses, label='Train', color='orange')
            axes[1].set_title('Reconstruction Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True)
            
        elif self.train_losses:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(self.train_losses, label='Train', color='blue')
            ax.set_title('Total Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
            
        elif self.reconstruction_losses:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(self.reconstruction_losses, label='Train', color='orange')
            ax.set_title('Reconstruction Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_contrastive_dataloaders(dataset_type: str,
                                 data_path: str,
                                 window_size: int = 128,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 mask_mode: str = 'none',
                                 mask_ratio: float = 0.0,
                                 mask_seed: int = None,
                                 **kwargs) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create dataloaders for contrastive learning
    
    Args:
        dataset_type: Type of dataset
        data_path: Path to dataset
        window_size: Size of windows
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Additional arguments for dataset loading
    
    Returns:
        Tuple of (train_dataloader, None)
    """
    # Load dataset using existing dataloader
    dataloaders = create_dataloaders(
        dataset_type=dataset_type,
        data_path=data_path,
        window_size=window_size,
        stride=1,  # Non-overlapping windows
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
    
    # Get training data
    train_dataset = dataloaders['train'].dataset
    
    # Create contrastive dataset
    contrastive_train_dataset = ContrastiveDataset(
        data=train_dataset.data,
        window_size=window_size,
        stride=1,
        mask_mode=mask_mode,
        mask_ratio=mask_ratio,
        mask_seed=mask_seed
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        contrastive_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    
    return train_dataloader, None