import argparse
import torch
import numpy as np
import os
import sys
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from src.model.contrastive_model import ContrastiveModel
from src.model.train_contrastive import ContrastiveTrainer, create_contrastive_dataloaders

import setproctitle
setproctitle.setproctitle("TamTC's train")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Contrastive Learning for Time Series Anomaly Detection')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='ecg', 
                       choices=['ecg', 'gesture', 'pd', 'psm', 'smap_msl', 'smd', 'ucr', 'nab'],
                       help='Type of dataset to use')
    parser.add_argument('--data_path', type=str, default='D:\Hoc_voi_cha_hanh\FPT\Hoc_rieng\ICIIT2025\MainModel\datasets\ecg',
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='chfdb_chf01_275.pkl',
                       help='Specific dataset name (for ecg, nab, smap_msl, smd)')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (number of features). If not set, auto-detected')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension for output projection')
    parser.add_argument('--dropout', type=float, default=0.01,
                       help='Dropout rate')
    
    # Stacked_TCN arguments
    parser.add_argument('--stacked_tcn_channels', type=str, default='[128]',
                       help='Channels for Stacked_TCN (JSON list format)')
    parser.add_argument('--stacked_tcn_ks_list', type=str, default='[3]',
                       help='Kernel sizes for Stacked_TCN (JSON list format)')
    parser.add_argument('--stacked_tcn_activation', type=str, default='gelu',
                       choices=['relu', 'leak', 'gelu'],
                       help='Activation function for Stacked_TCN')
    
    # Training arguments
    parser.add_argument('--window_size', type=int, default=16,
                       help='Size of windows')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    
    # Masking options for training (augmented input masking)
    parser.add_argument('--mask_mode', type=str, default='time', choices=['none', 'time', 'feature'],
                       help='Masking mode for augmented input during training')
    parser.add_argument('--mask_ratio', type=float, default=0.01,
                       help='Fraction of timesteps/features to mask (0.0 - 1.0)')
    parser.add_argument('--mask_seed', type=int, default=None,
                       help='Random seed for masking reproducibility')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--save_dir', type=str, default=r'checkpoints',
                       help='Directory to save checkpoints')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> str:
    """Get device string"""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg




def get_input_dim(dataset: str, data_path: str = None) -> int:
    """Get input dimension based on dataset type or by loading data"""
    # For ECG dataset, always use 2 features (extracted from 3-column format)
    if dataset == 'ecg':
        print("ECG dataset: Using input_dim=2 (2 features extracted from [feature1, feature2, label] format)")
        return 2
    
    # Try to load data to get actual input dimension for other datasets
    if data_path:
        try:
            from src.utils.dataloader import DatasetFactory
            loader = DatasetFactory.create_loader(dataset, data_path, normalize=True)
            data = loader.load_all_datasets()
            input_dim = data['train_data'].shape[0]  # Number of features
            print(f"Detected input dimension for {dataset}: {input_dim}")
            return input_dim
        except Exception as e:
            print(f"Could not detect input dimension for {dataset}: {e}")
            print("Using default dimension")
    
    # Fallback to default dimensions
    dims = {
        'gesture': 2,
        'pd': 1,
        'ecg': 2,
        'psm': 25,
        'nab': 1,
        'smap_msl': 25,
        'smd': 38,
        'ucr': 1
    }
    return dims.get(dataset, 2)


def main():
    """Main function"""
    args = parse_args()
    
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    
    # Get input dimension
    if args.input_dim is None:
        args.input_dim = get_input_dim(args.dataset, args.data_path)
    
    # Print configuration
    print("=" * 60)
    print("Contrastive Learning for Time Series Anomaly Detection")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Data path: {args.data_path}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Window size: {args.window_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Decoder type: custom_linear")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Create save directory with timestamp to distinguish different training runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.dataset_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        import json
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    try:
        # Create dataloaders
        print("\nCreating dataloaders (no validation)...")
        train_dataloader, _ = create_contrastive_dataloaders(
            dataset_type=args.dataset,
            data_path=args.data_path,
            dataset_name=args.dataset_name,
            window_size=args.window_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mask_mode=args.mask_mode,
            mask_ratio=max(0.0, min(1.0, float(args.mask_ratio))),
            mask_seed=args.mask_seed
        )
        
        print(f"Train batches: {len(train_dataloader)}")
        # Validation is disabled; do not print val batches

        # Detect feature dimension from a sample batch to ensure consistency
        try:
            sample_orig, _ = next(iter(train_dataloader))
            detected_input_dim = int(sample_orig.shape[-1])
            if args.input_dim is None or args.input_dim != detected_input_dim:
                print(f"Detected input_dim from data: {detected_input_dim} (overriding args.input_dim={args.input_dim})")
                args.input_dim = detected_input_dim
        except Exception as e:
            print(f"Warning: Could not detect input_dim from dataloader: {e}")
        
        # Parse JSON arguments for Stacked_TCN
        import json
        stacked_tcn_channels = json.loads(args.stacked_tcn_channels) if args.stacked_tcn_channels else None
        stacked_tcn_ks_list = json.loads(args.stacked_tcn_ks_list) if args.stacked_tcn_ks_list else None
        
        # Create model
        print(f"\nCreating model with CustomLinear decoder...")
        print(f"Encoder configuration:")
        print(f"  - Stacked_TCN channels: {stacked_tcn_channels}")
        print(f"  - Stacked_TCN kernel sizes: {stacked_tcn_ks_list}")
        print(f"  - Stacked_TCN activation: {args.stacked_tcn_activation}")
        
        model = ContrastiveModel(
            input_dim=args.input_dim,
            d_model=args.d_model,
            dropout=args.dropout,
            window_size=args.window_size,
            stacked_tcn_channels=stacked_tcn_channels,
            stacked_tcn_ks_list=stacked_tcn_ks_list,
            stacked_tcn_activation=args.stacked_tcn_activation
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        print("\nCreating trainer...")
        trainer = ContrastiveTrainer(
            model=model,
            train_dataloader=train_dataloader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            reconstruction_weight=args.reconstruction_weight,
            device=device,
            save_dir=save_dir,
            window_size=args.window_size
        )
        
        # Train model
        print(f"\nStarting training...")
        trainer.train(
            num_epochs=args.num_epochs,
            save_best=True  # Save best model when loss improves
        )
        
        # Plot training history
        plot_path = os.path.join(save_dir, 'training_history.png')
        trainer.plot_training_history(plot_path)
        
        # Save final model
        final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
        # trainer.save_checkpoint(current_loss=None)
        print(f"\nFinal model saved to {final_checkpoint_path}")
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {save_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
