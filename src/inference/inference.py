#!/usr/bin/env python3
"""
Inference script for contrastive learning model
Tests the model on test data with sliding window and visualizes results
"""

import torch
import numpy as np
import os
import sys
import argparse
from typing import Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from src.model.contrastive_model import ContrastiveModel
from src.utils.dataloader import get_dataset_loader


def adjustment(gt, pred):
    """
    Point Adjustment algorithm for anomaly detection evaluation.
    Expands predicted anomalies to cover the entire ground truth anomaly region
    when at least one point in the region is correctly predicted.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Expand backward
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Expand forward
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def binary_classification_metrics(y_true, y_pred, zero_division=0.0):
    """
    Calculate accuracy, precision, recall, f1 for binary classification.
    - zero_division: value used when denominator = 0 (e.g., no predicted positives)
    """
    yt = np.asarray(y_true).astype(bool)
    yp = np.asarray(y_pred).astype(bool)

    tp = np.sum( yt &  yp)
    fp = np.sum(~yt &  yp)
    fn = np.sum( yt & ~yp)
    tn = np.sum(~yt & ~yp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return accuracy, precision, recall, f1, (tp, fp, fn, tn)


def get_dataset_paths(dataset_type: str, base_data_path: str) -> Dict[str, str]:
    """
    Get dataset-specific paths based on dataset type
    
    Args:
        dataset_type: Type of dataset (ecg, pd, psm, nab, smap_msl, smd, ucr, gesture)
        base_data_path: Base path to datasets directory
    
    Returns:
        Dictionary with dataset-specific paths
    """
    dataset_paths = {
        'ucr': {
            'test_path': base_data_path,
            'train_path': base_data_path,
            'file_pattern': '*_test.npy'
        },
        'ecg': {
            'test_path': os.path.join(base_data_path, 'labeled', 'test'),
            'train_path': os.path.join(base_data_path, 'labeled', 'train'),
            'file_pattern': '*.pkl'
        },
        'pd': {
            'test_path': os.path.join(base_data_path, 'labeled', 'test'),
            'train_path': os.path.join(base_data_path, 'labeled', 'train'),
            'file_pattern': '*.pkl'
        },
        'psm': {
            'test_path': base_data_path,
            'train_path': base_data_path,
            'file_pattern': 'test.csv'
        },
        'nab': {
            'test_path': base_data_path,
            'train_path': base_data_path,
            'file_pattern': '*_test.npy'
        },
        'smap_msl': {
            'test_path': os.path.join(base_data_path, 'processed'),
            'train_path': os.path.join(base_data_path, 'processed'),
            'file_pattern': '*_test.npy'
        },
        'smd': {
            'test_path': base_data_path,
            'train_path': base_data_path,
            'file_pattern': '*_test.npy'
        },
        'gesture': {
            'test_path': os.path.join(base_data_path, 'labeled', 'test'),
            'train_path': os.path.join(base_data_path, 'labeled', 'train'),
            'file_pattern': '*.pkl'
        },
    }
    
    return dataset_paths.get(dataset_type, dataset_paths['ecg'])


class ContrastiveInference:
    """Inference class for contrastive learning model"""
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize inference class
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.model = None
        self.window_size = None
        self.input_dim = None
        
        # Load model
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model from checkpoint"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize default values
        
        # Load all parameters from checkpoint only
        self.input_dim = checkpoint.get('input_dim', 2)
        d_model = checkpoint.get('d_model', 256)
        projection_dim = checkpoint.get('projection_dim', 128)
        nhead = checkpoint.get('nhead', 8)
        transformer_layers = checkpoint.get('transformer_layers', 6)
        tcn_output_dim = checkpoint.get('tcn_output_dim', None)
        tcn_kernel_size = checkpoint.get('tcn_kernel_size', 3)
        tcn_num_layers = checkpoint.get('tcn_num_layers', 3)
        dropout = checkpoint.get('dropout', 0.1)
        temperature = checkpoint.get('temperature', 1.0)
        combination_method = checkpoint.get('combination_method', 'concat')
        use_contrastive = checkpoint.get('use_contrastive', True)
        
        # Decoder type is always custom_linear
        decoder_type = 'custom_linear'
        
        # Load window_size from checkpoint if available
        self.window_size = checkpoint.get('window_size', 16)
        # Load batch_size from checkpoint if available
        self.batch_size = checkpoint.get('batch_size', 32)
        
        # Load augmentation parameters from checkpoint
        self.aug_kwargs = {}
        
        # Load weights from checkpoint if available
        self.contrastive_weight = checkpoint.get('contrastive_weight', 1.0)
        self.reconstruction_weight = checkpoint.get('reconstruction_weight', 1.0)
        
        # Load other training parameters from checkpoint
        self.learning_rate = checkpoint.get('learning_rate', 1e-4)
        self.weight_decay = checkpoint.get('weight_decay', 1e-5)
        self.epsilon = checkpoint.get('epsilon', 1e-5)
        self.mask_mode = checkpoint.get('mask_mode', 'time')
        self.mask_ratio = checkpoint.get('mask_ratio', 0.2)
        self.mask_seed = checkpoint.get('mask_seed', None)
        self.device_name = checkpoint.get('device', 'cuda')
        self.seed = checkpoint.get('seed', 42)
        
        # Load additional training parameters from checkpoint
        self.use_lr_scheduler = checkpoint.get('use_lr_scheduler', True)
        self.scheduler_type = checkpoint.get('scheduler_type', 'cosine')
        self.scheduler_params = checkpoint.get('scheduler_params', {})
        self.use_wandb = checkpoint.get('use_wandb', True)
        self.project_name = checkpoint.get('project_name', 'contrastive-learning')
        self.experiment_name = checkpoint.get('experiment_name', None)
        
        # Load dataset-specific parameters from checkpoint
        self.dataset_name = checkpoint.get('dataset_name', None)
        self.data_path = checkpoint.get('data_path', None)
        
        print("Loading parameters from checkpoint")
        print(f"Using input_dim: {self.input_dim}")
        print(f"Using window_size: {self.window_size}")
        print(f"Using batch_size: {self.batch_size}")
        print(f"Using decoder_type: {decoder_type}")
        print(f"Using contrastive: {use_contrastive}")
        print(f"Using learning_rate: {self.learning_rate}")
        print(f"Using contrastive_weight: {self.contrastive_weight}")
        print(f"Using reconstruction_weight: {self.reconstruction_weight}")
        print(f"Using mask_mode: {self.mask_mode}")
        print(f"Using mask_ratio: {self.mask_ratio}")
        print(f"Using seed: {self.seed}")
        
        # Create model
        self.model = ContrastiveModel(
            input_dim=self.input_dim,
            d_model=d_model,
            projection_dim=projection_dim,
            nhead=nhead,
            transformer_layers=transformer_layers,
            tcn_output_dim=tcn_output_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
            dropout=dropout,
            temperature=temperature,
            combination_method=combination_method,
            use_contrastive=use_contrastive,
            window_size=self.window_size,
            augmentation_kwargs=self.aug_kwargs if hasattr(self, 'aug_kwargs') else None
        )
        
        # Initialize bias1 parameters by running a dummy forward pass
        # This ensures all parameters exist before loading checkpoint
        print("Initializing augmentation parameters...")
        self.model.to(self.device)  # Ensure model is on the correct device first
        
        # Use window_size from checkpoint to create correct bias1 shape
        window_size = checkpoint.get('window_size', 100)
        print(f"Using window_size from checkpoint: {window_size}")
        dummy_input = torch.randn(1, window_size, self.input_dim).to(self.device)
        with torch.no_grad():
            self.model.eval()
            # Run forward pass to initialize bias1 parameters
            _ = self.model(dummy_input, dummy_input)
        
        # Load state dict - now all parameters should exist
        state_dict = checkpoint['model_state_dict']
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if missing:
            print(f"Warning: Missing keys when loading state_dict (will be initialized randomly): {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading state_dict: {unexpected}")
            print("These keys will be ignored.")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Input dimension: {self.input_dim}")
        print(f"Window size: {self.window_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_data_with_loader(self, dataset_type: str, data_path: str, dataset_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load data using the fixed dataset loaders"""
        try:
            # Get dataset loader
            loader = get_dataset_loader(dataset_type, data_path, normalize=False)  # Don't normalize here, will be done by model
            
            # Load dataset
            if dataset_type == 'psm':
                data = loader.load_dataset()
            else:
                if not dataset_name:
                    raise ValueError(f"Dataset name required for {dataset_type}")
                data = loader.load_dataset(dataset_name)
            
            # Return test data and labels
            test_data = data['test_data']  # (features, time_steps)
            test_labels = data['test_labels']  # (time_steps,)
            
            return test_data, test_labels
            
        except Exception as e:
            print(f"Error loading data with loader for {dataset_type}: {e}")
            return None, None
    
    
    def run_inference(self, data: np.ndarray, window_size: int, stride: int = 1, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on data and return timestep anomaly scores and reconstruction
        
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of sliding window
            stride: Step size for sliding window
            batch_size: Batch size for processing windows
            
        Returns:
            Tuple of (timestep_scores, reconstruction)
        """
        # Compute timestep anomaly scores and get actual reconstruction
        timestep_scores, reconstruction = self.compute_timestep_anomaly_scores_with_reconstruction(
            data, window_size, stride, batch_size
        )
        
        return timestep_scores, reconstruction
    
    def compute_timestep_anomaly_scores(self, data: np.ndarray, window_size: int, stride: int = 1, batch_size: int = 32) -> np.ndarray:
        """
        Compute anomaly score for each timestep directly
        
        Strategy:
        - Window 0 [0:window_size-1]: Calculate anomaly score for ALL timesteps in the window
        - Window 1+ [i:i+window_size-1]: Only calculate anomaly score for the LAST timestep (i+window_size-1)
        - This ensures every timestep gets exactly one score
        
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of sliding window
            stride: Step size for sliding window
            batch_size: Batch size for processing windows
            
        Returns:
            timestep_scores: (time_steps,) - anomaly score for each timestep
        """
        _, time_steps = data.shape
        
        # Calculate number of windows to cover all timesteps
        n_windows = time_steps - window_size + 1
        
        # Initialize timestep scores
        timestep_scores = np.zeros(time_steps, dtype=np.float32)
        
        print(f"Computing timestep anomaly scores for {time_steps} timesteps from {n_windows} windows...")
        print(f"Strategy: Window 0 uses all timesteps, subsequent windows use only last timestep")
        
        # Process windows in batches
        with torch.no_grad():
            for batch_start in range(0, n_windows, batch_size):
                batch_end = min(batch_start + batch_size, n_windows)
                
                # Prepare batch data
                batch_windows = []
                batch_indices = []
                
                for i in range(batch_start, batch_end):
                    start_idx = i * stride
                    end_idx = start_idx + window_size
                    
                    # Extract window
                    window_data = data[:, start_idx:end_idx]  # (features, window_size)
                    batch_windows.append(window_data.T)  # (window_size, features)
                    batch_indices.append(i)
                
                if not batch_windows:
                    break
                
                # Convert to tensor batch: (batch_size, window_size, features)
                batch_tensor = torch.FloatTensor(np.array(batch_windows)).to(self.device)
                
                # Forward pass through model (no masking during inference)
                outputs = self.model(batch_tensor, batch_tensor)
                
                # Get reconstruction
                reconstructed = outputs['reconstructed']  # (batch_size, window_size, features)
                
                # Calculate absolute error |Eo - Ep| per timestep
                absolute_error = torch.abs(batch_tensor - reconstructed)  # (batch_size, window_size, features)
                
                # Calculate mean absolute error per timestep (across features)
                mean_absolute_error = torch.mean(absolute_error, dim=2)  # (batch_size, window_size)
                mean_absolute_error = mean_absolute_error.cpu().numpy()  # (batch_size, window_size)
                
                # Store results for this batch
                for j, window_idx in enumerate(batch_indices):
                    start_idx = window_idx * stride
                    end_idx = min(start_idx + window_size, time_steps)
                    valid_len = end_idx - start_idx
                    
                    if window_idx == 0:
                        # First window [0:window_size-1]: use all timesteps
                        timestep_scores[start_idx:end_idx] = mean_absolute_error[j][:valid_len]
                    elif window_idx == n_windows - 1:
                        # Last window: use all remaining timesteps to ensure full coverage
                        timestep_scores[start_idx:end_idx] = mean_absolute_error[j][:valid_len]
                    else:
                        # Middle windows [i:i+window_size-1]: only use the last timestep
                        last_timestep_idx = end_idx - 1
                        if last_timestep_idx < time_steps:
                            timestep_scores[last_timestep_idx] = mean_absolute_error[j][valid_len - 1]
                
                if (batch_end) % (batch_size * 10) == 0 or batch_end == n_windows:
                    print(f"Processed {batch_end}/{n_windows} windows")
        
        # Verify all timesteps are covered
        uncovered_timesteps = np.sum(timestep_scores == 0)
        if uncovered_timesteps > 0:
            print(f"Warning: {uncovered_timesteps} timesteps were not covered by any window")
        else:
            print(f"All {time_steps} timesteps are covered by windows")
        
        print(f"Timestep scores computed: {len(timestep_scores)} timesteps")
        print(f"Score range: [{np.min(timestep_scores):.6f}, {np.max(timestep_scores):.6f}]")
        print(f"Non-zero scores: {np.sum(timestep_scores != 0)}")
        
        return timestep_scores
    
    def compute_timestep_anomaly_scores_with_reconstruction(self, data: np.ndarray, window_size: int, stride: int = 1, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anomaly score for each timestep AND create full reconstruction
        
        Args:
            data: Input data of shape (features, time_steps)
            window_size: Size of sliding window
            stride: Step size for sliding window
            batch_size: Batch size for processing windows
            
        Returns:
            Tuple of (timestep_scores, reconstruction)
        """
        _, time_steps = data.shape
        
        # Calculate number of windows to cover all timesteps
        n_windows = time_steps - window_size + 1
        
        # Initialize timestep scores and reconstruction
        timestep_scores = np.zeros(time_steps, dtype=np.float32)
        reconstruction = np.zeros_like(data, dtype=np.float32)
        reconstruction_counts = np.zeros(time_steps, dtype=np.int32)
        
        print(f"Computing timestep anomaly scores and reconstruction for {time_steps} timesteps from {n_windows} windows...")
        
        # Process windows in batches
        with torch.no_grad():
            for batch_start in range(0, n_windows, batch_size):
                batch_end = min(batch_start + batch_size, n_windows)
                
                # Prepare batch data
                batch_windows = []
                batch_indices = []
                
                for i in range(batch_start, batch_end):
                    start_idx = i * stride
                    end_idx = start_idx + window_size
                    
                    # Extract window
                    window_data = data[:, start_idx:end_idx]  # (features, window_size)
                    batch_windows.append(window_data.T)  # (window_size, features)
                    batch_indices.append(i)
                
                if not batch_windows:
                    break
                
                # Convert to tensor batch: (batch_size, window_size, features)
                batch_tensor = torch.FloatTensor(np.array(batch_windows)).to(self.device)
                
                # Forward pass through model
                outputs = self.model(batch_tensor, batch_tensor)
                
                # Get reconstruction
                reconstructed = outputs['reconstructed']  # (batch_size, window_size, features)
                
                # Calculate absolute error |Eo - Ep| per timestep
                absolute_error = torch.abs(batch_tensor - reconstructed)  # (batch_size, window_size, features)
                
                # Calculate mean absolute error per timestep (across features)
                mean_absolute_error = torch.mean(absolute_error, dim=2)  # (batch_size, window_size)
                mean_absolute_error = mean_absolute_error.cpu().numpy()  # (batch_size, window_size)
                
                # Convert reconstruction back to numpy
                reconstructed_np = reconstructed.cpu().numpy()  # (batch_size, window_size, features)
                
                # Store results for this batch
                for j, window_idx in enumerate(batch_indices):
                    start_idx = window_idx * stride
                    end_idx = min(start_idx + window_size, time_steps)
                    valid_len = end_idx - start_idx
                    
                    if window_idx == 0:
                        # First window [0:window_size-1]: use all timesteps
                        timestep_scores[start_idx:end_idx] = mean_absolute_error[j][:valid_len]
                        
                        # Add reconstruction for all timesteps
                        for t in range(valid_len):
                            reconstruction[:, start_idx + t] += reconstructed_np[j, t, :]
                            reconstruction_counts[start_idx + t] += 1
                    elif window_idx == n_windows - 1:
                        # Last window: use all remaining timesteps to ensure full coverage
                        timestep_scores[start_idx:end_idx] = mean_absolute_error[j][:valid_len]
                        
                        # Add reconstruction for all timesteps
                        for t in range(valid_len):
                            reconstruction[:, start_idx + t] += reconstructed_np[j, t, :]
                            reconstruction_counts[start_idx + t] += 1
                    else:
                        # Middle windows [i:i+window_size-1]: only use the last timestep
                        last_timestep_idx = end_idx - 1
                        if last_timestep_idx < time_steps:
                            timestep_scores[last_timestep_idx] = mean_absolute_error[j][valid_len - 1]
                            
                            # Add reconstruction for the last timestep
                            reconstruction[:, last_timestep_idx] += reconstructed_np[j, valid_len - 1, :]
                            reconstruction_counts[last_timestep_idx] += 1
                
                if (batch_end) % (batch_size * 10) == 0 or batch_end == n_windows:
                    print(f"Processed {batch_end}/{n_windows} windows")
        
        # Average reconstruction across overlapping windows
        for t in range(time_steps):
            if reconstruction_counts[t] > 0:
                reconstruction[:, t] /= reconstruction_counts[t]
            else:
                # If no reconstruction available, use original data
                reconstruction[:, t] = data[:, t]
        
        # Verify all timesteps are covered
        uncovered_timesteps = np.sum(timestep_scores == 0)
        if uncovered_timesteps > 0:
            print(f"Warning: {uncovered_timesteps} timesteps were not covered by any window")
        else:
            print(f"All {time_steps} timesteps are covered by windows")
        
        print(f"Timestep scores computed: {len(timestep_scores)} timesteps")
        print(f"Score range: [{np.min(timestep_scores):.6f}, {np.max(timestep_scores):.6f}]")
        print(f"Non-zero scores: {np.sum(timestep_scores != 0)}")
        
        # Debug: Check if reconstruction matches original (should give anomaly score = 0)
        reconstruction_error = np.mean(np.abs(data - reconstruction))
        print(f"Reconstruction error (should be close to anomaly scores): {reconstruction_error:.6f}")
        
        return timestep_scores, reconstruction
    
    
    

    
    
    
    def evaluate_threshold_range(self, timestep_scores: np.ndarray, labels: np.ndarray,
                                num_thresholds: int = 1000, use_adjustment: bool = True) -> Dict:
        """
        Evaluate performance across a range of threshold values based on timestep anomaly scores
        
        Args:
            timestep_scores: Timestep-level anomaly scores (shape: time_steps,)
            labels: Ground truth labels (shape: time_steps,)
            num_thresholds: Number of threshold values to test (500-1000 recommended)
            use_adjustment: Whether to apply Point Adjustment
            
        Returns:
            Dictionary containing results for all thresholds
        """
        if labels is None or len(labels) == 0:
            return {"error": "No ground truth labels available"}
        
        # Create threshold range from [0, max(anomaly_score)]
        max_score = np.max(timestep_scores)
        min_score = 0.0  # Start from 0 as requested
        thresholds = np.linspace(min_score, max_score, num_thresholds)
        
        print(f"Evaluating {num_thresholds} thresholds")
        
        results = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': [],
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'true_negatives': []
        }
        
        best_f1 = 0
        best_threshold = 0
        best_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'tn': 0
        }
        
        # Convert labels to binary arrays
        gt = labels.astype(int)
        
        for i, threshold in enumerate(thresholds):
            # Detect anomalies at timestep level
            pred_before_pa = (timestep_scores > threshold).astype(int)
            
            # Apply Point Adjustment if requested
            if use_adjustment:
                _, pred_after_pa = adjustment(gt.copy(), pred_before_pa.copy())
            else:
                pred_after_pa = pred_before_pa
            
            # Calculate metrics after Point Adjustment
            acc, precision, recall, f1, (tp, fp, fn, tn) = binary_classification_metrics(gt, pred_after_pa)
            
            # Store results
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['accuracy'].append(acc)
            results['true_positives'].append(tp)
            results['false_positives'].append(fp)
            results['false_negatives'].append(fn)
            results['true_negatives'].append(tn)
            
            # Track best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,  # Changed from 'f1_score' to 'f1'
                    'accuracy': acc,
                    'tp': tp,  # Changed from 'true_positives' to 'tp'
                    'fp': fp,  # Changed from 'false_positives' to 'fp'
                    'fn': fn,  # Changed from 'false_negatives' to 'fn'
                    'tn': tn   # Changed from 'true_negatives' to 'tn'
                }
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_thresholds} thresholds (F1: {f1:.4f})")
        
        # Add best results
        results['best_threshold'] = best_threshold
        results['best_f1'] = best_f1
        results['best_metrics'] = best_metrics
        
        print(f"\n" + "="*80)
        print(f"THRESHOLD ANALYSIS RESULTS")
        print(f"="*80)
        print(f"Total thresholds evaluated: {num_thresholds}")
        print(f"Best threshold: {best_threshold:.6f}")
        print(f"Best F1-score: {best_f1:.4f}")
        print(f"Best precision: {best_metrics['precision']:.4f}")
        print(f"Best recall: {best_metrics['recall']:.4f}")
        print(f"Best accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Best TP: {best_metrics['tp']}")
        print(f"Best FP: {best_metrics['fp']}")
        print(f"Best FN: {best_metrics['fn']}")
        print(f"Best TN: {best_metrics['tn']}")
        print(f"="*80)
        
        return results
    
    def save_threshold_results_to_excel(self, threshold_results: Dict, save_path: str, config_info: Dict = None):
        """
        Save threshold analysis results to Excel file for single file inference
        
        Args:
            threshold_results: Results from evaluate_threshold_range
            save_path: Path to save Excel file
            config_info: Configuration information to save
        """
        if "error" in threshold_results:
            print(f"Cannot save to Excel: {threshold_results['error']}")
            return
        
        # Extract model timestamp and test filename for sheet naming
        model_timestamp = "unknown"
        test_filename = "unknown"
        
        if config_info and 'Model_Path' in config_info:
            model_path = config_info['Model_Path']
            # Extract timestamp from path like: checkpoints/ecg_20250929_143022/best_model.pth
            import re
            match = re.search(r'(\d{8}_\d{6})', model_path)
            if match:
                model_timestamp = match.group(1)
        
        if config_info and 'Test_File' in config_info:
            test_filename = config_info['Test_File'].replace('.pkl', '').replace('.npy', '').replace('.csv', '')
        
        # Create sheet name with timestamp and test file
        sheet_name = f"{model_timestamp}_{test_filename}"
        
        print(f"Saving threshold analysis results to Excel: {save_path}")
        print(f"Sheet name: {sheet_name}")
        
        # Create DataFrame with all threshold results
        df_results = pd.DataFrame({
            'Threshold': threshold_results['thresholds'],
            'Precision': threshold_results['precision'],
            'Recall': threshold_results['recall'],
            'F1_Score': threshold_results['f1_score'],
            'Accuracy': threshold_results['accuracy'],
            'True_Positives': threshold_results['true_positives'],
            'False_Positives': threshold_results['false_positives'],
            'False_Negatives': threshold_results['false_negatives'],
            'True_Negatives': threshold_results['true_negatives']
        })
        
        # Create summary DataFrame with model and test file info
        best_metrics = threshold_results['best_metrics']
        summary_data = {
            'Metric': ['Model_Timestamp', 'Test_File', 'Best_Threshold', 'Best_F1_Score', 'Best_Precision', 'Best_Recall', 'Best_Accuracy',
                      'Best_True_Positives', 'Best_False_Positives', 'Best_False_Negatives', 'Best_True_Negatives',
                      'Total_Thresholds', 'Total_Samples', 'Total_Anomalies', 'Detected_Anomalies'],
            'Value': [
                model_timestamp,
                test_filename,
                threshold_results['best_threshold'],
                best_metrics['f1'],
                best_metrics['precision'],
                best_metrics['recall'],
                best_metrics['accuracy'],
                best_metrics['tp'],
                best_metrics['fp'],
                best_metrics['fn'],
                best_metrics['tn'],
                len(threshold_results['thresholds']),
                threshold_results.get('total_samples', 'N/A'),
                threshold_results.get('total_anomalies', 'N/A'),
                threshold_results.get('detected_anomalies', 'N/A')
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # Create top performers DataFrame
        f1_array = np.array(threshold_results['f1_score'])
        top10_indices = np.argsort(f1_array)[-10:][::-1]  # Top 10 F1 scores
        
        top_performers_data = {
            'Rank': list(range(1, 11)),
            'Threshold': [threshold_results['thresholds'][i] for i in top10_indices],
            'F1_Score': [f1_array[i] for i in top10_indices],
            'Precision': [threshold_results['precision'][i] for i in top10_indices],
            'Recall': [threshold_results['recall'][i] for i in top10_indices],
            'Accuracy': [threshold_results['accuracy'][i] for i in top10_indices],
            'True_Positives': [threshold_results['true_positives'][i] for i in top10_indices],
            'False_Positives': [threshold_results['false_positives'][i] for i in top10_indices],
            'False_Negatives': [threshold_results['false_negatives'][i] for i in top10_indices],
            'True_Negatives': [threshold_results['true_negatives'][i] for i in top10_indices]
        }
        df_top_performers = pd.DataFrame(top_performers_data)
        
        # Create detailed statistics DataFrame
        stats_data = {
            'Metric': ['Mean_F1_Score', 'Std_F1_Score', 'Max_F1_Score', 'Min_F1_Score',
                      'Mean_Precision', 'Std_Precision', 'Max_Precision', 'Min_Precision',
                      'Mean_Recall', 'Std_Recall', 'Max_Recall', 'Min_Recall',
                      'Mean_Accuracy', 'Std_Accuracy', 'Max_Accuracy', 'Min_Accuracy'],
            'Value': [
                np.mean(threshold_results['f1_score']),
                np.std(threshold_results['f1_score']),
                np.max(threshold_results['f1_score']),
                np.min(threshold_results['f1_score']),
                np.mean(threshold_results['precision']),
                np.std(threshold_results['precision']),
                np.max(threshold_results['precision']),
                np.min(threshold_results['precision']),
                np.mean(threshold_results['recall']),
                np.std(threshold_results['recall']),
                np.max(threshold_results['recall']),
                np.min(threshold_results['recall']),
                np.mean(threshold_results['accuracy']),
                np.std(threshold_results['accuracy']),
                np.max(threshold_results['accuracy']),
                np.min(threshold_results['accuracy'])
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Sheet 1: All Results for this test file
            df_results.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 2: Summary for this test file
            df_summary.to_excel(writer, sheet_name=f'{sheet_name}_Summary', index=False)
            
            # Sheet 3: Top 10 Performers for this test file
            df_top_performers.to_excel(writer, sheet_name=f'{sheet_name}_Top10', index=False)
            
            # Sheet 4: Statistics for this test file
            df_stats.to_excel(writer, sheet_name=f'{sheet_name}_Stats', index=False)
            
            # Sheet 5: Threshold Analysis Info for this test file
            info_data = {
                'Parameter': ['Model_Timestamp', 'Test_File', 'Analysis_Method', 'Point_Adjustment_Used', 'Total_Thresholds_Tested',
                             'Evaluation_Metrics', 'Best_Threshold_Found',
                             'Best_F1_Score_Achieved', 'Best_Precision_Achieved', 'Best_Recall_Achieved'],
                'Value': [model_timestamp, test_filename, 'Comprehensive Threshold Analysis', 'Yes', len(threshold_results['thresholds']),
                         'Precision, Recall, F1-Score, Accuracy, TP, FP, FN, TN',
                         f"{threshold_results['best_threshold']:.6f}",
                         f"{best_metrics['f1']:.4f}",
                         f"{best_metrics['precision']:.4f}",
                         f"{best_metrics['recall']:.4f}"]
            }
            df_info = pd.DataFrame(info_data)
            df_info.to_excel(writer, sheet_name=f'{sheet_name}_Info', index=False)
            
            # Sheet 6: Configuration Information for this test file
            if config_info is not None:
                config_data = {
                    'Parameter': list(config_info.keys()),
                    'Value': [str(config_info[key]) for key in config_info.keys()]
                }
                df_config = pd.DataFrame(config_data)
                df_config.to_excel(writer, sheet_name=f'{sheet_name}_Config', index=False)
            
            print(f"Created Excel file: {save_path}")
        
        # Determine sheet count
        sheet_count = 6 if config_info is not None else 5
        
        print(f"Excel file saved successfully with {sheet_count} sheets:")
        print(f"  1. {sheet_name}: Complete threshold analysis results for {test_filename}")
        print(f"  2. {sheet_name}_Summary: Best performance summary for {test_filename}")
        print(f"  3. {sheet_name}_Top10: Top 10 F1-score results for {test_filename}")
        print(f"  4. {sheet_name}_Stats: Statistical analysis of all metrics for {test_filename}")
        print(f"  5. {sheet_name}_Info: Analysis parameters and information for {test_filename}")
        if config_info is not None:
            print(f"  6. {sheet_name}_Config: Model and inference configuration for {test_filename}")
            print(f"     - Includes: Use_Contrastive, Best_Training_Loss, model architecture")
            print(f"     - Training params: learning rate, epochs, batch size, etc.")
            print(f"     - Model params: dimensions, layers, weights, etc.")
    
    
    
    def evaluate_performance(self, labels: np.ndarray, timestep_anomalies: np.ndarray, 
                           window_size: int, stride: int = 1, use_adjustment: bool = True) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance with optional Point Adjustment
        
        Args:
            labels: Ground truth labels
            timestep_anomalies: Detected anomalies at timestep level
            window_size: Window size (not used for timestep-level evaluation)
            stride: Stride (not used for timestep-level evaluation)
            use_adjustment: Whether to apply Point Adjustment algorithm
            
        Returns:
            Dictionary of performance metrics
        """
        if labels is None or len(labels) == 0:
            return {"error": "No ground truth labels available"}
        
        # Ensure labels and timestep_anomalies have same length
        min_len = min(len(labels), len(timestep_anomalies))
        labels = labels[:min_len]
        timestep_anomalies = timestep_anomalies[:min_len]
        
        # Convert to binary arrays
        gt = labels.astype(int)
        pred_before_pa = timestep_anomalies.astype(int)
        
        # Apply Point Adjustment if requested
        if use_adjustment:
            _, pred_after_pa = adjustment(gt.copy(), pred_before_pa.copy())
        else:
            pred_after_pa = pred_before_pa
        
        # Calculate metrics after Point Adjustment
        acc, precision, recall, f1, (tp, fp, fn, tn) = binary_classification_metrics(gt, pred_after_pa)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": acc,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }
    
    def plot_inference_results(self, data: np.ndarray, reconstruction: np.ndarray, 
                              timestep_scores: np.ndarray, labels: np.ndarray = None,
                              best_threshold: float = None, save_path: str = None):
        """
        Plot inference results: test data vs reconstruction and anomaly scores
        
        Args:
            data: Original test data (features, time_steps)
            reconstruction: Reconstructed data (features, time_steps)
            timestep_scores: Anomaly scores for each timestep
            labels: Ground truth labels (optional)
            best_threshold: Best threshold found (optional)
            save_path: Path to save plot
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Test Data vs Reconstruction
        ax1 = axes[0]
        
        # Plot all available features for visualization
        num_features_to_plot = data.shape[0]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Create time axis (1 to number of timesteps)
        time_steps = np.arange(1, data.shape[1] + 1)
        
        for i in range(num_features_to_plot):
            ax1.plot(time_steps, data[i, :], label=f'Original Feature {i+1}', 
                    color=colors[i % len(colors)], alpha=0.7, linewidth=1)
            ax1.plot(time_steps, reconstruction[i, :], label=f'Reconstruction Feature {i+1}', 
                    color=colors[i % len(colors)], alpha=0.5, linewidth=1, linestyle='--')
        
        ax1.set_title('Test Data vs Reconstruction', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add ground truth labels if available
        if labels is not None:
            # Create anomaly regions overlay
            anomaly_regions = []
            in_anomaly = False
            start_idx = 0
            
            for i, label in enumerate(labels):
                if label == 1 and not in_anomaly:
                    start_idx = i
                    in_anomaly = True
                elif label == 0 and in_anomaly:
                    anomaly_regions.append((start_idx, i-1))
                    in_anomaly = False
            
            # Add last region if it ends with anomaly
            if in_anomaly:
                anomaly_regions.append((start_idx, len(labels)-1))
            
            # Highlight anomaly regions
            for start, end in anomaly_regions:
                ax1.axvspan(time_steps[start], time_steps[end], alpha=0.2, color='red', 
                           label='Ground Truth Anomalies' if start == anomaly_regions[0][0] else "")
        
        # Plot 2: Anomaly Scores
        ax2 = axes[1]
        
        # Plot anomaly scores with same time axis
        ax2.plot(time_steps, timestep_scores, label='Anomaly Score', color='purple', linewidth=1)
        
        # Add threshold line if available
        if best_threshold is not None:
            ax2.axhline(y=best_threshold, color='red', linestyle='--', 
                       label=f'Best Threshold: {best_threshold:.4f}', linewidth=2)
            
            # Highlight detected anomalies
            detected_anomalies = timestep_scores > best_threshold
            anomaly_indices = np.where(detected_anomalies)[0]
            if len(anomaly_indices) > 0:
                ax2.scatter(time_steps[anomaly_indices], timestep_scores[anomaly_indices], 
                           color='red', s=10, alpha=0.6, label='Detected Anomalies')
        
        ax2.set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean Score: {np.mean(timestep_scores):.4f}\n'
        stats_text += f'Std Score: {np.std(timestep_scores):.4f}\n'
        stats_text += f'Max Score: {np.max(timestep_scores):.4f}\n'
        stats_text += f'Min Score: {np.min(timestep_scores):.4f}'
        
        if best_threshold is not None:
            detected_count = np.sum(timestep_scores > best_threshold)
            stats_text += f'\nDetected Anomalies: {detected_count}'
            stats_text += f'\nDetection Rate: {detected_count/len(timestep_scores)*100:.2f}%'
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Contrastive Learning Model Inference')
    
    # Model arguments
    parser.add_argument('--dataset', type=str, default='ecg',
                       choices=['ecg', 'pd', 'psm', 'nab', 'smap_msl', 'smd', 'ucr', 'gesture'],
                       help='Type of dataset')
    parser.add_argument('--data_path', type=str, default=r'D:/Hoc_voi_cha_hanh/FPT/Hoc_rieng/ICIIT2025/MainModel/datasets',
                       help='Base path to datasets directory')
    parser.add_argument('--model_path', type=str, default=r'D:\Hoc_voi_cha_hanh\FPT\Hoc_rieng\ICIIT2025\MainModel\checkpoints_pd_min\pd_20250930_103524\best_model.pth',
                       help='Path to model checkpoint (if None, will use checkpoints/{dataset}/best_model.pth)')
    # Optional: specific test filename. If not provided, will try to read from config.json next to model_path
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Specific test filename to run (e.g., chfdb_chf01_275.pkl). If omitted, will try to read dataset_name from config.json')
    # Inference arguments
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size for sliding window (will be overridden by config if available)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for sliding window')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (will be overridden by config if available)')
    parser.add_argument('--num_thresholds', type=int, default=1000,
                       help='Number of threshold values to test in range evaluation (500-1000 recommended)')
    # Masking options
    parser.add_argument('--mask_mode', type=str, default='none', choices=['none', 'time', 'feature'],
                       help='Masking mode for augmented input (none, time, feature)')
    parser.add_argument('--mask_ratio', type=float, default=0.0,
                       help='Fraction of timesteps/features to mask (0.0 - 1.0)')
    parser.add_argument('--mask_seed', type=int, default=None,
                       help='Random seed for masking reproducibility')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (if None, will use inference_results_{dataset})')
    parser.add_argument('--save_plot', action='store_true', default=True,
                       help='Save inference results plot to file')
    parser.add_argument('--save_excel', action='store_true', default=True,
                       help='Save threshold analysis results to Excel file')
    
    args = parser.parse_args()
    
    # Get dataset-specific paths
    dataset_paths = get_dataset_paths(args.dataset, args.data_path)
    print(f"Dataset: {args.dataset}")
    print(f"Test path: {dataset_paths['test_path']}")
    print(f"Train path: {dataset_paths['train_path']}")
    
    # Set default model path if not provided
    if args.model_path is None:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Find latest timestamped checkpoint for this dataset
        checkpoints_base = os.path.join('/kaggle', 'working', 'ICIIT2025', 'checkpoints')
        latest_timestamp = None
        
        if os.path.exists(checkpoints_base):
            try:
                # Get all directories that start with dataset name
                import re
                pattern = f"^{args.dataset}_\\d{{8}}_\\d{{6}}$"
                matching_dirs = []
                
                for item in os.listdir(checkpoints_base):
                    if os.path.isdir(os.path.join(checkpoints_base, item)) and re.match(pattern, item):
                        matching_dirs.append(item)
                
                if matching_dirs:
                    # Sort by timestamp (newest first)
                    matching_dirs.sort(reverse=True)
                    latest_timestamp = matching_dirs[0]
                    print(f"Found latest checkpoint: {latest_timestamp}")
                else:
                    print(f"No timestamped checkpoints found for dataset: {args.dataset}")
            except Exception as e:
                print(f"Error scanning checkpoints: {e}")
        
        if latest_timestamp:
            args.model_path = os.path.join(checkpoints_base, latest_timestamp, 'best_model.pth')
        else:
            # Fallback to old format
            args.model_path = os.path.join(checkpoints_base, args.dataset, 'best_model.pth')
        
        print(f"Using model path: {args.model_path}")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model file not found at {args.model_path}")
            print(f"Available checkpoints in {checkpoints_base}:")
            if os.path.exists(checkpoints_base):
                for item in os.listdir(checkpoints_base):
                    print(f"  - {item}")
            else:
                print(f"  Checkpoints directory does not exist: {checkpoints_base}")
            return
    
    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f'inference_results_{args.dataset}'
        print(f"Using default output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference
    print(f"\nInitializing inference with model: {args.model_path}")
    inference = ContrastiveInference(args.model_path)
    
    # Auto-detect dataset_name from config if not provided
    if args.dataset_name is None:
        config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            args.dataset_name = config.get('dataset_name', None)
            if args.dataset_name:
                print(f"Auto-detected dataset_name from config: {args.dataset_name}")
            else:
                print("No dataset_name found in config, will process all available test files")
        else:
            print("No config.json found, will process all available test files")
    
    # Determine dataset name to process (no need to check file existence since we use loaders)
    if not args.dataset_name:
        # Find the first available dataset name
        import glob
        if args.dataset in ['psm']:
            # PSM uses CSV files
            pattern = os.path.join(dataset_paths['test_path'], 'test.csv')
            files = glob.glob(pattern)
            if files:
                args.dataset_name = 'psm'  # PSM doesn't need specific name
        elif args.dataset in ['nab', 'smap_msl', 'smd', 'ucr']:
            # These datasets use numpy files
            pattern = os.path.join(dataset_paths['test_path'], '*_test.npy')
            files = glob.glob(pattern)
            if files:
                # Extract dataset name from first file
                first_file = files[0]
                args.dataset_name = os.path.basename(first_file).replace('_test.npy', '')
        else:
            # ECG, PD, Gesture use pickle files
            pattern = os.path.join(dataset_paths['test_path'], '*.pkl')
            files = glob.glob(pattern)
            if files:
                # Extract dataset name from first file
                first_file = files[0]
                args.dataset_name = os.path.basename(first_file).replace('.pkl', '')
        
        if not args.dataset_name:
            print(f"❌ No test files found for dataset {args.dataset}")
            print(f"Expected pattern: {dataset_paths['file_pattern']}")
            print(f"Search path: {dataset_paths['test_path']}")
            return
        
        print(f"Auto-selected dataset: {args.dataset_name}")
    
    print(f"Processing dataset: {args.dataset_name}")
    
    # Configure masking
    inference.mask_mode = args.mask_mode
    inference.mask_ratio = max(0.0, min(1.0, float(args.mask_ratio)))
    inference.mask_seed = args.mask_seed
    
    # Use window_size from config if available, otherwise use command line argument
    if inference.window_size is not None:
        print(f"Using window_size from config: {inference.window_size}")
        args.window_size = inference.window_size
    else:
        print(f"Using window_size from command line: {args.window_size}")
    
    # Use batch_size from config if available, otherwise use command line argument
    if hasattr(inference, 'batch_size') and inference.batch_size is not None:
        print(f"Using batch_size from config: {inference.batch_size}")
        args.batch_size = inference.batch_size
    else:
        print(f"Using batch_size from command line: {args.batch_size}")
    
    # Load test data and run inference on single file
    print(f"\nStarting inference on single file...")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {args.dataset_name if args.dataset_name else args.dataset}")
    print(f"{'='*60}")
    
    try:
        # Load test data using the fixed dataset loaders
        print(f"Loading data using fixed dataset loader for {args.dataset}...")
        
        # Use dataset_name directly (already processed above)
        dataset_name = args.dataset_name
        if args.dataset == 'psm':
            dataset_name = None  # PSM doesn't need dataset name
        
        test_data, labels = inference.load_data_with_loader(args.dataset, args.data_path, dataset_name)
        
        if test_data is None:
            print(f"❌ Failed to load data for dataset {args.dataset_name if args.dataset_name else args.dataset}")
            return
        
        print(f"✅ Loaded data shape: {test_data.shape}")
        print(f"   Data range: [{np.min(test_data):.4f}, {np.max(test_data):.4f}]")
        print(f"   Data mean: {np.mean(test_data):.4f}, std: {np.std(test_data):.4f}")
        if labels is not None:
            print(f"✅ Loaded labels shape: {labels.shape}")
            print(f"   Anomaly ratio: {np.sum(labels) / len(labels) * 100:.2f}%")
        else:
            print("⚠️ No labels available")
        
        # Run inference
        print(f"Running inference with window_size={args.window_size}, stride={args.stride}...")
        timestep_scores, reconstruction = inference.run_inference(
            test_data, args.window_size, args.stride, args.batch_size
        )
        
        print(f"✅ Reconstruction shape: {reconstruction.shape}")
        print(f"   Reconstruction range: [{np.min(reconstruction):.4f}, {np.max(reconstruction):.4f}]")
        print(f"   Reconstruction mean: {np.mean(reconstruction):.4f}, std: {np.std(reconstruction):.4f}")
        print(f"✅ Timestep scores shape: {timestep_scores.shape}")
        print(f"   Scores range: [{np.min(timestep_scores):.4f}, {np.max(timestep_scores):.4f}]")
        
        # Adjust labels length if needed
        if labels is not None:
            if len(labels) > len(timestep_scores):
                # Truncate labels to match timestep scores
                original_length = len(labels)
                labels = labels[:len(timestep_scores)]
                print(f"Truncated labels from {original_length} to {len(labels)}")
            elif len(labels) < len(timestep_scores):
                # Pad labels with the last label value
                original_length = len(labels)
                last_label = labels[-1] if len(labels) > 0 else 0
                padding = np.full(len(timestep_scores) - len(labels), last_label)
                labels = np.concatenate([labels, padding])
                print(f"Padded labels from {original_length} to {len(labels)}")
            
            print(f"Final labels length: {len(labels)}")
        
        # Initialize threshold_results
        threshold_results = None
        
        # Run threshold analysis
        if labels is not None:
            print(f"Running threshold analysis for {args.dataset_name if args.dataset_name else args.dataset}...")
            threshold_results = inference.evaluate_threshold_range(
                timestep_scores, labels, args.num_thresholds, use_adjustment=True
            )
            
            # Add filename and additional info to threshold results
            threshold_results['filename'] = args.dataset_name if args.dataset_name else args.dataset
            threshold_results['data_shape'] = test_data.shape
            threshold_results['total_samples'] = len(labels)
            threshold_results['total_anomalies'] = np.sum(labels)
            
            # Add detected anomalies info using best threshold
            if 'best_threshold' in threshold_results and 'best_metrics' in threshold_results:
                best_threshold = threshold_results['best_threshold']
                predictions = timestep_scores > best_threshold
                threshold_results['detected_anomalies'] = np.sum(predictions)
                threshold_results['anomaly_detection_rate'] = np.sum(predictions) / len(predictions) if len(predictions) > 0 else 0.0
                
                # Print file-specific results
                print(f"  Dataset: {args.dataset_name if args.dataset_name else args.dataset}")
                print(f"  Total samples: {threshold_results['total_samples']}")
                print(f"  Total anomalies (GT): {threshold_results['total_anomalies']}")
                print(f"  Detected anomalies: {threshold_results['detected_anomalies']}")
                print(f"  Detection rate: {threshold_results['anomaly_detection_rate']:.4f}")
                print(f"  Best F1-Score: {threshold_results['best_metrics']['f1']:.4f}")
                print(f"  Best Threshold: {best_threshold:.6f}")
            
                
        # Evaluate performance using best threshold
        performance = None
        if labels is not None and threshold_results:
            # Use best threshold to detect anomalies
            best_threshold = threshold_results['best_threshold']
            timestep_anomalies = timestep_scores > best_threshold
            
            performance = inference.evaluate_performance(
                labels, timestep_anomalies, args.window_size, args.stride, use_adjustment=True
            )
            performance['filename'] = args.dataset_name if args.dataset_name else args.dataset
            
            print(f"\n🎯 Performance for {args.dataset_name if args.dataset_name else args.dataset}:")
            print(f"  F1-Score: {performance['f1_score']:.4f}")
            print(f"  Precision: {performance['precision']:.4f}")
            print(f"  Recall: {performance['recall']:.4f}")
            print(f"  Accuracy: {performance['accuracy']:.4f}")
        else:
            print(f"\n📊 Statistics for {args.dataset_name if args.dataset_name else args.dataset}:")
            print(f"  Total timesteps processed: {len(timestep_scores)}")
            print(f"  Mean anomaly score: {np.mean(timestep_scores):.6f}")
            print(f"  Std anomaly score: {np.std(timestep_scores):.6f}")
            print(f"  Min anomaly score: {np.min(timestep_scores):.6f}")
            print(f"  Max anomaly score: {np.max(timestep_scores):.6f}")
        
        # Plot inference results
        if args.save_plot:
            plot_path = os.path.join(args.output_dir, f'inference_results_{args.dataset_name if args.dataset_name else args.dataset}.png')
            
            # Get best threshold if available
            best_threshold = None
            if threshold_results and 'best_threshold' in threshold_results:
                best_threshold = threshold_results['best_threshold']
            
            inference.plot_inference_results(
                test_data, reconstruction, timestep_scores, labels, best_threshold, plot_path
            )
        
    except Exception as e:
        print(f"❌ Error processing {args.dataset_name if args.dataset_name else args.dataset}: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results for single file
    if threshold_results:
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        
        # Save threshold analysis results to Excel
        excel_path = os.path.join(args.output_dir, f'threshold_analysis_{args.dataset}_{args.dataset_name if args.dataset_name else "default"}.xlsx')
        
        # Get additional info from config
        additional_info = {}
        config_path = os.path.join(os.path.dirname(args.model_path), 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            additional_info = {
                'Decoder_Type': 'custom_linear',
                'Aug_nhead': config.get('aug_nhead', 'Unknown'),
                'Aug_num_layers': config.get('aug_num_layers', 'Unknown'),
                'Aug_dropout': config.get('aug_dropout', 'Unknown'),
                'Aug_temperature': config.get('aug_temperature', 'Unknown'),
                'Learning_Rate': config.get('learning_rate', 'Unknown'),
                'Weight_Decay': config.get('weight_decay', 'Unknown'),
                'Epsilon': config.get('epsilon', 'Unknown'),
                'Num_Epochs': config.get('num_epochs', 'Unknown'),
                'Use_LR_Scheduler': config.get('use_lr_scheduler', 'Unknown'),
                'Scheduler_Type': config.get('scheduler_type', 'Unknown'),
                'Use_Wandb': config.get('use_wandb', 'Unknown'),
                'Project_Name': config.get('project_name', 'Unknown'),
                'Experiment_Name': config.get('experiment_name', 'Unknown'),
                'Dataset_Name': config.get('dataset_name', 'Unknown'),
                'Data_Path': config.get('data_path', 'Unknown'),
            }
        
        config_info = {
            'Dataset': args.dataset,
            'Model_Path': args.model_path,
            'Window_Size': args.window_size,
            'Stride': args.stride,
            'Batch_Size': args.batch_size,
            'Num_Thresholds': args.num_thresholds,
            'Mask_Mode': args.mask_mode,
            'Mask_Ratio': args.mask_ratio,
            'Mask_Seed': args.mask_seed,
            'Save_Plot': args.save_plot,
            'Save_Excel': args.save_excel,
            'Output_Directory': args.output_dir,
            'Input_Dimension': inference.input_dim,
            'Model_Parameters': sum(p.numel() for p in inference.model.parameters()),
            'Device': inference.device,
            'Model_Architecture': 'ContrastiveModel',
            'Test_File': args.dataset_name if args.dataset_name else args.dataset,
            'Best_Threshold': threshold_results['best_threshold'],
            'Best_F1_Score': threshold_results['best_metrics']['f1'],
            'Best_Precision': threshold_results['best_metrics']['precision'],
            'Best_Recall': threshold_results['best_metrics']['recall'],
            'Best_Accuracy': threshold_results['best_metrics']['accuracy']
        }
        
        # Merge additional info
        config_info.update(additional_info)
        
        inference.save_threshold_results_to_excel(threshold_results, excel_path, config_info)
        
        print(f"✅ Results saved to: {excel_path}")
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Processed dataset: {args.dataset_name if args.dataset_name else args.dataset}")
    print(f"Results saved to: {args.output_dir}")
    
    if performance:
        print(f"\n📊 FINAL PERFORMANCE:")
        print(f"  F1-Score: {performance['f1_score']:.4f}")
        print(f"  Precision: {performance['precision']:.4f}")
        print(f"  Recall: {performance['recall']:.4f}")
        print(f"  Accuracy: {performance['accuracy']:.4f}")


if __name__ == "__main__":
    main()