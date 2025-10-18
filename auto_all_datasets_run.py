#!/usr/bin/env python3
"""
Automated training and inference script for all dataset types
Supports: ECG, PD, Gesture, PSM, NAB, SMAP-MSL, SMD, UCR, WADI, SWAT
"""

import os
import subprocess
import sys
import glob
import shutil
from pathlib import Path

# Base paths
BASE = r"D:\Hoc_voi_cha_hanh\FPT\Hoc_rieng\Chungbok"
DATASETS_DIR = os.path.join(BASE, "datasets")
OUTPUT_BASE = os.path.join(BASE, "src", "inference", "inference_results")

# Dataset configurations
DATASET_CONFIGS = {
    'ucr': {
        'data_path': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'train_dir': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'test_dir': os.path.join(DATASETS_DIR, "ucr", "labeled"),
        'file_pattern': "*_train.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "ucr")
    },
    'pd': {
        'data_path': os.path.join(DATASETS_DIR, "pd"),
        'train_dir': os.path.join(DATASETS_DIR, "pd", "labeled", "train"),
        'test_dir': os.path.join(DATASETS_DIR, "pd", "labeled", "test"),
        'file_pattern': "*.pkl",
        'output_dir': os.path.join(OUTPUT_BASE, "pd")
    },
    'gesture': {
        'data_path': os.path.join(DATASETS_DIR, "gesture"),
        'train_dir': os.path.join(DATASETS_DIR, "gesture", "labeled", "train"),
        'test_dir': os.path.join(DATASETS_DIR, "gesture", "labeled", "test"),
        'file_pattern': "*.pkl",
        'output_dir': os.path.join(OUTPUT_BASE, "gesture")
    },
    'ecg': {
        'data_path': os.path.join(DATASETS_DIR, "ecg"),
        'train_dir': os.path.join(DATASETS_DIR, "ecg", "labeled", "train"),
        'test_dir': os.path.join(DATASETS_DIR, "ecg", "labeled", "test"),
        'file_pattern': "*.pkl",
        'output_dir': os.path.join(OUTPUT_BASE, "ecg")
    },
    'psm': {
        'data_path': os.path.join(DATASETS_DIR, "psm"),
        'train_dir': os.path.join(DATASETS_DIR, "psm"),
        'test_dir': os.path.join(DATASETS_DIR, "psm"),
        'file_pattern': "train.csv",  # PSM has single train/test files
        'output_dir': os.path.join(OUTPUT_BASE, "psm")
    },
    'smap_msl': {
        'data_path': os.path.join(DATASETS_DIR, "smap_msl_"),
        'train_dir': os.path.join(DATASETS_DIR, "smap_msl_", "processed"),
        'test_dir': os.path.join(DATASETS_DIR, "smap_msl_", "processed"),
        'file_pattern': "*_train.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "smap_msl")
    },
    'smd': {
        'data_path': os.path.join(DATASETS_DIR, "smd"),
        'train_dir': os.path.join(DATASETS_DIR, "smd"),
        'test_dir': os.path.join(DATASETS_DIR, "smd"),
        'file_pattern': "*_train.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "smd")
    },
    'nab': {
        'data_path': os.path.join(DATASETS_DIR, "nab"),
        'train_dir': os.path.join(DATASETS_DIR, "nab"),
        'test_dir': os.path.join(DATASETS_DIR, "nab"),
        'file_pattern': "*_train.npy",
        'output_dir': os.path.join(OUTPUT_BASE, "nab")
    }
}

def validate_dataset_paths(datasets_to_validate=None):
    """Validate dataset paths and report issues"""
    print("🔍 Validating dataset paths...")
    issues = []
    
    # If no specific datasets provided, validate all
    if datasets_to_validate is None:
        datasets_to_validate = DATASET_CONFIGS
    
    for dataset_type, config in datasets_to_validate.items():
        print(f"\nChecking {dataset_type}:")
        
        # Check data_path
        data_path = config['data_path']
        if not os.path.exists(data_path):
            issues.append(f"❌ {dataset_type}: data_path not found: {data_path}")
            print(f"  ❌ data_path: {data_path}")
        else:
            print(f"  ✅ data_path: {data_path}")
        
        # Check train_dir
        train_dir = config['train_dir']
        if not os.path.exists(train_dir):
            issues.append(f"❌ {dataset_type}: train_dir not found: {train_dir}")
            print(f"  ❌ train_dir: {train_dir}")
        else:
            print(f"  ✅ train_dir: {train_dir}")
            
            # Check for files in train_dir
            file_pattern = config['file_pattern']
            train_files = glob.glob(os.path.join(train_dir, file_pattern))
            if not train_files:
                issues.append(f"⚠️  {dataset_type}: No files found in train_dir with pattern '{file_pattern}'")
                print(f"  ⚠️  No files found with pattern: {file_pattern}")
            else:
                print(f"  ✅ Found {len(train_files)} train files")
        
        # Check test_dir
        test_dir = config['test_dir']
        if not os.path.exists(test_dir):
            issues.append(f"❌ {dataset_type}: test_dir not found: {test_dir}")
            print(f"  ❌ test_dir: {test_dir}")
        else:
            print(f"  ✅ test_dir: {test_dir}")
        
        # Check output_dir (will be created if needed)
        output_dir = config['output_dir']
        print(f"  📁 output_dir: {output_dir}")
    
    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"\n✅ All dataset paths are valid!")
        return True


def run_command(cmd, description=""):
    """Run a command inheriting the console to preserve tqdm/wandb live rendering"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")

    # Ensure real-time flushing from child Python processes
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Inherit parent's stdout/stderr so tqdm/wandb detect a TTY and update in-place
    result = subprocess.run(cmd, shell=True, env=env)

    if result.returncode != 0:
        print(f"❌ ERROR in {description} (exit code {result.returncode})")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        return True

def get_dataset_files(dataset_type, config):
    """Get list of files to process for a dataset type"""
    files = []
    
    if dataset_type in ['ecg', 'pd', 'gesture']:
        # ECG, PD, Gesture: Get all .pkl files from train directory
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for train_file in train_files:
            file_name = os.path.basename(train_file)
            test_file = os.path.join(config['test_dir'], file_name)
            if os.path.exists(test_file):
                files.append({
                    'name': file_name.replace('.pkl', ''),  # Remove extension for dataset_name
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'psm':
        # PSM: Single train/test file pair
        train_file = os.path.join(config['train_dir'], 'train.csv')
        test_file = os.path.join(config['test_dir'], 'test.csv')
        if os.path.exists(train_file) and os.path.exists(test_file):
            files.append({
                'name': 'psm_dataset',
                'train_path': train_file,
                'test_path': test_file
            })
    
    elif dataset_type == 'nab':
        # NAB: Require train/test/labels triplets
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(config['test_dir'], f"{file_name}_test.npy")
            labels_file = os.path.join(config['test_dir'], f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'smap_msl':
        # SMAP-MSL processed mode: iterate all *_train.npy triplets in processed dir
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], "*_train.npy")))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(config['test_dir'], f"{file_name}_test.npy")
            labels_file = os.path.join(config['test_dir'], f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'smd':
        # SMD: Require train/test/labels triplets in a single directory
        base_dir = config['train_dir']
        train_files = sorted(glob.glob(os.path.join(base_dir, "*_train.npy")))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(base_dir, f"{file_name}_test.npy")
            labels_file = os.path.join(base_dir, f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    elif dataset_type == 'ucr':
        # UCR: Require train/test/labels triplets in labeled dir
        train_files = sorted(glob.glob(os.path.join(config['train_dir'], config['file_pattern'])))
        for train_file in train_files:
            file_name = os.path.basename(train_file).replace('_train.npy', '')
            test_file = os.path.join(config['test_dir'], f"{file_name}_test.npy")
            labels_file = os.path.join(config['test_dir'], f"{file_name}_labels.npy")
            if os.path.exists(test_file) and os.path.exists(labels_file):
                files.append({
                    'name': file_name,
                    'train_path': train_file,
                    'test_path': test_file
                })
    
    return files

def process_dataset(dataset_type, config):
    """Process all files in a dataset type"""
    print(f"\n{'#'*80}")
    print(f"PROCESSING DATASET: {dataset_type.upper()}")
    print(f"Data path: {config['data_path']}")
    print(f"Output dir: {config['output_dir']}")
    print(f"{'#'*80}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Get files to process
    files = get_dataset_files(dataset_type, config)
    
    if not files:
        print(f"❌ No files found for dataset {dataset_type}")
        return False
    
    print(f"Found {len(files)} files to process:")
    for f in files:
        print(f"  - {f['name']}")
    
    success_count = 0
    
    for i, file_info in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE {i}/{len(files)}: {file_info['name']}")
        print(f"{'='*60}")
        
        # 1. Train model
        train_cmd = (
            f"python {os.path.join(BASE, 'src', 'model', 'main.py')} "
            f"--dataset {dataset_type} "
            f"--data_path {config['data_path']} "
            f"--dataset_name {file_info['name']} "
        )
        
        if not run_command(train_cmd, f"Training {file_info['name']}"):
            print(f"❌ Training failed for {file_info['name']}, skipping...")
            continue
        
        # 2. Run inference with the latest checkpoint
        # Find the checkpoint directory for this dataset
        checkpoints_dir = os.path.join(BASE, "checkpoints")
        if os.path.exists(checkpoints_dir):
            # Look for timestamped directories: {dataset}_{YYYYMMDD_HHMMSS}
            import re
            pattern = f"^{dataset_type}_\\d{{8}}_\\d{{6}}$"
            matching_dirs = []
            
            for item in os.listdir(checkpoints_dir):
                if os.path.isdir(os.path.join(checkpoints_dir, item)) and re.match(pattern, item):
                    matching_dirs.append(item)
            
            if matching_dirs:
                # Sort by timestamp (newest first)
                matching_dirs.sort(reverse=True)
                latest_dir = matching_dirs[0]
                
                # Check if best_model.pth exists
                model_path = os.path.join(checkpoints_dir, latest_dir, "best_model.pth")
                if os.path.exists(model_path):
                    print(f"🔍 Found latest checkpoint: {model_path}")
                    
                    # Run inference with the latest checkpoint
                    infer_cmd = (
                        f"python {os.path.join(BASE, 'src', 'inference', 'inference.py')} "
                        f"--dataset {dataset_type} "
                        f"--data_path {config['data_path']} "
                        f"--model_path {model_path} "
                        f"--dataset_name {file_info['name']} "
                        f"--save_excel "
                        f"--save_plot"
                    )
                    
                    if not run_command(infer_cmd, f"Inference {file_info['name']}"):
                        print(f"❌ Inference failed for {file_info['name']}")
                        continue
                else:
                    print(f"⚠️  No best_model.pth found in {latest_dir}, skipping inference...")
            else:
                print(f"⚠️  No timestamped checkpoints found for {dataset_type}, skipping inference...")
        else:
            print(f"⚠️  No checkpoints directory found, skipping inference...")
        
        success_count += 1
        print(f"✅ Successfully processed {file_info['name']}")
    
    print(f"\n{'='*60}")
    print(f"DATASET {dataset_type.upper()} COMPLETED")
    print(f"Successfully processed: {success_count}/{len(files)} files")
    print(f"{'='*60}")
    
    return success_count > 0

def run_inference_only():
    """Run inference only for all available checkpoints"""
    print("🔍 Running inference-only mode for all available checkpoints...")
    
    checkpoints_dir = os.path.join(BASE, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return False
    
    # Find all dataset types with checkpoints
    import re
    dataset_types = set()
    for item in os.listdir(checkpoints_dir):
        if os.path.isdir(os.path.join(checkpoints_dir, item)):
            # Extract dataset type from directory name
            match = re.match(r'^(\w+)_\d{8}_\d{6}$', item)
            if match:
                dataset_types.add(match.group(1))
    
    if not dataset_types:
        print("❌ No timestamped checkpoints found")
        return False
    
    print(f"Found checkpoints for datasets: {', '.join(sorted(dataset_types))}")
    
    success_count = 0
    
    for dataset_type in sorted(dataset_types):
        if dataset_type not in DATASET_CONFIGS:
            print(f"⚠️  Dataset {dataset_type} not configured, skipping...")
            continue
        
        config = DATASET_CONFIGS[dataset_type]
        
        # Find the checkpoint directory for this dataset
        checkpoints_dir = os.path.join(BASE, "checkpoints")
        model_path = None
        if os.path.exists(checkpoints_dir):
            # Look for timestamped directories: {dataset}_{YYYYMMDD_HHMMSS}
            import re
            pattern = f"^{dataset_type}_\\d{{8}}_\\d{{6}}$"
            matching_dirs = []
            
            for item in os.listdir(checkpoints_dir):
                if os.path.isdir(os.path.join(checkpoints_dir, item)) and re.match(pattern, item):
                    matching_dirs.append(item)
            
            if matching_dirs:
                # Sort by timestamp (newest first)
                matching_dirs.sort(reverse=True)
                latest_dir = matching_dirs[0]
                
                # Check if best_model.pth exists
                model_path = os.path.join(checkpoints_dir, latest_dir, "best_model.pth")
                if not os.path.exists(model_path):
                    model_path = None
        
        if not model_path:
            print(f"⚠️  No valid checkpoint found for {dataset_type}")
            continue
        
        print(f"\n{'='*60}")
        print(f"RUNNING INFERENCE FOR: {dataset_type.upper()}")
        print(f"Checkpoint: {model_path}")
        print(f"{'='*60}")
        
        # Get files to process
        files = get_dataset_files(dataset_type, config)
        if not files:
            print(f"❌ No files found for dataset {dataset_type}")
            continue
        
        print(f"Found {len(files)} files to process:")
        for f in files:
            print(f"  - {f['name']}")
        
        file_success = 0
        
        for file_info in files:
            # Run inference with the latest checkpoint
            infer_cmd = (
                f"python {os.path.join(BASE, 'src', 'inference', 'inference.py')} "
                f"--dataset {dataset_type} "
                f"--data_path {config['data_path']} "
                f"--model_path {model_path} "
                f"--dataset_name {file_info['name']} "
                f"--save_excel "
                f"--save_plot"
            )
            
            if run_command(infer_cmd, f"Inference {file_info['name']}"):
                file_success += 1
                print(f"✅ Successfully processed {file_info['name']}")
            else:
                print(f"❌ Inference failed for {file_info['name']}")
        
        if file_success > 0:
            success_count += 1
            print(f"✅ Dataset {dataset_type}: {file_success}/{len(files)} files processed")
        else:
            print(f"❌ Dataset {dataset_type}: No files processed successfully")
    
    print(f"\n{'='*60}")
    print(f"INFERENCE-ONLY COMPLETED")
    print(f"Successfully processed: {success_count}/{len(dataset_types)} dataset types")
    print(f"{'='*60}")
    
    return success_count > 0

def main():
    """Main function to process all datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated training and inference for all datasets')
    parser.add_argument('--inference-only', action='store_true', 
                       help='Run inference only for existing checkpoints (skip training)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Process only specific dataset type')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset paths without running training/inference')
    args = parser.parse_args()
    
    print("🚀 Starting automated training and inference for all datasets")
    print(f"Base directory: {BASE}")
    
    # Check if base directories exist
    if not os.path.exists(BASE):
        print(f"❌ Base directory not found: {BASE}")
        sys.exit(1)
    
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ Datasets directory not found: {DATASETS_DIR}")
        sys.exit(1)
    
    # Handle validate-only mode
    if args.validate_only:
        print("🔍 Running in validation-only mode...")
        datasets_to_validate = DATASET_CONFIGS
        if args.dataset:
            if args.dataset in DATASET_CONFIGS:
                datasets_to_validate = {args.dataset: DATASET_CONFIGS[args.dataset]}
                print(f"🎯 Validating only dataset: {args.dataset}")
            else:
                print(f"❌ Dataset {args.dataset} not found in configurations")
                sys.exit(1)
        
        if validate_dataset_paths(datasets_to_validate):
            print("🎉 All dataset paths are valid!")
            sys.exit(0)
        else:
            print("❌ Some dataset paths have issues!")
            sys.exit(1)
    
    # Validate paths before proceeding
    print("🔍 Validating dataset paths before processing...")
    if not validate_dataset_paths():
        print("❌ Dataset validation failed! Please fix the issues above.")
        sys.exit(1)
    
    # Handle inference-only mode
    if args.inference_only:
        print("🔍 Running in inference-only mode...")
        if run_inference_only():
            print("🎉 Inference completed successfully!")
        else:
            print("❌ Inference failed!")
        return
    
    # Process each dataset type
    total_success = 0
    datasets_to_process = DATASET_CONFIGS
    
    # Filter by specific dataset if requested
    if args.dataset:
        if args.dataset in DATASET_CONFIGS:
            datasets_to_process = {args.dataset: DATASET_CONFIGS[args.dataset]}
            print(f"🎯 Processing only dataset: {args.dataset}")
        else:
            print(f"❌ Dataset {args.dataset} not found in configurations")
            print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
            sys.exit(1)
    
    total_datasets = len(datasets_to_process)
    
    for dataset_type, config in datasets_to_process.items():
        try:
            if process_dataset(dataset_type, config):
                total_success += 1
        except Exception as e:
            print(f"❌ Error processing dataset {dataset_type}: {e}")
            continue
    
    print(f"\n{'#'*80}")
    print(f"FINAL SUMMARY")
    print(f"Successfully processed: {total_success}/{total_datasets} dataset types")
    print(f"{'#'*80}")
    
    if total_success == total_datasets:
        print("🎉 All datasets processed successfully!")
    else:
        print(f"⚠️  {total_datasets - total_success} dataset types failed to process")

if __name__ == '__main__':
    main()
