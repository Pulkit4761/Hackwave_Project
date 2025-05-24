import argparse
import os
import datetime
import json
import itertools
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import torch
import sys


# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


HP_RANGES = {
    # Core training parameters
    'epochs': [15],  
    'patience': [10], 
    'batch': [16, 32], 
    
    # Optimization parameters
    'optimizer': ['AdamW', 'SGD'], 
    'momentum': [0.85, 0.9, 0.95],  
    'weight_decay': [0.0, 0.001, 0.01],  
    'lr0': [0.001, 0.01],  # Initial learning rate
    'lrf': [0.01, 0.001],  # Final learning rate factor
    
    # Data augmentation parameters
    'augment': [True],  
    'mosaic': [0.5, 0.7, 0.9],  
    'mixup': [0.1, 0.3],  
    'copy_paste': [0.1],  
    
    # Geometric transformations
    'fliplr': [0.5],  # Horizontal flip probability
    'flipud': [0.0, 0.1],  # Vertical flip probability
    'degrees': [10.0, 20.0],  # Rotation (+/- deg)
    'translate': [0.1, 0.2],  # Translation (+/- fraction)
    'scale': [0.3, 0.5],  # Scale (+/- gain)
    'shear': [5.0, 10.0],  # Increased shear (+/- deg)
    'perspective': [0.0001, 0.001],  # Perspective distortion 
    
    # Color transformations
    'hsv_h': [0.015, 0.03],  # HSV-Hue augmentation
    'hsv_s': [0.5, 0.7],  # HSV-Saturation augmentation
    'hsv_v': [0.3, 0.5],  # HSV-Value augmentation
    
    # Advanced parameters
    'cache': [True],  
    'rect': [False],  # Rectangular training
    'cos_lr': [True],  
    'close_mosaic': [10],  
    'overlap_mask': [True],  # Mask overlapping objects
    'mask_ratio': [4],  # Mask downsampling ratio
    'dropout': [0.0, 0.1], 
    'single_cls': [False],  
}

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_data_directories(data_yaml_path):
    """Check if data directories exist."""
    try:
        import yaml
        with open(data_yaml_path, 'r') as file:
            data_config = yaml.safe_load(file)
        
        # Get directory paths
        train_dir = os.path.join(os.path.dirname(data_yaml_path), data_config.get('train', ''))
        val_dir = os.path.join(os.path.dirname(data_yaml_path), data_config.get('val', ''))
        test_dir = os.path.join(os.path.dirname(data_yaml_path), data_config.get('test', ''))
        
        # Verify directories exist
        for dir_path in [train_dir, val_dir, test_dir]:
            if dir_path and not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist!")
                return False
                
        return True
    except Exception as e:
        print(f"Error checking data directories: {e}")
        return False

def run_training(hyperparams, this_dir, run_id):
    """Run a single training with given hyperparameters."""
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select model based on dataset size and complexity
    model_path = os.path.join(this_dir, "yolov8s.pt")
    
    # Create model instance
    model = YOLO(model_path)
    
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data YAML path
    data_yaml_path = os.path.join(this_dir, "yolo_params.yaml")
    
    # Check data directories
    check_data_directories(data_yaml_path)
    
    # Set up training parameters
    train_params = {
        # Core training parameters
        'data': data_yaml_path,
        'epochs': hyperparams['epochs'],
        'patience': hyperparams.get('patience', 5),
        'batch': hyperparams.get('batch', 16),
        'imgsz': hyperparams.get('imgsz', 640),
        'device': device,  # Use CPU if CUDA is not available
        
        # Optimization parameters
        'optimizer': hyperparams['optimizer'],
        'momentum': hyperparams['momentum'],
        'weight_decay': hyperparams.get('weight_decay', 0.0),
        'lr0': hyperparams['lr0'],
        'lrf': hyperparams['lrf'],
        
        # Data augmentation
        'augment': hyperparams.get('augment', True),
        'mosaic': hyperparams['mosaic'],
        'mixup': hyperparams.get('mixup', 0.0),
        'copy_paste': hyperparams.get('copy_paste', 0.0),
        
        # Geometric transformations
        'fliplr': hyperparams.get('fliplr', 0.5),
        'flipud': hyperparams.get('flipud', 0.0),
        'degrees': hyperparams.get('degrees', 0.0),
        'translate': hyperparams.get('translate', 0.1),
        'scale': hyperparams.get('scale', 0.5),
        'shear': hyperparams.get('shear', 0.0),
        'perspective': hyperparams.get('perspective', 0.0),
        
        # Color transformations
        'hsv_h': hyperparams.get('hsv_h', 0.015),
        'hsv_s': hyperparams.get('hsv_s', 0.7),
        'hsv_v': hyperparams.get('hsv_v', 0.4),
        
        # Advanced parameters
        'cache': hyperparams.get('cache', False),
        'rect': hyperparams.get('rect', False),
        'cos_lr': hyperparams.get('cos_lr', False),
        'close_mosaic': hyperparams.get('close_mosaic', 0),
        'overlap_mask': hyperparams.get('overlap_mask', True),
        'mask_ratio': hyperparams.get('mask_ratio', 4),
        'dropout': hyperparams.get('dropout', 0.0),
        'single_cls': hyperparams.get('single_cls', False),
        
        # Output parameters
        'project': os.path.join(this_dir, "results"),
        'name': f"tuning_run_{run_id}",
        'exist_ok': True,
        'pretrained': True,
        'verbose': True
    }
    
    # Run training
    try:
        results = model.train(**train_params)
        
        # Get mAP50 from training results
        if hasattr(results, 'results_dict') and 'metrics/mAP50(B)' in results.results_dict:
            map50 = results.results_dict['metrics/mAP50(B)']
        else:
            # Try to get mAP50 from the last epoch's results
            try:
                map50 = round(results.metrics.get('map50', 0), 4)
            except (AttributeError, KeyError):
                map50 = 0.0
        
        if isinstance(map50, str):
            try:
                map50 = float(map50)
            except ValueError:
                map50 = 0.0
        
        # Create a results dictionary
        result_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "mAP50": map50,
            "hyperparameters": hyperparams,
            "metrics": results.metrics if hasattr(results, 'metrics') else {}
        }
        
        return result_data
    
    except Exception as e:
        print(f"Error in training run {run_id}: {e}")
        return {
            "run_id": run_id,
            "timestamp": timestamp,
            "mAP50": 0.0,
            "hyperparameters": hyperparams,
            "metrics": {},
            "error": str(e)
        }

def create_ensemble(best_models, this_dir, output_dir):
    """Create an ensemble model from the best models."""
    # Create ensemble directory
    ensemble_dir = os.path.join(output_dir, "ensemble")
    create_directory_if_not_exists(ensemble_dir)
    
    # Copy best models to ensemble directory
    model_paths = []
    for i, model_info in enumerate(best_models):
        run_id = model_info['run_id']
        source_path = os.path.join(this_dir, "results", f"tuning_run_{run_id}", "weights", "best.pt")
        if os.path.exists(source_path):
            dest_path = os.path.join(ensemble_dir, f"model_{i+1}.pt")
            shutil.copy(source_path, dest_path)
            model_paths.append(dest_path)
            print(f"Copied model {i+1} to ensemble: {dest_path}")
    
    # Create ensemble script
    ensemble_script = os.path.join(ensemble_dir, "ensemble_predict.py")
    with open(ensemble_script, 'w') as f:
        f.write('''
                
def ensemble_predict(model_paths, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run ensemble prediction on an image."""
    all_results = []
    
    # Run predictions with each model
    for model_path in model_paths:
        model = YOLO(model_path)
        results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
        all_results.append(results)
    
    # Return the ensemble results
    return all_results

if __name__ == "__main__":
    # Get image path from command line
    if len(sys.argv) < 2:
        print("Usage: python ensemble_predict.py <image_path> [conf_threshold] [iou_threshold]")
        sys.exit(1)
        
    image_path = sys.argv[1]
    conf_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    iou_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
    
    # Get all model paths in the current directory
    model_paths = [f for f in os.listdir('.') if f.endswith('.pt')]
    
    # Run ensemble prediction
    results = ensemble_predict(model_paths, image_path, conf_threshold, iou_threshold)
    
    # Print results
    for i, res in enumerate(results):
        print(f"Model {i+1} results:")
        for r in res:
            print(f"  Found {len(r.boxes)} objects")
            for box in r.boxes:
                print(f"    {box.cls} {box.conf:.2f} {box.xyxy}")
''')
    
    print(f"Created ensemble script: {ensemble_script}")
    print(f"To use ensemble: python {ensemble_script} <image_path> [conf_threshold] [iou_threshold]")
    
    return model_paths

def grid_search(param_ranges, max_combinations=None):
    """Generate all combinations of hyperparameters or a random subset."""
    keys = list(param_ranges.keys())
    values = list(param_ranges.values())
    
    all_combinations = list(itertools.product(*values))
    total_combinations = len(all_combinations)
    
    if max_combinations and max_combinations < total_combinations:
        print(f"Total possible combinations: {total_combinations}, selecting {max_combinations} random samples")
        selected_combinations = random.sample(all_combinations, max_combinations)
    else:
        print(f"Total combinations: {total_combinations}")
        selected_combinations = all_combinations
    
    # Convert combinations to dictionaries
    param_dicts = []
    for combo in selected_combinations:
        param_dict = {keys[i]: combo[i] for i in range(len(keys))}
        param_dicts.append(param_dict)
    
    return param_dicts

def smart_search(param_ranges, max_runs=20):
    """Smart search for hyperparameters based on known effective combinations."""
    # Define some effective baseline configurations
    base_configs = [
        # Config 1: High augmentation focus
        {
            'epochs': 15,  
            'mosaic': 0.9,
            'mixup': 0.3,
            'hsv_h': 0.03,
            'hsv_s': 0.7,
            'hsv_v': 0.5,
            'degrees': 20.0,
            'translate': 0.2,
            'scale': 0.5,
            'shear': 10.0,
            'fliplr': 0.5,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.001,
            'batch': 32,
            'imgsz': 640,
            'cos_lr': True,
            'close_mosaic': 10,  
            'patience': 10,  
        },
        # Config 2: Optimization focus
        {
            'epochs': 15,  
            'mosaic': 0.7,
            'mixup': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.3,
            'shear': 5.0,
            'fliplr': 0.5,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.95,
            'weight_decay': 0.01,
            'batch': 16,
            'imgsz': 640,
            'cos_lr': True,
            'close_mosaic': 10,  
            'patience': 10,  # Increased patience
        },
        # Config 3: Balanced approach
        {
            'epochs': 15,  
            'mosaic': 0.5,
            'mixup': 0.1,
            'copy_paste': 0.1,
            'hsv_h': 0.03,
            'hsv_s': 0.5,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 5.0,
            'fliplr': 0.5,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.001,
            'batch': 32,
            'imgsz': 640,
            'cos_lr': True,
            'close_mosaic': 10,  
            'patience': 10,  # Increased patience
        },
    ]
    
    # Create variations of base configs by modifying some parameters
    param_dicts = []
    for config in base_configs:
        param_dicts.append(config)  # Add the base config
        
        # Create variations by modifying one parameter at a time
        for param, values in param_ranges.items():
            if param in config and len(values) > 1:
                for value in values:
                    if value != config[param]:
                        # Create a new config with this parameter changed
                        new_config = config.copy()
                        new_config[param] = value
                        param_dicts.append(new_config)
                        
                        # Stop if we've reached the maximum number of runs
                        if len(param_dicts) >= max_runs:
                            break
                if len(param_dicts) >= max_runs:
                    break
        if len(param_dicts) >= max_runs:
            break
    
    # Adding some random combinations
    if len(param_dicts) < max_runs:
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        while len(param_dicts) < max_runs:
            # Generate a random combination
            random_combo = {}
            for i, key in enumerate(keys):
                random_combo[key] = random.choice(values[i])
            
            # Check if this combination is already in param_dicts
            if random_combo not in param_dicts:
                param_dicts.append(random_combo)
    
    # Limit to max_runs
    return param_dicts[:max_runs]

def plot_results(results_df, output_dir):
    """Create visualizations of hyperparameter tuning results."""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    create_directory_if_not_exists(plots_dir)
    
    # Plot mAP50 for each run
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(results_df)), results_df['mAP50'], color='skyblue')
    plt.xlabel('Run ID')
    plt.ylabel('mAP50')
    plt.title('mAP50 Across Hyperparameter Tuning Runs')
    plt.xticks(range(len(results_df)), results_df['run_id'], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'map50_by_run.png'))
    
    # Plot impact of key hyperparameters on mAP50
    numeric_params = [p for p in results_df.columns if p not in ['run_id', 'mAP50'] and 
                     results_df[p].dtype in [np.int64, np.float64, int, float]]
    
    if numeric_params:
        # Create subplots for numeric parameters
        num_rows = (len(numeric_params) + 1) // 2  # Calculate rows needed for 2 columns
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4*num_rows))
        axes = axes.flatten()
        
        for i, param in enumerate(numeric_params):
            if i < len(axes):
                axes[i].scatter(results_df[param], results_df['mAP50'])
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('mAP50')
                axes[i].set_title(f'Impact of {param} on mAP50')
                
                # Try to fit a trend line if we have numerical data
                try:
                    z = np.polyfit(results_df[param], results_df['mAP50'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(results_df[param]), max(results_df[param]), 100)
                    axes[i].plot(x_range, p(x_range), "r--")
                except Exception as e:
                    print(f"Could not fit trend line for {param}: {e}")
        
        # Turn off any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'hyperparameter_impact.png'))
    
    # For categorical parameters (like optimizer), create bar plots
    for param in results_df.columns:
        if param not in numeric_params and param not in ['run_id', 'mAP50']:
            try:
                plt.figure(figsize=(10, 6))
                grouped = results_df.groupby(param)['mAP50'].mean().reset_index()
                plt.bar(grouped[param], grouped['mAP50'], color='skyblue')
                plt.xlabel(param)
                plt.ylabel('Average mAP50')
                plt.title(f'Impact of {param} on mAP50')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{param}_impact.png'))
            except Exception as e:
                print(f"Could not create plot for {param}: {e}")
    
    # Create correlation heatmap
    try:
        plt.figure(figsize=(12, 10))
        numeric_df = results_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        plt.imshow(corr, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title('Hyperparameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
    except Exception as e:
        print(f"Could not create correlation matrix: {e}")
    
    plt.close('all')

def save_best_model(results_df, this_dir, output_dir):
    """Save the best model based on mAP50."""
    best_run = results_df.loc[results_df['mAP50'].idxmax()]
    best_run_id = best_run['run_id']
    best_map50 = best_run['mAP50']
    
    # Format timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create base filename
    base_filename = f"{timestamp}_mAP50_{best_map50:.4f}_best_from_tuning"
    
    # Path to best model from the tuning run
    best_model_path = os.path.join(this_dir, "results", f"tuning_run_{best_run_id}", "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        weights_filename = os.path.join(output_dir, "weights", f"{base_filename}_best.pt")
        shutil.copy(best_model_path, weights_filename)
        print(f"Best model weights saved to: {weights_filename}")
        
        # Save best hyperparameters to a separate file
        best_hp_file = os.path.join(output_dir, "logs", f"{base_filename}_best_hyperparams.json")
        with open(best_hp_file, 'w') as f:
            json.dump({
                "best_run_id": int(best_run_id),
                "mAP50": float(best_map50),
                "hyperparameters": {
                    key: (
                        bool(value) if key == 'single_cls' 
                        else float(value) if isinstance(value, (np.integer, np.floating)) 
                        else value
                    )
                    for key, value in best_run.items() 
                    if key in HP_RANGES.keys()
                }
            }, f, indent=4, cls=NumpyEncoder)
        print(f"Best hyperparameters saved to: {best_hp_file}")
        
        # Return path to best model
        return best_model_path
    else:
        print(f"Warning: Could not find best model for run {best_run_id} at {best_model_path}")
        return None

def create_ensemble_from_best_models(results_df, this_dir, output_dir, top_n=3):
    """Create an ensemble from the top N models."""
    # Sort by mAP50 and get top N runs
    top_runs = results_df.sort_values('mAP50', ascending=False).head(top_n)
    
    # Get information about the top runs
    top_models = []
    for _, row in top_runs.iterrows():
        run_id = row['run_id']
        map50 = row['mAP50']
        
        # Check if the model exists
        model_path = os.path.join(this_dir, "results", f"tuning_run_{run_id}", "weights", "best.pt")
        if os.path.exists(model_path):
            top_models.append({
                'run_id': run_id,
                'mAP50': map50,
                'path': model_path
            })
    
    # Create ensemble if we have at least 2 models
    if len(top_models) >= 2:
        print(f"\nCreating ensemble from {len(top_models)} best models...")
        model_paths = create_ensemble(top_models, this_dir, output_dir)
        return model_paths
    else:
        print("Not enough models to create an ensemble.")
        return []

def validate_test_set(best_model_path, test_data_path, output_dir):
    """Validate the best model on a test set."""
    if best_model_path and os.path.exists(best_model_path):
        print(f"\nValidating best model on test set: {test_data_path}")
        
        try:
            # Load the best model
            model = YOLO(best_model_path)
            
            # Run validation
            results = model.val(data=test_data_path)
            
            # Save validation results
            val_results = {
                "mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                "precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                "recall": float(results.results_dict.get('metrics/recall(B)', 0)),
                "f1": float(results.results_dict.get('metrics/F1(B)', 0)),
                "all_metrics": results.results_dict
            }
            
            # Save validation results to file
            val_file = os.path.join(output_dir, "logs", "validation_results.json")
            with open(val_file, 'w') as f:
                json.dump(val_results, f, indent=4)
                
            print(f"Validation results saved to: {val_file}")
            print(f"mAP50: {val_results['mAP50']:.4f}")
            print(f"Precision: {val_results['precision']:.4f}")
            print(f"Recall: {val_results['recall']:.4f}")
            print(f"F1 Score: {val_results['f1']:.4f}")
            
            return val_results
            
        except Exception as e:
            print(f"Error validating model: {e}")
            return None
    else:
        print("No best model found to validate.")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Hyperparameter tuning for YOLOv8")
    
    # Tuning method and limits
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'smart'], default='smart',
                        help='Tuning method: grid search, random search, or smart search')
    parser.add_argument('--max_runs', type=int, default=3,  
                        help='Maximum number of tuning runs to perform')
    parser.add_argument('--ensemble', action='store_true', default=True,
                        help='Create an ensemble of best models')
    parser.add_argument('--ensemble_size', type=int, default=3,
                        help='Number of models to include in ensemble')
    
    # Add optional validation dataset
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation dataset (YAML) for final evaluation')
    
    # Add option to continue from previous runs
    parser.add_argument('--continue_from', type=str, default=None,
                        help='Path to previous results JSON file to continue from')
    
    # Add option to enable CUDA benchmarking
    parser.add_argument('--cuda_benchmark', action='store_true', default=False,  
                        help='Enable CUDA benchmarking for faster training')
    
    # Add option to use test time augmentation
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Use test time augmentation for final evaluation')
    
    # Hyperparameter ranges (optional, can override defaults)
    for param, values in HP_RANGES.items():
        if isinstance(values[0], bool):
            parser.add_argument(f'--{param}', nargs='+', type=lambda x: x.lower() == 'true', 
                               default=values, help=f'Values for {param}')
        elif isinstance(values[0], float):
            parser.add_argument(f'--{param}', nargs='+', type=float, default=values, 
                               help=f'Values for {param}')
        elif isinstance(values[0], int):
            parser.add_argument(f'--{param}', nargs='+', type=int, default=values, 
                               help=f'Values for {param}')
        else:
            parser.add_argument(f'--{param}', nargs='+', type=str, default=values, 
                               help=f'Values for {param}')
    
    args = parser.parse_args()
    
    # Configure CUDA benchmarking 
    if args.cuda_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA benchmarking enabled.")
    
    # Get current directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is {'available' if cuda_available else 'not available'}, using {'GPU' if cuda_available else 'CPU'} for training")
    
    # Create directories for results
    output_dir = os.path.join(this_dir, "results")
    logs_dir = os.path.join(output_dir, "logs")
    weights_dir = os.path.join(output_dir, "weights")
    create_directory_if_not_exists(logs_dir)
    create_directory_if_not_exists(weights_dir)
    
    # Load previous results if continuing
    all_results = []
    if args.continue_from and os.path.exists(args.continue_from):
        with open(args.continue_from, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} previous results from {args.continue_from}")
    
    # Create parameter ranges dictionary from args
    param_ranges = {}
    for param in HP_RANGES.keys():
        param_values = getattr(args, param, None)
        if param_values is not None:
            param_ranges[param] = param_values
    
    # Generate parameter combinations
    if args.method == 'grid':
        param_combinations = grid_search(param_ranges)
    elif args.method == 'smart':
        param_combinations = smart_search(param_ranges, args.max_runs)
    else:  # random search
        param_combinations = grid_search(param_ranges, args.max_runs)
    
    # Limit the number of runs if needed
    if len(param_combinations) > args.max_runs:
        param_combinations = param_combinations[:args.max_runs]
    
    # Run hyperparameter tuning
    print(f"Starting hyperparameter tuning with {len(param_combinations)} runs...")
    start_idx = len(all_results) + 1
    
    for i, params in enumerate(param_combinations):
        run_id = start_idx + i
        print(f"\nRun {run_id}/{start_idx + len(param_combinations) - 1}")
        print(f"Parameters: {params}")
        
        # Run training with these parameters
        result = run_training(params, this_dir, run_id)
        all_results.append(result)
        
        # Save intermediate results
        intermediate_file = os.path.join(logs_dir, f"tuning_results_intermediate.json")
        with open(intermediate_file, 'w') as f:
            json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    
    # Process and analyze results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = os.path.join(logs_dir, f"tuning_results_{timestamp}.json")
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=4, cls=NumpyEncoder)
    print(f"All tuning results saved to: {final_results_file}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame([{
        'run_id': r['run_id'],
        'mAP50': float(r['mAP50']) if isinstance(r['mAP50'], (int, float)) or (isinstance(r['mAP50'], str) and r['mAP50'].replace('.', '', 1).isdigit()) else 0.0,
        **r['hyperparameters']
    } for r in all_results])
    
    # Generate plots
    if len(results_df) > 0:
        plot_results(results_df, output_dir)
        
        # Print results summary
        print("\n--- Hyperparameter Tuning Results ---")
        print(f"Total runs: {len(results_df)}")
        if len(results_df) > 0:
            best_idx = results_df['mAP50'].idxmax()
            print(f"Best mAP50: {results_df.loc[best_idx, 'mAP50']:.4f} (Run {results_df.loc[best_idx, 'run_id']})")
            print("Best hyperparameters:")
            for param in HP_RANGES.keys():
                if param in results_df.columns:
                    print(f"  {param}: {results_df.loc[best_idx, param]}")
                    
            # Save the best model
            best_model_path = save_best_model(results_df, this_dir, output_dir)
            
            # Create ensemble of best models if enabled
            if args.ensemble and len(results_df) >= args.ensemble_size:
                ensemble_paths = create_ensemble_from_best_models(results_df, this_dir, output_dir, args.ensemble_size)
                
            # Run validation on test set if provided
            if args.val_data and best_model_path:
                val_results = validate_test_set(best_model_path, args.val_data, output_dir)
                
                # If using TTA, run validation with TTA
                if args.tta and best_model_path:
                    print("\nValidating with Test Time Augmentation...")
                    try:
                        # Load the best model
                        model = YOLO(best_model_path)
                        
                        # Run validation with TTA
                        tta_results = model.val(data=args.val_data, augment=True)
                        
                        # Save TTA validation results
                        tta_val_results = {
                            "mAP50": float(tta_results.results_dict.get('metrics/mAP50(B)', 0)),
                            "precision": float(tta_results.results_dict.get('metrics/precision(B)', 0)),
                            "recall": float(tta_results.results_dict.get('metrics/recall(B)', 0)),
                            "f1": float(tta_results.results_dict.get('metrics/F1(B)', 0)),
                            "all_metrics": tta_results.results_dict
                        }
                        
                        # Save TTA validation results to file
                        tta_val_file = os.path.join(output_dir, "logs", "tta_validation_results.json")
                        with open(tta_val_file, 'w') as f:
                            json.dump(tta_val_results, f, indent=4)
                            
                        print(f"TTA Validation results saved to: {tta_val_file}")
                        print(f"TTA mAP50: {tta_val_results['mAP50']:.4f}")
                        print(f"TTA Precision: {tta_val_results['precision']:.4f}")
                        print(f"TTA Recall: {tta_val_results['recall']:.4f}")
                        print(f"TTA F1 Score: {tta_val_results['f1']:.4f}")
                        
                    except Exception as e:
                        print(f"Error validating model with TTA: {e}")
    else:
        print("No valid results found.")
        
    print("\nHyperparameter tuning complete!")