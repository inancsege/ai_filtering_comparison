#!/usr/bin/env python3
"""
AI Filtering Comparison - Main Entry Point

This script provides entry points and command-line interface for the AI filtering comparison tools.
It contains the core functionality that was previously in src/main.py, which has been consolidated
here to simplify the project structure while maintaining all functionality.

The project structure now uses this file as the main application entry point, with supporting
modules and utilities still in the src/ directory.
"""

import os
import sys
import argparse
import traceback
import logging
import time
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import standard scikit-learn modules (will be used as fallback)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

# Import project modules
try:
    from src import config
    from src.data_loader import CSVDataLoader, DataManager
    from src.filters import DataFilterProcessor
    from src.utils import (
        timer, cache_result, optimize_dataframe, process_in_chunks,
        clean_memory, memory_usage, setup_gpu, logger, get_system_info
    )
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running main.py from the project root directory.")
    sys.exit(1)

# Check for GPU support - moved to utils.setup_gpu()
USE_GPU = config.USE_GPU
GPU_TYPE = None

if USE_GPU:
    GPU_TYPE = setup_gpu()
    if not GPU_TYPE:
        USE_GPU = False
        logger.warning("GPU acceleration requested but no compatible libraries found. Using CPU.")

# Import GPU-specific libraries if GPU is available
if USE_GPU and GPU_TYPE == "NVIDIA":
    try:
        import cudf
        import cuml
        from cuml.ensemble import RandomForestRegressor as cuRFR
        from cuml.linear_model import LinearRegression as cuLR
        from cuml.svm import SVR as cuSVR
    except ImportError:
        logger.warning("Failed to import NVIDIA GPU libraries. Falling back to CPU.")
        USE_GPU = False
        GPU_TYPE = None

class ModelFactory:
    """Factory class for creating models with GPU support when available"""
    @staticmethod
    def create_model(model_type="random_forest", **kwargs):
        """Create a model based on the specified type, with GPU support if available"""
        if USE_GPU:
            if GPU_TYPE == "NVIDIA":
                # Create RAPIDS cuML model for NVIDIA GPUs
                if model_type == "random_forest":
                    return cuRFR(
                        n_estimators=kwargs.get("n_estimators", 100),
                        max_depth=kwargs.get("max_depth", 10),
                        random_state=42
                    )
                elif model_type == "linear_regression":
                    return cuLR()
                elif model_type == "svr":
                    return cuSVR(
                        kernel=kwargs.get("kernel", "rbf"),
                        C=kwargs.get("C", 1.0),
                        epsilon=kwargs.get("epsilon", 0.1)
                    )
                else:
                    # Fall back to random forest if unsupported model type
                    logger.warning(f"Model type '{model_type}' not supported on GPU. Using Random Forest instead.")
                    return cuRFR(n_estimators=100, max_depth=10, random_state=42)
            else:
                # For Intel or other GPU types, use patched scikit-learn
                # sklearnex automatically uses optimized versions when patched
                pass  # Continue to sklearn implementations which will use Intel optimizations
                
        # Standard scikit-learn models (CPU or with Intel GPU optimization via sklearnex)
        if model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                random_state=42,
                n_jobs=config.NUM_CORES  # Use configured number of cores
            )
        elif model_type == "linear_regression":
            return LinearRegression(n_jobs=config.NUM_CORES)
        elif model_type == "svr":
            return SVR(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                epsilon=kwargs.get("epsilon", 0.1)
            )
        elif model_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (100, 50)),
                max_iter=kwargs.get("max_iter", 1000),
                random_state=42
            )
        else:
            # Default to Random Forest
            logger.warning(f"Unknown model type '{model_type}'. Using Random Forest as default.")
            return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=config.NUM_CORES)

class ModelEvaluator:
    """Class for evaluating model performance"""
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        # Convert to CPU numpy arrays if using GPU
        if USE_GPU and GPU_TYPE == "NVIDIA":
            if hasattr(y_true, 'to_numpy'):
                y_true = y_true.to_numpy()
            if hasattr(y_pred, 'to_numpy'):
                y_pred = y_pred.to_numpy()
                
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE with handling for zero values
        mask = y_true != 0
        if np.any(mask):
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = float('nan')
            
        # Calculate correlation
        try:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        except:
            correlation = float('nan')
            
        # Calculate percentage of predictions within error ranges
        rel_errors = np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))
        within_5percent = np.mean(rel_errors <= 0.05) * 100
        within_10percent = np.mean(rel_errors <= 0.1) * 100
        
        return {
            "RMSE": rmse,
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "MAPE": mape,
            "Corr": correlation,
            "samples_within_5percent": within_5percent,
            "samples_within_10percent": within_10percent
        }
        
    def get_best_model(self, results_list, primary_metric="RMSE"):
        """
        Determine the best model based on specified metric
        
        Args:
            results_list: List of result dictionaries from different models
            primary_metric: Metric to use for ranking (default: RMSE)
            
        Returns:
            Tuple of (best_index, best_result)
        """
        if not results_list:
            return -1, {}
            
        # Determine if higher is better
        higher_is_better = primary_metric in ["R2", "Corr", "samples_within_5percent", "samples_within_10percent"]
        
        # Extract the metric values
        metric_values = []
        for r in results_list:
            metrics = r.get('metrics', {})
            value = metrics.get(primary_metric, float('-inf') if higher_is_better else float('inf'))
            metric_values.append(value)
        
        # Find the best index
        if higher_is_better:
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)
            
        return best_idx, results_list[best_idx]

class ExperimentRunner:
    """Class responsible for running filter comparison experiments"""
    def __init__(self, data_file: str = config.DATA_FILE, save_models: bool = False):
        self.data_file = data_file
        self.data_loader = CSVDataLoader(data_file)
        self.data_manager = DataManager(self.data_loader)
        self.filter_processor = DataFilterProcessor()
        self.evaluator = ModelEvaluator()
        self.results: List[Dict[str, Any]] = []
        self.detailed_reports: List[Dict[str, Any]] = []
        self.save_models = save_models
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.models_dir = os.path.join(config.RESULTS_DIR, 'models', self.timestamp)
        
        # Print system info
        self._print_system_info()
        
        if self.save_models:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"Models will be saved to: {self.models_dir}")

    def _print_system_info(self):
        """Print information about the system and available accelerators"""
        logger.info("\n=== System Information ===")
        sys_info = get_system_info()
        for key, value in sys_info.items():
            logger.info(f"{key}: {value}")
        logger.info("==============================\n")

    def run(self) -> pd.DataFrame:
        """Run the complete experiment"""
        logger.info("--- Starting Filter Comparison Experiment ---")

        # --- 1. Load Raw Data ---
        try:
            raw_df = self.data_loader.load()
        except FileNotFoundError as e:
            logger.error(e)
            return pd.DataFrame()  # Return empty DataFrame if data not found

        # --- 2. Loop Through Filters ---
        for filter_type in config.FILTERS_TO_COMPARE:
            logger.info(f"\n{'='*20} Testing Filter: {filter_type.upper()} {'='*20}")
            
            try:
                result, detailed_report = self._run_single_experiment(raw_df, filter_type)
                if result:
                    self.results.append(result)
                    self.detailed_reports.append(detailed_report)
            except Exception as e:
                logger.error(f"Error in experiment with filter {filter_type}: {e}")
                continue
        
        # Find best model based on RMSE
        if self.detailed_reports:
            best_idx, best_report = self.evaluator.get_best_model(
                self.detailed_reports, primary_metric="RMSE"
            )
            logger.info(f"\n🏆 Best model: {best_report['filter_type']} filter (RMSE: {best_report['metrics']['RMSE']:.6f})")
            
            # Print top 3 models if we have at least 3
            if len(self.detailed_reports) >= 3:
                # Sort by RMSE (lower is better)
                sorted_reports = sorted(self.detailed_reports, 
                                     key=lambda x: x['metrics'].get('RMSE', float('inf')))
                
                logger.info("\nTop 3 models by RMSE:")
                for i, report in enumerate(sorted_reports[:3]):
                    logger.info(f"  {i+1}. {report['filter_type']} (RMSE: {report['metrics']['RMSE']:.6f})")

        return self._save_results()

    def _run_single_experiment(self, raw_df: pd.DataFrame, filter_type: str) -> tuple:
        """Run experiment for a single filter type"""
        start_time = time.time()
        
        # --- 3. Apply Filter ---
        df_filtered = self.filter_processor.process(raw_df.copy(), filter_type)

        # --- 4. Preprocess Data ---
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = self.data_manager.preprocessor.preprocess(df_filtered)
        except ValueError as e:
            logger.error(f"Skipping filter {filter_type} due to preprocessing error: {e}")
            return None, None

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            logger.error(f"Skipping filter {filter_type} due to insufficient data after preprocessing.")
            return None, None
            
        # --- 5. Build and Train Model ---
        # Reshape data for sklearn models (flatten sequence data if needed)
        X_train_2d = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        X_val_2d = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val
        X_test_2d = X_test.reshape(X_test.shape[0], -1) if len(X_test.shape) > 2 else X_test
        
        logger.info(f"Training data shape after preprocessing: {X_train_2d.shape}")
        
        # Convert data to GPU format if using NVIDIA GPU
        # Use the global USE_GPU variable
        global USE_GPU, GPU_TYPE
        
        if USE_GPU and GPU_TYPE == "NVIDIA":
            try:
                import cudf
                # Only convert to GPU if data size is reasonable
                if X_train_2d.shape[0] * X_train_2d.shape[1] < 1e8:  # Size check to avoid OOM
                    X_train_2d = cudf.DataFrame(X_train_2d)
                    y_train = cudf.Series(y_train)
                    X_test_2d = cudf.DataFrame(X_test_2d)
                    y_test = cudf.Series(y_test)
                    logger.info("Data converted to GPU format")
                else:
                    logger.info("Data too large for GPU memory, using CPU")
                    USE_GPU = False
            except Exception as e:
                logger.error(f"Error converting data to GPU format: {e}")
                logger.info("Falling back to CPU processing")
                USE_GPU = False
        
        # Choose model type based on data size
        if X_train_2d.shape[0] > 5000:
            # Use linear regression for large datasets
            model_name = "linear_regression"
        elif X_train_2d.shape[0] > 1000:
            # Use SVR for medium datasets if not using GPU (SVR can be slow on large datasets)
            if USE_GPU and GPU_TYPE == "NVIDIA":
                model_name = "linear_regression"  # cuSVR might still be too slow
            else:
                model_name = "svr"
        else:
            # Use Random Forest for smaller datasets
            model_name = "random_forest"
            
        logger.info(f"Using {model_name} model for {filter_type} filter")
        
        # Create model
        model = ModelFactory.create_model(model_name)

        # Train model
        logger.info(f"\nTraining model with {filter_type} filtered data...")
        train_start_time = time.time()
        model.fit(X_train_2d, y_train)
        training_time = time.time() - train_start_time
        
        # --- 6. Evaluate Model ---
        logger.info(f"\nEvaluating model ({filter_type} filter) on test data...")
        inference_start_time = time.time()
        y_pred = model.predict(X_test_2d)
        inference_time = time.time() - inference_start_time
        inference_time_per_sample = inference_time / len(X_test)
        
        # Generate evaluation report
        metrics = self.evaluator.evaluate(y_test, y_pred)
        metrics["training_time"] = training_time
        metrics["inference_time"] = inference_time
        
        # Calculate error statistics
        if USE_GPU and GPU_TYPE == "NVIDIA":
            # Convert to CPU for calculations
            if hasattr(y_test, 'to_numpy'):
                y_test_cpu = y_test.to_numpy()
                y_pred_cpu = y_pred.to_numpy()
            else:
                y_test_cpu = y_test
                y_pred_cpu = y_pred
            
            errors = y_test_cpu - y_pred_cpu
            abs_errors = np.abs(errors)
        else:
            errors = y_test - y_pred
            abs_errors = np.abs(errors)
        
        error_stats = {
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'abs_error_mean': float(np.mean(abs_errors)),
            'abs_error_median': float(np.median(abs_errors)),
            'error_max': float(np.max(abs_errors)),
            'error_min': float(np.min(abs_errors)),
            'error_q1': float(np.percentile(abs_errors, 25)),
            'error_q3': float(np.percentile(abs_errors, 75)),
            'samples_within_5percent': metrics["samples_within_5percent"],
            'samples_within_10percent': metrics["samples_within_10percent"],
        }
        
        # Detailed report
        detailed_report = {
            'filter_type': filter_type,
            'metrics': metrics,
            'error_stats': error_stats,
            'sample_count': len(y_test),
            'model_type': model_name,
            'gpu_used': USE_GPU,
            'gpu_type': GPU_TYPE if USE_GPU else None
        }
        
        # Calculate total experiment time
        total_time = time.time() - start_time
        detailed_report['total_experiment_time'] = total_time
        
        # Save model if requested
        if self.save_models:
            model_path = os.path.join(self.models_dir, f"{filter_type}_{model_name}_model.joblib")
            try:
                # Move model to CPU before saving if it's a GPU model
                if USE_GPU and GPU_TYPE == "NVIDIA" and hasattr(model, 'to_cpu'):
                    cpu_model = model.to_cpu()
                    joblib.dump(cpu_model, model_path)
                else:
                    joblib.dump(model, model_path)
                detailed_report['model_path'] = model_path
                logger.info(f"Model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
        
        # Summary metrics for simple results table
        result = {
            'filter_type': filter_type,
            'mape (%)': metrics['MAPE'],
            'r2_score': metrics['R2'],
            'mae': metrics['MAE'],
            'mse': metrics['MSE'],
            'rmse': metrics['RMSE'],
            'correlation': metrics.get('Corr', float('nan')),
            'training_time (s)': training_time,
            'inference_time_per_sample (s)': inference_time_per_sample,
            'total_inference_time (s)': inference_time,
            'model_type': model_name,
            'total_experiment_time (s)': total_time,
            'samples_within_5percent': metrics["samples_within_5percent"],
            'samples_within_10percent': metrics["samples_within_10percent"],
            'gpu_used': USE_GPU
        }

        return result, detailed_report

    def _save_results(self) -> pd.DataFrame:
        """Save experiment results to CSV and JSON"""
        logger.info("\n--- Experiment Finished ---")
        if not self.results:
            logger.warning("No results generated.")
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        logger.info("\nComparison Results:")
        logger.info(results_df)

        # Ensure results directory exists
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        try:
            # Save summary results to CSV
            results_df.to_csv(config.RESULTS_FILE, index=False)
            logger.info(f"\nSummary results saved to: {config.RESULTS_FILE}")
            
            # Save detailed reports to JSON
            detailed_results_file = os.path.join(
                config.RESULTS_DIR, 
                f"detailed_results_{self.timestamp}.json"
            )
            
            with open(detailed_results_file, 'w') as f:
                json.dump(self.detailed_reports, f, indent=2)
            logger.info(f"Detailed results saved to: {detailed_results_file}")
            
        except IOError as e:
            logger.error(f"Error saving results: {e}")

        return results_df

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Filtering Comparison Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data options
    parser.add_argument('--data', type=str, help='Path to input data file (CSV)')
    parser.add_argument('--output', type=str, help='Path to save results')
    
    # Filter options
    parser.add_argument('--filters', type=str, 
                        help='Comma-separated list of filters to compare (e.g., "savitzky_golay,moving_average,kalman")')
    
    # Performance options
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--chunk-size', type=int, help='Chunk size for processing large datasets')
    parser.add_argument('--jobs', type=int, help='Number of parallel jobs to run')
    
    # Model options
    parser.add_argument('--save-models', action='store_true', help='Save trained models to disk')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'linear_regression', 'svr', 'mlp'],
                        help='Model type to use for comparison')
    parser.add_argument('--sequence-length', type=int, help='Sequence length for time series analysis')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    
    # Debug options
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional info')
    parser.add_argument('--system-info', action='store_true', help='Display system information and exit')
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Setup environment based on command line arguments"""
    # Create config overrides dictionary
    overrides = {}
    
    # Data paths
    if args.data:
        overrides['DATA_FILE'] = os.path.abspath(args.data)
    
    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        overrides['RESULTS_DIR'] = output_dir
        overrides['RESULTS_FILE'] = os.path.join(output_dir, 'comparison_results.csv')
    
    # Filter options
    if args.filters:
        filters = [f.strip() for f in args.filters.split(',')]
        overrides['FILTERS_TO_COMPARE'] = filters
    
    # Performance options
    if args.no_gpu:
        overrides['USE_GPU'] = False
    
    if args.no_cache:
        overrides['USE_CACHE'] = False
    
    if args.chunk_size:
        overrides['CHUNK_SIZE'] = args.chunk_size
    
    if args.jobs:
        overrides['NUM_CORES'] = args.jobs
    
    # Model options
    if args.sequence_length:
        overrides['SEQUENCE_LENGTH'] = args.sequence_length
    
    if args.batch_size:
        overrides['BATCH_SIZE'] = args.batch_size
    
    if args.epochs:
        overrides['EPOCHS'] = args.epochs
    
    # Logging options
    if args.verbose:
        overrides['VERBOSE'] = True
        logger.setLevel(logging.DEBUG)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Enable debug mode for all libraries
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        # Set reasonable log levels for verbose libraries
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Apply overrides
    for key, value in overrides.items():
        setattr(config, key, value)
        logger.info(f"Config override: {key} = {value}")
    
    return overrides

def run_main(data_file=None, save_models=False, model_type="random_forest", disable_gpu=False, **kwargs):
    """
    Main function to run experiments comparing different filtering techniques
    
    Args:
        data_file: Path to input data file (overrides config)
        save_models: Whether to save trained models
        model_type: Type of model to use
        disable_gpu: Disable GPU acceleration
        **kwargs: Additional overrides passed from command line
    """
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Display system information
    if config.VERBOSE:
        sys_info = get_system_info()
        logger.info("System Information:")
        for key, value in sys_info.items():
            logger.info(f"  {key}: {value}")
    
    # Override USE_GPU if specified
    global USE_GPU, GPU_TYPE
    if disable_gpu:
        USE_GPU = False
        GPU_TYPE = None
        logger.info("GPU acceleration disabled by user.")
    
    # Set the data file if provided
    if data_file:
        config.DATA_FILE = data_file
        logger.info(f"Using data file: {data_file}")
    
    # Run the experiment
    with timer("Full experiment"):
        experiment = ExperimentRunner(data_file=data_file, save_models=save_models)
        result_df = experiment.run()
    
    # Final cleanup
    clean_memory()
    
    return result_df

def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Display system info if requested
    if args.system_info:
        info = get_system_info()
        print("\nSystem Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        return 0
    
    try:
        # Setup environment with command line args
        overrides = setup_environment(args)
        
        # Run the main function with overrides
        run_main(
            data_file=args.data,
            save_models=args.save_models,
            model_type=args.model_type,
            disable_gpu=args.no_gpu,
            **overrides
        )
        
        return 0  # Success
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.debug:
            logger.error(traceback.format_exc())
        return 1  # Failure

if __name__ == "__main__":
    # Run the main function and exit with its return code
    sys.exit(main()) 