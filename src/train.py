import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from typing import Dict, List, Any
import time
import json
from datetime import datetime

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU devices")
    try:
        # Configure TensorFlow to use all available GPU memory
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Set TensorFlow to use the GPU for operations
        tf.config.set_visible_devices(physical_devices, 'GPU')
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Import project modules
try:
    import config
    from data_loader import CSVDataLoader, DataManager
    from filters import DataFilterProcessor
    from model import ModelFactory
    from evaluate import ModelEvaluator, ModelPerformanceTracker
except ModuleNotFoundError:
    print("Error: Make sure you are running train.py from the project root directory"
          " or have the 'src' directory in your PYTHONPATH.")
    # Attempt relative imports if run directly within src (less ideal)
    try:
        import config
        from .data_loader import CSVDataLoader, DataManager
        from .filters import DataFilterProcessor
        from .model import ModelFactory
        from .evaluate import ModelEvaluator, ModelPerformanceTracker
    except (ImportError, ModuleNotFoundError):
        print("Failed to import necessary modules. Exiting.")
        exit(1)

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
        
        if self.save_models:
            os.makedirs(self.models_dir, exist_ok=True)
            print(f"Models will be saved to: {self.models_dir}")
        
        # Set memory growth for GPUs if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.using_gpu = True
                print(f"Using {len(gpus)} GPU(s) with memory growth enabled")
                
                # Print GPU device information
                for i, gpu in enumerate(gpus):
                    details = tf.config.experimental.get_device_details(gpu)
                    if details and 'device_name' in details:
                        print(f"  GPU {i}: {details['device_name']}")
                    else:
                        print(f"  GPU {i}: Unknown")
            except RuntimeError as e:
                print(f"GPU memory growth setting error: {e}")
                self.using_gpu = False
        else:
            self.using_gpu = False
            print("No GPU available. Using CPU.")

    def run(self) -> pd.DataFrame:
        """Run the complete experiment"""
        print("--- Starting Filter Comparison Experiment ---")

        # --- 1. Load Raw Data ---
        try:
            raw_df = self.data_loader.load()
        except FileNotFoundError as e:
            print(e)
            return pd.DataFrame()  # Return empty DataFrame if data not found

        # --- 2. Loop Through Filters ---
        for filter_type in config.FILTERS_TO_COMPARE:
            print(f"\n{'='*20} Testing Filter: {filter_type.upper()} {'='*20}")
            
            try:
                result, detailed_report = self._run_single_experiment(raw_df, filter_type)
                if result:
                    self.results.append(result)
                    self.detailed_reports.append(detailed_report)
            except Exception as e:
                print(f"Error in experiment with filter {filter_type}: {e}")
                continue
        
        # Find best model based on RMSE
        if self.detailed_reports:
            best_idx, best_report = self.evaluator.get_best_model(
                self.detailed_reports, primary_metric="RMSE"
            )
            print(f"\n🏆 Best model: {best_report['filter_type']} filter (RMSE: {best_report['metrics']['RMSE']:.6f})")
            
            # Print top 3 models if we have at least 3
            if len(self.detailed_reports) >= 3:
                # Sort by RMSE (lower is better)
                sorted_reports = sorted(self.detailed_reports, 
                                     key=lambda x: x['metrics'].get('RMSE', float('inf')))
                
                print("\nTop 3 models by RMSE:")
                for i, report in enumerate(sorted_reports[:3]):
                    print(f"  {i+1}. {report['filter_type']} (RMSE: {report['metrics']['RMSE']:.6f})")

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
            print(f"Skipping filter {filter_type} due to preprocessing error: {e}")
            return None, None

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"Skipping filter {filter_type} due to insufficient data after preprocessing.")
            return None, None
            
        # Calculate optimal batch size (power of 2, based on dataset size)
        batch_size = min(self._get_optimal_batch_size(X_train.shape[0]), config.BATCH_SIZE)
        print(f"Using batch size: {batch_size}")

        # --- 5. Build and Train Model ---
        input_shape = (X_train.shape[1], X_train.shape[2])
        tf.keras.backend.clear_session()
        
        # Create model with optimized architecture
        model_obj = ModelFactory.create_lstm_model(
            input_shape=input_shape,
            lstm_units=(64, 32),  # Default values, can be tuned
            batch_norm=True,
            dropout_rate=0.3 if X_train.shape[0] < 5000 else 0.2  # Adjust dropout based on data size
        )
        model = model_obj.build()

        # Setup performance tracking
        performance_tracker = ModelPerformanceTracker(self.evaluator)

        # Get training callbacks including early stopping, LR reduction, etc.
        callbacks = model_obj.get_callbacks(patience_early_stop=10, patience_lr=5)

        # Confirm that TensorFlow is configured to use GPU
        if self.using_gpu:
            print("\nGPU is enabled for training and prediction")
            # Show which device operations are being placed on
            tf_device = tf.config.list_logical_devices('GPU')[0].name if tf.config.list_logical_devices('GPU') else 'CPU'
            print(f"TensorFlow is using device: {tf_device}")
        else:
            print("\nWARNING: Using CPU for training - this will be much slower")

        print(f"\nTraining model with {filter_type} filtered data...")
        
        # Use the TensorFlow device context to ensure GPU usage
        with tf.device('/GPU:0' if self.using_gpu else '/CPU:0'):
            history = performance_tracker.track_training(
                model.fit,
                X_train, y_train,
                epochs=config.EPOCHS,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # --- 6. Evaluate Model ---
            print(f"\nEvaluating model ({filter_type} filter) on test data...")
            y_pred, inference_time_per_sample = performance_tracker.track_inference(
                model.predict, X_test, batch_size=batch_size*2  # Larger batch size for inference
            )

        # Flatten predictions if needed
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Generate detailed evaluation report
        detailed_report = performance_tracker.generate_evaluation_report(y_test, y_pred, filter_type)
        
        # Calculate total experiment time
        total_time = time.time() - start_time
        detailed_report['total_experiment_time'] = total_time
        
        # Save model if requested
        if self.save_models:
            model_path = os.path.join(self.models_dir, f"{filter_type}_model")
            try:
                model_obj.save(model_path)
                detailed_report['model_path'] = model_path
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Summary metrics for simple results table
        result = {
            'filter_type': filter_type,
            'mape (%)': detailed_report['metrics']['MAPE'],
            'r2_score': detailed_report['metrics']['R2'],
            'mae': detailed_report['metrics']['MAE'],
            'mse': detailed_report['metrics']['MSE'],
            'rmse': detailed_report['metrics']['RMSE'],
            'correlation': detailed_report['metrics'].get('Corr', float('nan')),
            'training_time (s)': detailed_report['metrics']['training_time'],
            'inference_time_per_sample (s)': inference_time_per_sample,
            'total_inference_time (s)': detailed_report['metrics']['inference_time'],
            'epochs_run': len(history.history['loss']),
            'batch_size': batch_size,
            'total_experiment_time (s)': total_time,
            'gpu_used': self.using_gpu,
            'samples_within_5percent': detailed_report['error_stats']['samples_within_5percent'],
            'samples_within_10percent': detailed_report['error_stats']['samples_within_10percent']
        }

        return result, detailed_report
        
    def _get_optimal_batch_size(self, dataset_size: int) -> int:
        """Calculate optimal batch size (power of 2) based on dataset size"""
        # Start with minimum batch size of 16
        batch_size = 16
        
        # Increase batch size to at most 1/10 of dataset size
        max_size = max(16, min(256, int(dataset_size / 10)))
        
        # Find the largest power of 2 that's less than max_size
        while batch_size * 2 <= max_size:
            batch_size *= 2
            
        return batch_size

    def _save_results(self) -> pd.DataFrame:
        """Save experiment results to CSV and JSON"""
        print("\n--- Experiment Finished ---")
        if not self.results:
            print("No results generated.")
            return pd.DataFrame()

        results_df = pd.DataFrame(self.results)
        print("\nComparison Results:")
        print(results_df)

        # Ensure results directory exists
        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        try:
            # Save summary results to CSV
            results_df.to_csv(config.RESULTS_FILE, index=False)
            print(f"\nSummary results saved to: {config.RESULTS_FILE}")
            
            # Save detailed reports to JSON
            detailed_results_file = os.path.join(
                config.RESULTS_DIR, 
                f"detailed_results_{self.timestamp}.json"
            )
            
            with open(detailed_results_file, 'w') as f:
                json.dump(self.detailed_reports, f, indent=2)
            print(f"Detailed results saved to: {detailed_results_file}")
            
        except IOError as e:
            print(f"Error saving results: {e}")

        return results_df

def main():
    # Suppress TensorFlow INFO messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='AI Filtering Comparison Training')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--filters', type=str, help='Comma-separated list of filters to use')
    args = parser.parse_args()
    
    # Update config if filters specified
    if args.filters:
        filters = [f.strip() for f in args.filters.split(',')]
        valid_filters = [f for f in filters if f in config.FILTERS_TO_COMPARE]
        if valid_filters:
            config.FILTERS_TO_COMPARE = valid_filters
            print(f"Using filters: {valid_filters}")
        else:
            print(f"No valid filters specified. Using all available filters.")

    # Run experiment
    experiment = ExperimentRunner(
        data_file=args.data if args.data else config.DATA_FILE,
        save_models=args.save_models
    )
    experiment.run()

if __name__ == "__main__":
    main() 