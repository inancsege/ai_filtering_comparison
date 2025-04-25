import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from typing import Dict, List, Any

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
    def __init__(self, data_file: str = config.DATA_FILE):
        self.data_file = data_file
        self.data_loader = CSVDataLoader(data_file)
        self.data_manager = DataManager(self.data_loader)
        self.filter_processor = DataFilterProcessor()
        self.evaluator = ModelEvaluator()
        self.results: List[Dict[str, Any]] = []

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
                result = self._run_single_experiment(raw_df, filter_type)
                if result:
                    self.results.append(result)
            except Exception as e:
                print(f"Error in experiment with filter {filter_type}: {e}")
                continue

        return self._save_results()

    def _run_single_experiment(self, raw_df: pd.DataFrame, filter_type: str) -> Dict[str, Any]:
        """Run experiment for a single filter type"""
        # --- 3. Apply Filter ---
        df_filtered = self.filter_processor.process(raw_df.copy(), filter_type)

        # --- 4. Preprocess Data ---
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = self.data_manager.preprocessor.preprocess(df_filtered)
        except ValueError as e:
            print(f"Skipping filter {filter_type} due to preprocessing error: {e}")
            return None

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            print(f"Skipping filter {filter_type} due to insufficient data after preprocessing.")
            return None

        # --- 5. Build and Train Model ---
        input_shape = (X_train.shape[1], X_train.shape[2])
        tf.keras.backend.clear_session()
        model = ModelFactory.create_lstm_model(input_shape)
        model = model.build()

        # Setup performance tracking
        performance_tracker = ModelPerformanceTracker(self.evaluator)

        # Training with early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        print(f"\nTraining model with {filter_type} filtered data...")
        history = performance_tracker.track_training(
            model.fit,
            X_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # --- 6. Evaluate Model ---
        print(f"\nEvaluating model ({filter_type} filter) on test data...")
        y_pred, inference_time_per_sample = performance_tracker.track_inference(
            model.predict, X_test
        )

        # Flatten predictions if needed
        if y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()

        # Calculate all metrics
        metrics = performance_tracker.get_performance_metrics(y_test, y_pred)

        # Return experiment results
        return {
            'filter_type': filter_type,
            'mape (%)': metrics['MAPE'],
            'r2_score': metrics['R2'],
            'training_time (s)': metrics['training_time'],
            'inference_time_per_sample (s)': inference_time_per_sample,
            'total_inference_time (s)': metrics['inference_time'],
            'epochs_run': len(history.history['loss'])
        }

    def _save_results(self) -> pd.DataFrame:
        """Save experiment results to CSV"""
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
            results_df.to_csv(config.RESULTS_FILE, index=False)
            print(f"\nResults saved to: {config.RESULTS_FILE}")
        except IOError as e:
            print(f"Error saving results to CSV: {e}")

        return results_df

def main():
    # Suppress TensorFlow INFO messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run experiment
    experiment = ExperimentRunner()
    experiment.run()

if __name__ == "__main__":
    main() 