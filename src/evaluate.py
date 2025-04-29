from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
from sklearn.metrics import (mean_absolute_percentage_error, r2_score, mean_absolute_error,
                           mean_squared_error, mean_squared_log_error)
import time
import os
import tensorflow as tf
import math
from scipy.stats import pearsonr

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    # If running script directly, try relative import (adjust as needed)
    import sys
    sys.path.append(os.path.dirname(__file__))
    import config

class Timer:
    """Context manager for timing code execution"""
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

class Metric(ABC):
    """Abstract base class for evaluation metrics"""
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric value"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric"""
        pass
        
    @property
    def higher_is_better(self) -> bool:
        """Indicates if higher values of this metric are better"""
        return False

class MAPE(Metric):
    """Mean Absolute Percentage Error metric"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage

    @property
    def name(self) -> str:
        return "MAPE"

class R2Score(Metric):
    """R-squared score metric"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)

    @property
    def name(self) -> str:
        return "R2"
        
    @property
    def higher_is_better(self) -> bool:
        return True
        
class MAE(Metric):
    """Mean Absolute Error metric"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)
        
    @property
    def name(self) -> str:
        return "MAE"

class MSE(Metric):
    """Mean Squared Error metric"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)
        
    @property
    def name(self) -> str:
        return "MSE"

class RMSE(Metric):
    """Root Mean Squared Error metric"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return math.sqrt(mean_squared_error(y_true, y_pred))
        
    @property
    def name(self) -> str:
        return "RMSE"

class MSLE(Metric):
    """Mean Squared Logarithmic Error metric - for data with exponential trends"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Handle negative values by using max(0, val)
        y_true_pos = np.maximum(y_true, 0)
        y_pred_pos = np.maximum(y_pred, 0)
        
        try:
            return mean_squared_log_error(y_true_pos, y_pred_pos)
        except ValueError:
            # If there are still issues, return infinity
            return float('inf')
        
    @property
    def name(self) -> str:
        return "MSLE"

class RMSLE(Metric):
    """Root Mean Squared Logarithmic Error"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Handle negative values by using max(0, val)
        y_true_pos = np.maximum(y_true, 0)
        y_pred_pos = np.maximum(y_pred, 0)
        
        try:
            return math.sqrt(mean_squared_log_error(y_true_pos, y_pred_pos))
        except ValueError:
            # If there are still issues, return infinity
            return float('inf')
        
    @property
    def name(self) -> str:
        return "RMSLE"

class RelativeError(Metric):
    """Mean Relative Error (as percentage)"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
            
        # Calculate relative error only for non-zero true values
        rel_error = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
        return 100.0 * np.mean(rel_error)  # Convert to percentage
        
    @property
    def name(self) -> str:
        return "RelErr"

class PearsonCorrelation(Metric):
    """Pearson correlation coefficient between true and predicted values"""
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) < 2:  # Correlation requires at least 2 points
            return 0.0
            
        try:
            corr, _ = pearsonr(y_true, y_pred)
            return corr
        except Exception:
            return 0.0
        
    @property
    def name(self) -> str:
        return "Corr"
        
    @property
    def higher_is_better(self) -> bool:
        return True

class ModelEvaluator:
    """Class responsible for model evaluation"""
    def __init__(self, metrics: Optional[List[Metric]] = None):
        if metrics is None:
            self.metrics = [
                MAPE(),           # Mean Absolute Percentage Error
                R2Score(),        # R-squared 
                MAE(),            # Mean Absolute Error
                MSE(),            # Mean Squared Error
                RMSE(),           # Root Mean Squared Error
                MSLE(),           # Mean Squared Logarithmic Error
                RMSLE(),          # Root Mean Squared Logarithmic Error
                RelativeError(),  # Mean Relative Error
                PearsonCorrelation()  # Correlation coefficient
            ]
        else:
            self.metrics = metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        results = {}
        print("\nEvaluation Metrics:")
        
        # Calculate and print each metric
        for metric in self.metrics:
            try:
                value = metric.calculate(y_true, y_pred)
                results[metric.name] = value
                
                # Format display based on metric type
                if metric.name in ["MAPE", "RelErr"]:
                    print(f"  {metric.name:<6} = {value:.2f}% {'👍' if value < 10 else '👎'}")
                elif metric.name == "Corr":
                    print(f"  {metric.name:<6} = {value:.4f} {'👍' if value > 0.9 else '👎' if value < 0.7 else '👌'}")
                elif metric.name == "R2":
                    print(f"  {metric.name:<6} = {value:.4f} {'👍' if value > 0.9 else '👎' if value < 0.5 else '👌'}")
                else:
                    print(f"  {metric.name:<6} = {value:.6f}")
            except Exception as e:
                print(f"  Error calculating {metric.name}: {e}")
                results[metric.name] = float('nan')
        
        return results
        
    def get_best_model(self, results_list: List[Dict[str, Any]], primary_metric: str = "RMSE") -> Tuple[int, Dict[str, Any]]:
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
            
        # Find the metric object to determine if higher is better
        metric_obj = next((m for m in self.metrics if m.name == primary_metric), None)
        higher_is_better = metric_obj.higher_is_better if metric_obj else False
        
        # Extract the metric values
        metric_values = [r.get(primary_metric, float('inf') if not higher_is_better else float('-inf')) 
                        for r in results_list]
        
        # Find the best index
        if higher_is_better:
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)
            
        return best_idx, results_list[best_idx]

class ModelPerformanceTracker:
    """Class for tracking model performance metrics and timing"""
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.train_timer = Timer()
        self.inference_timer = Timer()
        
        # Check if GPU is available
        self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if self.gpu_available:
            print("ModelPerformanceTracker: GPU is available for inference")
        else:
            print("ModelPerformanceTracker: No GPU available, using CPU")

    def track_training(self, func: callable, *args, **kwargs) -> Any:
        """Track training time"""
        with self.train_timer:
            # Ensure TensorFlow operations run before timer ends
            result = func(*args, **kwargs)
            if self.gpu_available:
                # Ensure all GPU operations have completed
                tf.keras.backend.clear_session()
        return result

    def track_inference(self, func: callable, X_test: np.ndarray, batch_size: Optional[int] = None) -> Tuple[Any, float]:
        """
        Track inference time and calculate per-sample time
        
        Args:
            func: The prediction function to call
            X_test: Test data to predict
            batch_size: Optional batch size for prediction (for GPU efficiency)
            
        Returns:
            Tuple of (predictions, inference_time_per_sample)
        """
        n_samples = len(X_test)
        
        # If batch size is provided and GPU is available, use batched prediction
        if batch_size is not None and self.gpu_available and n_samples > batch_size:
            print(f"Using batched prediction with batch size {batch_size}")
            
            # Pre-allocate results array based on the shape of a single prediction
            with tf.device('/GPU:0'):
                # Get output shape from a single sample prediction
                sample_pred = func(X_test[:1])
                
                # Prepare output array
                if len(sample_pred.shape) > 1 and sample_pred.shape[1] == 1:
                    # Output is 2D with a single feature
                    y_pred = np.zeros((n_samples,), dtype=np.float32)
                else:
                    # Output has multiple features or is a different shape
                    output_shape = list(sample_pred.shape)
                    output_shape[0] = n_samples
                    y_pred = np.zeros(output_shape, dtype=np.float32)
                
                # Track total inference time
                with self.inference_timer:
                    # Process in batches
                    for i in range(0, n_samples, batch_size):
                        end_idx = min(i + batch_size, n_samples)
                        batch_pred = func(X_test[i:end_idx])
                        
                        if len(batch_pred.shape) > 1 and batch_pred.shape[1] == 1:
                            # Flatten if it's a 2D array with a single column
                            y_pred[i:end_idx] = batch_pred.flatten()
                        else:
                            y_pred[i:end_idx] = batch_pred
                            
                    # Force execution of pending GPU operations
                    tf.keras.backend.clear_session()
        else:
            # Standard prediction without batching
            with tf.device('/GPU:0' if self.gpu_available else '/CPU:0'):
                with self.inference_timer:
                    y_pred = func(X_test)
                    # Force execution of pending GPU operations
                    if self.gpu_available:
                        tf.keras.backend.clear_session()
        
        # Calculate per-sample time
        time_per_sample = self.inference_timer.elapsed_time / n_samples
        print(f"Inference completed in {self.inference_timer.elapsed_time:.4f}s, {time_per_sample:.6f}s per sample")

        return y_pred, time_per_sample

    def get_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Get all performance metrics including timing"""
        metrics = self.evaluator.evaluate(y_true, y_pred)
        metrics.update({
            'training_time': getattr(self.train_timer, 'elapsed_time', 0),
            'inference_time': getattr(self.inference_timer, 'elapsed_time', 0)
        })
        return metrics
        
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  filter_type: str = "unknown") -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report with metrics, timings and analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            filter_type: Name of the filter used
            
        Returns:
            Dictionary with evaluation results
        """
        # Calculate metrics
        metrics = self.evaluator.evaluate(y_true, y_pred)
        
        # Add timing information
        metrics.update({
            'training_time': getattr(self.train_timer, 'elapsed_time', 0),
            'inference_time': getattr(self.inference_timer, 'elapsed_time', 0),
            'inference_time_per_sample': getattr(self.inference_timer, 'elapsed_time', 0) / len(y_true)
        })
        
        # Calculate error distribution statistics
        errors = y_true - y_pred
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
            'samples_within_5percent': float(np.mean(abs_errors <= 0.05 * np.abs(y_true)) * 100),
            'samples_within_10percent': float(np.mean(abs_errors <= 0.1 * np.abs(y_true)) * 100),
        }
        
        # Create final report
        report = {
            'filter_type': filter_type,
            'metrics': metrics,
            'error_stats': error_stats,
            'sample_count': len(y_true),
            'gpu_used': self.gpu_available
        }
        
        return report

# Legacy wrapper function for backward compatibility
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Legacy wrapper for calculating metrics"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred)
    return results.get('MAPE', float('nan')), results.get('R2', float('nan'))

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Create more realistic test data
    np.random.seed(42)
    n_samples = 100
    
    # Create true signal: y = 2*x + 1 + noise
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 1, n_samples)
    
    # Create predicted signal with some error
    y_pred = 2 * x + 0.8 + np.random.normal(0, 0.5, n_samples)
    
    print(f"Testing with {n_samples} samples (y = 2x + 1 + noise)")
    print("Testing ModelEvaluator...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Test model performance tracker with dummy functions
    print("\nTesting ModelPerformanceTracker...")
    tracker = ModelPerformanceTracker(evaluator)
    
    # Create mock model predict function
    def mock_predict(X):
        # Simulate model prediction with y = 2x + 0.8 + small noise
        time.sleep(0.01)  # Simulate processing time
        x_vals = np.linspace(0, 10, len(X))
        return 2 * x_vals + 0.8 + np.random.normal(0, 0.2, len(X))
    
    # Create test samples
    X_test = np.random.rand(n_samples, 10)  # 10 features
    
    # Test inference tracking
    y_pred, time_per_sample = tracker.track_inference(mock_predict, X_test)
    
    # Generate comprehensive report
    report = tracker.generate_evaluation_report(y_true, y_pred, "test_filter")
    
    # Print detailed error statistics
    print("\nError Statistics:")
    for key, value in report['error_stats'].items():
        print(f"  {key}: {value:.4f}")
        
    # Print percentage of samples within error bounds
    print(f"\nSamples within 5% error: {report['error_stats']['samples_within_5percent']:.2f}%")
    print(f"Samples within 10% error: {report['error_stats']['samples_within_10percent']:.2f}%") 