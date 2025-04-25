from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import time
import os

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

class ModelEvaluator:
    """Class responsible for model evaluation"""
    def __init__(self, metrics: Tuple[Metric, ...] = (MAPE(), R2Score())):
        self.metrics = metrics

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        results = {}
        for metric in self.metrics:
            value = metric.calculate(y_true, y_pred)
            results[metric.name] = value
            print(f"{metric.name}={value:.4f}" + ("%" if metric.name == "MAPE" else ""))
        return results

class ModelPerformanceTracker:
    """Class for tracking model performance metrics and timing"""
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.train_timer = Timer()
        self.inference_timer = Timer()

    def track_training(self, func: callable, *args, **kwargs) -> Any:
        """Track training time"""
        with self.train_timer:
            result = func(*args, **kwargs)
        return result

    def track_inference(self, func: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Track inference time and calculate per-sample time"""
        with self.inference_timer:
            result = func(*args, **kwargs)
        
        # Calculate per-sample time if applicable
        n_samples = len(args[0]) if args else 1
        time_per_sample = self.inference_timer.elapsed_time / n_samples

        return result, time_per_sample

    def get_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Get all performance metrics including timing"""
        metrics = self.evaluator.evaluate(y_true, y_pred)
        metrics.update({
            'training_time': getattr(self.train_timer, 'elapsed_time', 0),
            'inference_time': getattr(self.inference_timer, 'elapsed_time', 0)
        })
        return metrics

# Legacy wrapper function for backward compatibility
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Legacy wrapper for calculating metrics"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred)
    return results['MAPE'], results['R2']

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Test the evaluation system
    y_true_sample = np.array([100, 110, 120, 105, 95])
    y_pred_sample = np.array([102, 108, 123, 106, 98])

    print("Testing ModelEvaluator...")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true_sample, y_pred_sample)
    print("\nMetrics:", metrics)

    print("\nTesting ModelPerformanceTracker...")
    tracker = ModelPerformanceTracker(evaluator)
    
    # Example of tracking a dummy training function
    def dummy_train():
        time.sleep(0.5)
        return "training_complete"

    # Example of tracking a dummy inference function
    def dummy_predict(X):
        time.sleep(0.1)
        return np.array([1.0] * len(X))

    # Track training
    result = tracker.track_training(dummy_train)
    print(f"Training completed in {tracker.train_timer.elapsed_time:.4f} seconds")

    # Track inference
    X_dummy = np.array([1, 2, 3, 4, 5])
    predictions, time_per_sample = tracker.track_inference(dummy_predict, X_dummy)
    print(f"Inference completed in {tracker.inference_timer.elapsed_time:.4f} seconds")
    print(f"Time per sample: {time_per_sample:.4f} seconds") 