from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy import linalg
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import time

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    # If running script directly, try relative import (adjust as needed)
    import sys
    sys.path.append(os.path.dirname(__file__))
    import config

class DataFilter(ABC):
    """Abstract base class for all filters"""
    @abstractmethod
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the filter to the signal"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the filter"""
        pass
        
    def get_params(self) -> Dict[str, Any]:
        """Return filter parameters"""
        return {}

class NoFilter(DataFilter):
    """Implements a pass-through filter"""
    def apply(self, signal: np.ndarray) -> np.ndarray:
        return signal

    @property
    def name(self) -> str:
        return 'none'

class SavitzkyGolayFilter(DataFilter):
    def __init__(self, window: int = config.SG_WINDOW, polyorder: int = config.SG_POLYORDER):
        self.window = window
        self.polyorder = polyorder
        self._validate_params()

    def _validate_params(self):
        if self.window <= self.polyorder:
            self.window = self.polyorder + 1 + (self.polyorder % 2)
        elif self.window % 2 == 0:
            self.window += 1

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.window:
            return signal
        return savgol_filter(signal, self.window, self.polyorder)

    @property
    def name(self) -> str:
        return 'savitzky_golay'
        
    def get_params(self) -> Dict[str, Any]:
        return {'window': self.window, 'polyorder': self.polyorder}

class MovingAverageFilter(DataFilter):
    def __init__(self, window: int = config.MA_WINDOW):
        self.window = window

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.window:
            return signal
            
        # Use cumulative sum for faster computation compared to rolling mean
        # This is much more efficient than pandas rolling windows for large datasets
        cs = np.cumsum(np.insert(signal, 0, 0))
        
        # Center the window
        half_window = self.window // 2
        
        # For the first half_window elements, use an expanding window
        filtered = np.zeros_like(signal)
        for i in range(half_window):
            window_size = half_window + i + 1
            filtered[i] = cs[i + window_size] / window_size
        
        # For the central part, use the full window
        for i in range(half_window, len(signal) - half_window):
            filtered[i] = (cs[i + half_window + 1] - cs[i - half_window]) / self.window
        
        # For the last half_window elements, use an expanding window in reverse
        for i in range(len(signal) - half_window, len(signal)):
            window_size = half_window + (len(signal) - i)
            filtered[i] = (cs[-1] - cs[i - half_window]) / window_size
        
        return filtered

    @property
    def name(self) -> str:
        return 'moving_average'
        
    def get_params(self) -> Dict[str, Any]:
        return {'window': self.window}

class GaussianFilter(DataFilter):
    def __init__(self, sigma: float = config.GAUSSIAN_SIGMA):
        self.sigma = sigma

    def apply(self, signal: np.ndarray) -> np.ndarray:
        # Ensure signal is float type for gaussian_filter1d
        signal_float = signal.astype(np.float64) if signal.dtype != np.float64 else signal
        return gaussian_filter1d(signal_float, sigma=self.sigma)

    @property
    def name(self) -> str:
        return 'gaussian'
        
    def get_params(self) -> Dict[str, Any]:
        return {'sigma': self.sigma}

class MedianFilter(DataFilter):
    def __init__(self, kernel_size: int = config.MEDIAN_KERNEL_SIZE):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.kernel_size:
            return signal
        return medfilt(signal, kernel_size=self.kernel_size)

    @property
    def name(self) -> str:
        return 'median'
        
    def get_params(self) -> Dict[str, Any]:
        return {'kernel_size': self.kernel_size}

class KalmanFilter(DataFilter):
    """Kalman filter implementation for 1D signals"""
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 1e-1):
        self.process_variance = process_variance  # Process noise
        self.measurement_variance = measurement_variance  # Measurement noise
        
    def apply(self, signal: np.ndarray) -> np.ndarray:
        # Initialize state and error
        x_hat = signal[0]  # Initial state estimate
        p = 1.0  # Initial error estimate
        
        # Pre-allocate output array
        filtered_signal = np.zeros_like(signal)
        filtered_signal[0] = x_hat
        
        # Forward pass
        for i in range(1, len(signal)):
            # Predict
            x_hat_minus = x_hat
            p_minus = p + self.process_variance
            
            # Update
            k = p_minus / (p_minus + self.measurement_variance)  # Kalman gain
            x_hat = x_hat_minus + k * (signal[i] - x_hat_minus)
            p = (1 - k) * p_minus
            
            filtered_signal[i] = x_hat
        
        return filtered_signal

    @property
    def name(self) -> str:
        return 'kalman'
        
    def get_params(self) -> Dict[str, Any]:
        return {
            'process_variance': self.process_variance, 
            'measurement_variance': self.measurement_variance
        }

class FilterFactory:
    """Factory class for creating filters"""
    _filters: Dict[str, type] = {
        'none': NoFilter,
        'savitzky_golay': SavitzkyGolayFilter,
        'moving_average': MovingAverageFilter,
        'gaussian': GaussianFilter,
        'median': MedianFilter,
        'kalman': KalmanFilter
    }

    @classmethod
    def create_filter(cls, filter_type: str, **kwargs) -> DataFilter:
        """Create a filter instance with optional custom parameters"""
        filter_class = cls._filters.get(filter_type.lower())
        if not filter_class:
            raise ValueError(f"Unknown filter type: {filter_type}")
            
        # Create instance with custom parameters if provided
        if kwargs:
            return filter_class(**kwargs)
        return filter_class()
        
    @classmethod
    def list_available_filters(cls) -> List[str]:
        """List all available filter types"""
        return list(cls._filters.keys())

class DataFilterProcessor:
    """Class responsible for applying filters to data"""
    def __init__(self, columns_to_filter: List[str] = config.FILTER_TARGET_COLUMNS):
        self.columns_to_filter = columns_to_filter
        
    def process(self, df: pd.DataFrame, filter_type: str, **filter_params) -> pd.DataFrame:
        """Apply the specified filter to the selected columns of the DataFrame"""
        df_filtered = df.copy()
        data_filter = FilterFactory.create_filter(filter_type, **filter_params)
        print(f"Applying {data_filter.name} filter with params: {data_filter.get_params()}")
        
        start_time = time.time()
        filtered_cols = 0

        for col in self.columns_to_filter:
            if col not in df_filtered.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping filtering.")
                continue
            
            signal = df_filtered[col].values
            df_filtered[col] = data_filter.apply(signal)
            filtered_cols += 1

        processing_time = time.time() - start_time
        print(f"Filtered {filtered_cols} columns in {processing_time:.4f} seconds")
        
        return df_filtered
        
    def process_multiple(self, df: pd.DataFrame, filter_types: List[str]) -> Dict[str, pd.DataFrame]:
        """Apply multiple filters to the data and return a dictionary of results"""
        results = {}
        for filter_type in filter_types:
            try:
                results[filter_type] = self.process(df.copy(), filter_type)
            except Exception as e:
                print(f"Error applying filter {filter_type}: {e}")
        return results
        
    def compare_filter_effects(self, df: pd.DataFrame, 
                              filter_types: Optional[List[str]] = None,
                              metric: str = 'mse') -> pd.DataFrame:
        """
        Compare the effects of different filters on the data using specified metric
        
        Args:
            df: Input DataFrame
            filter_types: List of filter types to compare (default: all available)
            metric: Metric to use for comparison ('mse', 'mae', 'var')
            
        Returns:
            DataFrame with comparison results
        """
        # Use all filters if not specified
        if filter_types is None:
            filter_types = FilterFactory.list_available_filters()
            
        # Get original data for comparison
        original_data = {}
        for col in self.columns_to_filter:
            if col in df.columns:
                original_data[col] = df[col].values
        
        # Initialize results
        results = []
        
        # Apply each filter and compute metrics
        for filter_type in filter_types:
            try:
                # Apply filter
                filtered_df = self.process(df.copy(), filter_type)
                
                # Compare with original for each column
                for col in self.columns_to_filter:
                    if col not in original_data:
                        continue
                        
                    original = original_data[col]
                    filtered = filtered_df[col].values
                    
                    # Compute metric
                    if metric.lower() == 'mse':
                        value = np.mean((original - filtered)**2)
                    elif metric.lower() == 'mae':
                        value = np.mean(np.abs(original - filtered))
                    elif metric.lower() == 'var':
                        value = np.var(filtered) / np.var(original) if np.var(original) > 0 else 0
                    else:
                        value = float('nan')
                        
                    results.append({
                        'filter_type': filter_type,
                        'column': col,
                        f'{metric}': value
                    })
            except Exception as e:
                print(f"Error comparing filter {filter_type}: {e}")
                
        return pd.DataFrame(results)

def apply_filter(df: pd.DataFrame, filter_type: str, 
                columns_to_filter: List[str] = config.FILTER_TARGET_COLUMNS) -> pd.DataFrame:
    """Legacy wrapper function for backward compatibility"""
    processor = DataFilterProcessor(columns_to_filter)
    return processor.process(df, filter_type)

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Create a sample DataFrame (replace with actual data loading if needed)
    try:
        from data_loader import load_data
        sample_data = load_data(config.DATA_FILE).head(100) # Load a small sample
    except (ImportError, FileNotFoundError):
         print("Could not load real data for filter testing, creating dummy data.")
         # Create synthetic test signal with noise
         x = np.linspace(0, 10, 500)
         # Clean signal: sine wave + linear trend
         clean_signal = np.sin(x) + 0.05 * x
         # Add noise
         np.random.seed(42)  # For reproducibility
         noisy_signal = clean_signal + np.random.normal(0, 0.2, len(x))
         
         sample_data = pd.DataFrame({
             'Time Elapsed (s)': x,
             'pack_voltage (V)': 4.2 + noisy_signal * 0.1,
             'charge_current (A)': -2.0 + np.random.randn(500) * 0.05 + np.cos(x/5.0) * 0.1,
             'max_temperature (℃)': 40 + np.random.randn(500) * 0.5,
             'available_capacity (Ah)': 1200 - x * 0.1
         })
         config.FILTER_TARGET_COLUMNS = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)']


    print("\nOriginal Data Sample:")
    print(sample_data[config.FILTER_TARGET_COLUMNS].head())

    processor = DataFilterProcessor()
    
    # Compare filter effects
    print("\n--- Filter Comparison ---")
    comparison = processor.compare_filter_effects(sample_data, metric='mse')
    print("\nMSE Comparison:")
    print(comparison.pivot(index='filter_type', columns='column', values='mse'))
    
    # Visual demonstration with matplotlib if available
    try:
        import matplotlib.pyplot as plt
        
        # Get a single column for visualization
        col = config.FILTER_TARGET_COLUMNS[0]
        x = sample_data.index
        y = sample_data[col].values
        
        plt.figure(figsize=(15, 8))
        plt.plot(x, y, 'gray', alpha=0.5, label='Original')
        
        # Apply each filter and plot
        for filter_type in FilterFactory.list_available_filters():
            filtered_df = processor.process(sample_data.copy(), filter_type)
            plt.plot(x, filtered_df[col].values, label=filter_type)
            
        plt.title(f'Filter Comparison on {col}')
        plt.xlabel('Sample Index')
        plt.ylabel(col)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization. Install with: pip install matplotlib") 