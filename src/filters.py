from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, gaussian_filter1d, medfilt
# For Kalman filter, we might need libraries like filterpy
# from filterpy.kalman import KalmanFilter # Example
# from filterpy.common import Q_discrete_white_noise # Example
import os
from typing import List, Dict, Any

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

class MovingAverageFilter(DataFilter):
    def __init__(self, window: int = config.MA_WINDOW):
        self.window = window

    def apply(self, signal: np.ndarray) -> np.ndarray:
        if len(signal) < self.window:
            return signal
        return pd.Series(signal).rolling(
            window=self.window, 
            center=True, 
            min_periods=1
        ).mean().values

    @property
    def name(self) -> str:
        return 'moving_average'

class GaussianFilter(DataFilter):
    def __init__(self, sigma: float = config.GAUSSIAN_SIGMA):
        self.sigma = sigma

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(signal, sigma=self.sigma)

    @property
    def name(self) -> str:
        return 'gaussian'

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

class FilterFactory:
    """Factory class for creating filters"""
    _filters: Dict[str, type] = {
        'none': NoFilter,
        'savitzky_golay': SavitzkyGolayFilter,
        'moving_average': MovingAverageFilter,
        'gaussian': GaussianFilter,
        'median': MedianFilter
    }

    @classmethod
    def create_filter(cls, filter_type: str) -> DataFilter:
        filter_class = cls._filters.get(filter_type)
        if not filter_class:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return filter_class()

class DataFilterProcessor:
    """Class responsible for applying filters to data"""
    def __init__(self, columns_to_filter: List[str] = config.FILTER_TARGET_COLUMNS):
        self.columns_to_filter = columns_to_filter

    def process(self, df: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Apply the specified filter to the selected columns of the DataFrame"""
        df_filtered = df.copy()
        data_filter = FilterFactory.create_filter(filter_type)
        print(f"Applying {data_filter.name} filter...")

        for col in self.columns_to_filter:
            if col not in df_filtered.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping filtering.")
                continue
            
            signal = df_filtered[col].values
            df_filtered[col] = data_filter.apply(signal)

        return df_filtered

def apply_filter(df: pd.DataFrame, filter_type: str, 
                columns_to_filter: List[str] = config.FILTER_TARGET_COLUMNS) -> pd.DataFrame:
    """Legacy wrapper function for backward compatibility"""
    processor = DataFilterProcessor(columns_to_filter)
    return processor.process(df, filter_type)

# Placeholder for Kalman filter implementation (requires more details about the system)
# def apply_kalman_filter(signal):
#     # Example structure (requires 'filterpy' or similar)
#     kf = KalmanFilter(dim_x=..., dim_z=...) # Define state and measurement dimensions
#     kf.x = ... # Initial state
#     kf.F = ... # State transition matrix
#     kf.H = ... # Measurement function
#     kf.P = ... # Covariance matrix
#     kf.R = ... # Measurement noise
#     kf.Q = ... # Process noise
#
#     filtered_signal = []
#     for z in signal:
#         kf.predict()
#         kf.update(z)
#         filtered_signal.append(kf.x[0, 0]) # Assuming state includes the value
#     return np.array(filtered_signal)


# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Create a sample DataFrame (replace with actual data loading if needed)
    try:
        from data_loader import load_data
        sample_data = load_data(config.DATA_FILE).head(100) # Load a small sample
    except (ImportError, FileNotFoundError):
         print("Could not load real data for filter testing, creating dummy data.")
         sample_data = pd.DataFrame({
             'Time Elapsed (s)': np.arange(100),
             'pack_voltage (V)': 4.2 + np.random.randn(100) * 0.1 + np.sin(np.arange(100)/10.0) * 0.2,
             'charge_current (A)': -2.0 + np.random.randn(100) * 0.05 + np.cos(np.arange(100)/5.0) * 0.1,
             'max_temperature (℃)': 40 + np.random.randn(100) * 0.5,
             'available_capacity (Ah)': 1200 - np.arange(100) * 0.1
         })
         config.FILTER_TARGET_COLUMNS = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)']


    print("\nOriginal Data Sample:")
    print(sample_data[config.FILTER_TARGET_COLUMNS].head())

    processor = DataFilterProcessor()
    for filter_type in config.FILTERS_TO_COMPARE:
        print("-" * 30)
        df_filtered = processor.process(sample_data.copy(), filter_type)
        print(f"\nData Sample after {filter_type} filter:")
        print(df_filtered[config.FILTER_TARGET_COLUMNS].head()) 