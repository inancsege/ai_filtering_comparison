from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    # If running script directly, try relative import (adjust as needed)
    import sys
    sys.path.append(os.path.dirname(__file__))
    import config

class DataLoader(ABC):
    """Abstract base class for data loading"""
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from source"""
        pass

class CSVDataLoader(DataLoader):
    """Concrete implementation for loading CSV data"""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load data from CSV file"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        print(f"Loading data from: {self.file_path}")
        df = pd.read_csv(self.file_path)
        print(f"Loaded dataframe shape: {df.shape}")
        return df

class DataPreprocessor:
    """Class responsible for data preprocessing"""
    def __init__(self, 
                 feature_cols: List[str] = config.FEATURE_COLUMNS,
                 target_col: str = config.TARGET_COLUMN,
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 test_size: float = 0.3,
                 validation_split: float = 0.5,
                 random_state: int = 42):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_split = validation_split
        self.random_state = random_state
        self.scaler = StandardScaler()

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Preprocess data:
        1. Select features and target
        2. Scale features
        3. Create sequences
        4. Split into train, validation, and test sets
        """
        print("Preprocessing data...")
        
        # 1. Select features and target
        features = df[self.feature_cols].values
        target = df[self.target_col].values

        # 2. Scale features
        features_scaled = self.scaler.fit_transform(features)

        # 3. Create sequences
        X, y = self._create_sequences(features_scaled, target)

        if X.shape[0] == 0:
            raise ValueError("Sequence creation resulted in zero samples. Check data length and sequence length.")

        # 4. Split data
        return self._split_data(X, y)

    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences for time series prediction"""
        X, y = [], []
        if len(features) <= self.sequence_length:
            print(f"Warning: Data length ({len(features)}) is less than or equal to sequence length ({self.sequence_length})")
            return np.array(X), np.array(y)

        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets"""
        # First split: separate training set from temp (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False  # Time series data, don't shuffle
        )

        # Second split: separate validation and test from temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.validation_split,
            random_state=self.random_state,
            shuffle=False
        )

        print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test, self.scaler

class DataManager:
    """High-level class for managing data operations"""
    def __init__(self, 
                 data_loader: DataLoader,
                 preprocessor: Optional[DataPreprocessor] = None):
        self.data_loader = data_loader
        self.preprocessor = preprocessor or DataPreprocessor()

    def get_processed_data(self) -> Tuple[np.ndarray, ...]:
        """Load and preprocess data"""
        df = self.data_loader.load()
        return self.preprocessor.preprocess(df)

# Legacy wrapper functions for backward compatibility
def load_data(file_path: str = config.DATA_FILE) -> pd.DataFrame:
    """Legacy wrapper for loading data"""
    loader = CSVDataLoader(file_path)
    return loader.load()

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """Legacy wrapper for preprocessing data"""
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df)

if __name__ == '__main__':
    try:
        # Using the new data management system
        data_loader = CSVDataLoader(config.DATA_FILE)
        data_manager = DataManager(data_loader)
        
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = data_manager.get_processed_data()
        print("Data loading and preprocessing successful.")
        print(f"X_train sample shape: {X_train[0].shape}")
        print(f"y_train sample: {y_train[0]}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 