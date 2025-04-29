from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import pickle
from joblib import Parallel, delayed
import multiprocessing
import hashlib
import time

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
    def __init__(self, file_path: str, use_cache: bool = True):
        self.file_path = file_path
        self.use_cache = use_cache
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self) -> str:
        """Generate cache file path based on the source file"""
        # Create hash of file path
        file_hash = hashlib.md5(self.file_path.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"data_cache_{file_hash}.pkl")
        
    def _check_cache_valid(self, cache_path: str) -> bool:
        """Check if cache file exists and is newer than source file"""
        if not os.path.exists(cache_path):
            return False
            
        # Check file modification times
        source_mtime = os.path.getmtime(self.file_path)
        cache_mtime = os.path.getmtime(cache_path)
        
        # Cache is valid if it's newer than source file
        return cache_mtime > source_mtime

    def load(self) -> pd.DataFrame:
        """Load data from CSV file with caching"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        # Check cache first if enabled
        if self.use_cache:
            cache_path = self._get_cache_path()
            if self._check_cache_valid(cache_path):
                try:
                    print(f"Loading cached data from: {cache_path}")
                    with open(cache_path, 'rb') as f:
                        df = pickle.load(f)
                    print(f"Loaded cached dataframe shape: {df.shape}")
                    return df
                except Exception as e:
                    print(f"Error loading cache, falling back to CSV: {e}")
        
        # Load from CSV if cache is not available or valid
        print(f"Loading data from: {self.file_path}")
        start_time = time.time()
        
        # Optimize CSV reading with appropriate parameters
        df = pd.read_csv(
            self.file_path,
            # Use float32 instead of float64 to save memory
            dtype={col: np.float32 for col in config.FEATURE_COLUMNS + [config.TARGET_COLUMN]},
            # Enable parallel processing for large files
            engine='c',  # C engine is faster than Python engine
        )
        
        load_time = time.time() - start_time
        print(f"Loaded dataframe shape: {df.shape} in {load_time:.2f} seconds")
        
        # Save to cache if enabled
        if self.use_cache:
            try:
                cache_path = self._get_cache_path()
                print(f"Caching data to: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f, protocol=4)  # Protocol 4 for better performance
            except Exception as e:
                print(f"Error caching data: {e}")
                
        return df

class DataPreprocessor:
    """Class responsible for data preprocessing"""
    def __init__(self, 
                 feature_cols: List[str] = config.FEATURE_COLUMNS,
                 target_col: str = config.TARGET_COLUMN,
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 test_size: float = 0.3,
                 validation_split: float = 0.5,
                 scaler_type: str = 'standard',
                 random_state: int = 42,
                 use_parallel: bool = True,
                 use_cache: bool = True):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_split = validation_split
        self.random_state = random_state
        
        self.use_parallel = use_parallel and multiprocessing.cpu_count() > 1
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1) if self.use_parallel else 1
        
        self.use_cache = use_cache
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'cache'
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize appropriate scaler
        if scaler_type.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        self.scaler_type = scaler_type

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        """
        Preprocess data:
        1. Select features and target
        2. Scale features
        3. Create sequences
        4. Split into train, validation, and test sets
        """
        print("Preprocessing data...")
        start_time = time.time()
        
        # Generate cache key based on dataframe and config
        cache_key = self._generate_cache_key(df)
        cache_path = os.path.join(self.cache_dir, f"preprocessed_{cache_key}.pkl")
        
        # Try to load from cache
        if self.use_cache and os.path.exists(cache_path):
            try:
                print(f"Loading preprocessed data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                print(f"Loaded preprocessed data from cache in {time.time() - start_time:.2f} seconds")
                return result
            except Exception as e:
                print(f"Cache loading error, proceeding with preprocessing: {e}")
        
        # 1. Select features and target
        features = df[self.feature_cols].values.astype(np.float32)  # Use float32 for better memory usage
        target = df[self.target_col].values.astype(np.float32)

        # 2. Scale features
        features_scaled = self.scaler.fit_transform(features)

        # 3. Create sequences
        X, y = self._create_sequences(features_scaled, target)

        if X.shape[0] == 0:
            raise ValueError("Sequence creation resulted in zero samples. Check data length and sequence length.")

        # 4. Split data
        result = self._split_data(X, y)
        
        # Save to cache if needed
        if self.use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f, protocol=4)
                print(f"Saved preprocessed data to cache: {cache_path}")
            except Exception as e:
                print(f"Error saving to cache: {e}")
        
        preprocessing_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
        
        return result
        
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """Generate a unique cache key based on dataframe and parameters"""
        # Create a hash based on data shape, column names, first/last values and parameters
        data_hash = hashlib.md5()
        data_hash.update(str(df.shape).encode())
        data_hash.update(str(list(df.columns)).encode())
        data_hash.update(str(df.iloc[0].values.tolist()).encode())
        data_hash.update(str(df.iloc[-1].values.tolist()).encode())
        
        # Add preprocessing parameters
        data_hash.update(str(self.feature_cols).encode())
        data_hash.update(str(self.target_col).encode())
        data_hash.update(str(self.sequence_length).encode())
        data_hash.update(str(self.test_size).encode())
        data_hash.update(str(self.validation_split).encode())
        data_hash.update(str(self.scaler_type).encode())
        data_hash.update(str(self.random_state).encode())
        
        return data_hash.hexdigest()

    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences for time series prediction, with optional parallel processing"""
        if len(features) <= self.sequence_length:
            print(f"Warning: Data length ({len(features)}) is less than or equal to sequence length ({self.sequence_length})")
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        total_sequences = len(features) - self.sequence_length
        print(f"Creating {total_sequences} sequences...")
        
        if not self.use_parallel or total_sequences < 1000:  # Only use parallel processing for larger datasets
            # Sequential processing
            X, y = [], []
            for i in range(total_sequences):
                X.append(features[i:(i + self.sequence_length)])
                y.append(target[i + self.sequence_length])
        else:
            # Parallel processing
            print(f"Using parallel processing with {self.n_jobs} workers")
            
            def create_sequence(i):
                return features[i:(i + self.sequence_length)], target[i + self.sequence_length]
            
            # Process in parallel
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(create_sequence)(i) for i in range(total_sequences)
            )
            
            # Unpack results
            X = [result[0] for result in results]
            y = [result[1] for result in results]
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

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
                 preprocessor: Optional[DataPreprocessor] = None,
                 use_cache: bool = True):
        self.data_loader = data_loader
        self.preprocessor = preprocessor or DataPreprocessor(use_cache=use_cache)
        self.use_cache = use_cache

    def get_processed_data(self) -> Tuple[np.ndarray, ...]:
        """Load and preprocess data"""
        df = self.data_loader.load()
        return self.preprocessor.preprocess(df)
        
    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Get summary statistics for the data"""
        if df is None:
            df = self.data_loader.load()
            
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            }
        }

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
        
        # Get and print data summary
        df = data_loader.load()
        summary = data_manager.get_data_summary(df)
        print("\nData Summary:")
        print(f"Shape: {summary['shape']}")
        print(f"Columns: {len(summary['columns'])}")
        print("\nFeature statistics:")
        for feature in config.FEATURE_COLUMNS[:3]:  # Show first 3 features only
            stats = summary['stats'][feature]
            print(f"  {feature}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        # Process data
        start_time = time.time()
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = data_manager.get_processed_data()
        print(f"Data loading and preprocessing completed in {time.time() - start_time:.2f} seconds")
        print(f"X_train sample shape: {X_train[0].shape}")
        print(f"y_train sample: {y_train[0]}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 