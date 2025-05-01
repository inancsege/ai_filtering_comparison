"""
Utility functions for performance optimization, memory management, and logging.
"""
import os
import sys
import time
import logging
import datetime
import gc
import psutil
from typing import Callable, Any, Dict, Optional, Union
import functools
import numpy as np
import pandas as pd
import pickle
import gzip
import hashlib
from contextlib import contextmanager

try:
    import config
except ModuleNotFoundError:
    # If running script directly, try relative import
    sys.path.append(os.path.dirname(__file__))
    import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ai_filtering')

# Suppress less important logs from other libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('joblib').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)


def setup_gpu():
    """Configure GPU settings for optimal performance"""
    try:
        # Check for NVIDIA GPU
        import cupy as cp
        import cuml
        
        # Set memory limit
        mem_limit = config.GPU_MEM_LIMIT
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        
        # Get total GPU memory and set limit
        total_memory = cp.cuda.Device(0).mem_info[1]
        memory_limit = int(total_memory * mem_limit)
        cp.cuda.set_allocator(cp.cuda.MemoryPool(memory_limit).malloc)
        
        logger.info(f"NVIDIA GPU configured: Memory limit set to {mem_limit*100:.0f}% ({memory_limit/(1024**3):.2f} GB)")
        return "NVIDIA"
    
    except (ImportError, Exception) as e:
        try:
            # Check for Intel optimization
            from sklearnex import patch_sklearn
            patch_sklearn()
            logger.info("Intel GPU/CPU acceleration configured")
            return "INTEL"
        except ImportError:
            logger.info("No GPU acceleration available")
            return None


def memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing code execution"""
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} completed in {elapsed:.4f} seconds")


def cache_result(cache_dir: str = config.CACHE_DIR, expiry_days: int = config.CACHE_EXPIRY_DAYS):
    """
    Decorator to cache function results to disk
    
    Args:
        cache_dir: Directory to store cache files
        expiry_days: Cache expiration in days
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not config.USE_CACHE:
                return func(*args, **kwargs)
                
            # Create cache key from function name, args, and kwargs
            key_parts = [func.__name__]
            for arg in args:
                if isinstance(arg, (np.ndarray, pd.DataFrame)):
                    # For large data structures, hash shape and a sample
                    if isinstance(arg, np.ndarray):
                        key_parts.append(f"array-{arg.shape}-{hash(str(arg.flatten()[:10]))}")
                    else:
                        key_parts.append(f"df-{arg.shape}-{hash(str(arg.iloc[:10].values.flatten()))}")
                else:
                    key_parts.append(str(arg))
                    
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
                
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{cache_key}.gz")
            
            # Check if cache exists and is not expired
            if os.path.exists(cache_file):
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                expiry_time = datetime.datetime.now() - datetime.timedelta(days=expiry_days)
                
                if file_time > expiry_time:
                    try:
                        with gzip.open(cache_file, 'rb') as f:
                            logger.info(f"Loading cached result for {func.__name__} from {cache_file}")
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Cache loading error: {e}")
            
            # Calculate result if not cached or expired
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                with gzip.open(cache_file, 'wb', compresslevel=config.CACHE_COMPRESSION_LEVEL) as f:
                    pickle.dump(result, f, protocol=4)
                logger.info(f"Cached result of {func.__name__} to {cache_file}")
            except Exception as e:
                logger.warning(f"Cache saving error: {e}")
                
            return result
        return wrapper
    return decorator


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame
    
    Args:
        df: Pandas DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    before_size = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        # Downcast integers
        c_min, c_max = df[col].min(), df[col].max()
        
        if c_min >= 0:  # Unsigned
            if c_max < 2**8:
                df[col] = df[col].astype(np.uint8)
            elif c_max < 2**16:
                df[col] = df[col].astype(np.uint16)
            elif c_max < 2**32:
                df[col] = df[col].astype(np.uint32)
        else:  # Signed
            if c_min > -2**7 and c_max < 2**7:
                df[col] = df[col].astype(np.int8)
            elif c_min > -2**15 and c_max < 2**15:
                df[col] = df[col].astype(np.int16)
            elif c_min > -2**31 and c_max < 2**31:
                df[col] = df[col].astype(np.int32)
    
    # Convert float64 to float32 if enabled
    if config.USE_FLOAT32:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype(np.float32)
    
    # Optimize string columns
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        if num_unique / len(df) < 0.5:  # If less than 50% unique values
            df[col] = df[col].astype('category')
    
    after_size = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    savings = (1 - after_size / before_size) * 100
    
    logger.info(f"DataFrame optimized: {before_size:.2f} MB → {after_size:.2f} MB ({savings:.1f}% reduction)")
    return df


def clean_memory():
    """Force garbage collection to free memory"""
    before = memory_usage()
    gc.collect()
    after = memory_usage()
    savings = before - after
    logger.info(f"Memory cleaned: {before:.1f} MB → {after:.1f} MB (freed {savings:.1f} MB)")


def process_in_chunks(df: pd.DataFrame, 
                      process_func: Callable[[pd.DataFrame], Any], 
                      chunk_size: int = config.CHUNK_SIZE) -> list:
    """
    Process a large DataFrame in chunks to avoid memory issues
    
    Args:
        df: Input DataFrame
        process_func: Function to apply to each chunk
        chunk_size: Number of rows per chunk
        
    Returns:
        List of results from each chunk
    """
    if len(df) <= chunk_size:
        return [process_func(df)]
        
    results = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(df) + chunk_size - 1) // chunk_size}")
        result = process_func(chunk)
        results.append(result)
        clean_memory()
        
    return results


def get_system_info() -> Dict[str, str]:
    """Get system information for logging purposes"""
    info = {
        "CPU Cores": str(psutil.cpu_count(logical=True)),
        "Physical CPUs": str(psutil.cpu_count(logical=False)),
        "Total RAM (GB)": f"{psutil.virtual_memory().total / (1024**3):.2f}",
        "Available RAM (GB)": f"{psutil.virtual_memory().available / (1024**3):.2f}",
        "OS": f"{os.name} - {sys.platform}",
        "Python Version": sys.version.split()[0],
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        import cupy as cp
        info["GPU"] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        info["GPU Memory (GB)"] = f"{cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3):.2f}"
    except (ImportError, Exception):
        info["GPU"] = "None detected"
    
    return info 