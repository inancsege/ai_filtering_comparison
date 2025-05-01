import os
import multiprocessing

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root: ai_filtering_comparison
DATA_DIR = os.path.join(BASE_DIR, 'scaledData')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Data ---
# Using one file initially for faster development
# TODO: Add logic to handle multiple files later
DATA_FILE = os.path.join(DATA_DIR, 'scaledData8.csv') # Example file

TARGET_COLUMN = 'available_capacity (Ah)'
# Use all other columns as features initially
# TODO: Add feature selection/engineering later if needed
FEATURE_COLUMNS = [
    'Time Elapsed (s)', 'soc', 'pack_voltage (V)', 'charge_current (A)',
    'max_cell_voltage (V)', 'min_cell_voltage (V)', 'max_temperature (℃)',
    'min_temperature (℃)'
]
# Columns to apply filtering on (e.g., voltage, current, temperature)
FILTER_TARGET_COLUMNS = [
    'pack_voltage (V)', 'charge_current (A)', 'max_cell_voltage (V)',
    'min_cell_voltage (V)', 'max_temperature (℃)', 'min_temperature (℃)'
]

# --- Performance Optimization ---
# Parallel processing
USE_MULTIPROCESSING = True
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free for system

# Memory optimization
USE_FLOAT32 = True  # Use float32 instead of float64 for reduced memory usage
CHUNK_SIZE = 10000  # For processing large datasets in chunks

# Cache settings
USE_CACHE = True
CACHE_EXPIRY_DAYS = 7  # Automatically invalidate cache after 7 days
CACHE_COMPRESSION_LEVEL = 3  # 0-9, higher = smaller file but slower read/write

# --- GPU Acceleration ---
USE_GPU = True  # Set to False to disable GPU acceleration completely
GPU_MEM_LIMIT = 0.8  # Use up to 80% of available GPU memory

# --- Filtering ---
FILTERS_TO_COMPARE = [
    'none', # Baseline
    'savitzky_golay',
    'moving_average',
    'gaussian',
    'median',
    'kalman' # Kalman might require more specific implementation/libraries
]
# Example parameters (can be tuned)
SG_WINDOW = 11
SG_POLYORDER = 3
MA_WINDOW = 5
GAUSSIAN_SIGMA = 2
MEDIAN_KERNEL_SIZE = 5

# --- Model ---
# Placeholder - will be used in model.py and train.py
SEQUENCE_LENGTH = 50 # Example: use previous 50 time steps to predict next
BATCH_SIZE = 64
EPOCHS = 10 # Keep low for initial testing

# --- Evaluation ---
RESULTS_FILE = os.path.join(RESULTS_DIR, 'comparison_results.csv')

# --- Logging ---
VERBOSE = True  # Set to False for less console output
LOG_FILE = os.path.join(RESULTS_DIR, 'run_log.txt') 