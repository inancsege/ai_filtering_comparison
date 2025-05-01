# AI Filtering Comparison

This project compares different filtering techniques for sensor data using machine learning models.

## Features

- Compare various filtering methods (Savitzky-Golay, Moving Average, Gaussian, Median, Kalman)
- GPU acceleration support for NVIDIA GPUs (via RAPIDS cuML) and Intel GPUs/CPUs (via scikit-learn-intelex)
- Comprehensive metrics reporting and visualization
- Support for model saving and evaluation
- Optimized memory usage and performance enhancements
- Automatic GPU detection and configuration

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd ai_filtering_comparison
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For GPU acceleration, run the automatic setup script:
   ```
   python scripts/setup_gpu.py
   ```
   This will detect your GPU hardware and install the appropriate packages.

   Alternatively, manually uncomment and install the appropriate GPU packages in requirements.txt:
   - For NVIDIA GPUs:
     ```
     pip install cudf-cu11>=23.10.0 cuml-cu11>=23.10.0 cupy-cuda11x>=12.0.0
     ```
   - For Intel CPUs/GPUs:
     ```
     pip install scikit-learn-intelex>=2023.0.0
     ```

## Usage

### Basic Usage

Run the experiment with default settings:
```
python main.py
```

### Command Line Options

- `--data`: Path to the input data file (CSV)
- `--output`: Path to save results
- `--filters`: Comma-separated list of filters to compare
- `--save-models`: Save trained models to disk
- `--model-type`: Model type to use (random_forest, linear_regression, svr, mlp)
- `--no-gpu`: Disable GPU acceleration
- `--no-cache`: Disable caching of results
- `--chunk-size`: Chunk size for processing large datasets
- `--jobs`: Number of parallel jobs to run
- `--sequence-length`: Sequence length for time series analysis
- `--batch-size`: Batch size for training
- `--epochs`: Number of epochs for training
- `--verbose`: Enable verbose logging
- `--debug`: Enable debug mode with additional info
- `--system-info`: Display system information and exit

### Examples

Compare specific filters:
```
python main.py --filters="savitzky_golay,moving_average,kalman"
```

Run with a custom dataset and save models:
```
python main.py --data="path/to/your/data.csv" --save-models
```

Disable GPU acceleration:
```
python main.py --no-gpu
```

Run with 4 parallel jobs and larger batch size:
```
python main.py --jobs=4 --batch-size=128
```

## Performance Optimization

This project includes several performance optimizations:

1. **Memory management**: Automatically downcasts data types to minimize memory usage and uses chunked processing for large datasets.

2. **Parallel processing**: Uses parallel processing when possible, with configurable number of workers.

3. **Caching**: Implements intelligent caching of intermediate results with automatic cache invalidation.

4. **GPU acceleration**: Automatically detects and uses available GPU hardware with memory management.

5. **Efficient data loading**: Optimized CSV loading with memory-mapped files for large datasets.

## Project Structure

- `main.py`: Main application entry point containing the core functionality
- `src/`: Supporting modules directory
  - `config.py`: Configuration settings
  - `data_loader.py`: Data loading and preprocessing
  - `filters.py`: Implementation of various filtering techniques
  - `utils.py`: Performance optimization and utility functions
- `scripts/`: Utility scripts
  - `setup_gpu.py`: GPU detection and setup script
- `results/`: Directory for output files and visualizations
- `scaledData/`: Directory for input data files
- `cache/`: Directory for cached results (created automatically)

## GPU Acceleration

The project supports GPU acceleration through:

1. **RAPIDS cuML** for NVIDIA GPUs - Provides significant speedups for RandomForest, LinearRegression, and SVR algorithms.

2. **Intel scikit-learn-intelex** for Intel CPUs/GPUs - Optimizes scikit-learn algorithms on Intel hardware.

The system will automatically detect available GPU support and use the appropriate acceleration.

## Requirements

- Python 3.8+
- Basic requirements: numpy, pandas, scikit-learn, scipy, matplotlib
- Performance optimizations: psutil, dask, joblib, numba
- For NVIDIA GPU support: CUDA 11.x or 12.x, cuDF, cuML
- For Intel acceleration: Intel oneAPI Base Toolkit, scikit-learn-intelex