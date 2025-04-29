# AI Filtering Comparison

This project compares different filtering techniques for sensor data using machine learning models.

## Features

- Compare various filtering methods (Savitzky-Golay, Moving Average, Gaussian, Median, Kalman)
- GPU acceleration support for NVIDIA GPUs (via RAPIDS cuML) and Intel GPUs/CPUs (via scikit-learn-intelex)
- Comprehensive metrics reporting and visualization
- Support for model saving and evaluation

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

3. For GPU acceleration, uncomment and install the appropriate GPU packages in requirements.txt:
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
- `--filters`: Comma-separated list of filters to compare
- `--save-models`: Save trained models to disk
- `--no-gpu`: Disable GPU acceleration

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

## Project Structure

- `main.py`: Entry point for the application
- `src/`: Source code directory
  - `config.py`: Configuration settings
  - `main.py`: Main implementation
  - `data_loader.py`: Data loading and preprocessing
  - `filters.py`: Implementation of various filtering techniques
- `results/`: Directory for output files and visualizations
- `scaledData/`: Directory for input data files

## GPU Acceleration

The project supports GPU acceleration through:

1. **RAPIDS cuML** for NVIDIA GPUs - Provides significant speedups for RandomForest, LinearRegression, and SVR algorithms.

2. **Intel scikit-learn-intelex** for Intel CPUs/GPUs - Optimizes scikit-learn algorithms on Intel hardware.

The system will automatically detect available GPU support and use the appropriate acceleration.

## Requirements

- Python 3.8+
- Basic requirements: numpy, pandas, scikit-learn, scipy, matplotlib
- For NVIDIA GPU support: CUDA 11.x, cuDF, cuML
- For Intel acceleration: Intel oneAPI Base Toolkit, scikit-learn-intelex