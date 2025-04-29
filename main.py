import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import sys

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
try:
    from src import config
    from src.train import ExperimentRunner
    from src.data_loader import CSVDataLoader, DataManager
    from src.filters import DataFilterProcessor
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you are running main.py from the project root directory.")
    exit(1)

def run_experiment(data_file: Optional[str] = None, 
                   filters: Optional[List[str]] = None,
                   visualize: bool = True) -> pd.DataFrame:
    """
    Run the experiment with the specified data file and filters
    
    Args:
        data_file: Path to the data file. If None, use the default from config
        filters: List of filters to compare. If None, use all filters from config
        visualize: Whether to visualize the results
        
    Returns:
        DataFrame with experiment results
    """
    # Use the default data file if none is provided
    if data_file is None:
        data_file = config.DATA_FILE
    
    # Use all filters if none are provided
    if filters is None:
        filters = config.FILTERS_TO_COMPARE
    else:
        # Validate filters
        for f in filters:
            if f not in config.FILTERS_TO_COMPARE:
                print(f"Warning: Unknown filter '{f}'. Available filters: {config.FILTERS_TO_COMPARE}")
                filters.remove(f)
        
        if not filters:
            print("No valid filters specified. Using all available filters.")
            filters = config.FILTERS_TO_COMPARE
    
    # Update config
    config.FILTERS_TO_COMPARE = filters
    
    # Run experiment
    experiment = ExperimentRunner(data_file=data_file)
    results_df = experiment.run()
    
    # Visualize results if requested
    if visualize and not results_df.empty:
        visualize_results(results_df)
    
    return results_df

def visualize_results(results_df: pd.DataFrame) -> None:
    """
    Visualize experiment results
    
    Args:
        results_df: DataFrame with experiment results
    """
    # Create results directory if it doesn't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot MAPE
    sns.barplot(x='filter_type', y='mape (%)', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title('MAPE by Filter Type (%)')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
    
    # Plot R2 Score
    sns.barplot(x='filter_type', y='r2_score', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title('R² Score by Filter Type')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
    
    # Plot Training Time
    sns.barplot(x='filter_type', y='training_time (s)', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('Training Time by Filter Type (s)')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
    
    # Plot Inference Time
    sns.barplot(x='filter_type', y='inference_time_per_sample (s)', data=results_df, ax=axes[1, 1])
    axes[1, 1].set_title('Inference Time per Sample by Filter Type (s)')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(config.RESULTS_DIR, 'filter_comparison.png'))
    print(f"Results visualization saved to: {os.path.join(config.RESULTS_DIR, 'filter_comparison.png')}")
    
    # Show figure
    plt.show()

def visualize_filter_effects(data_file: Optional[str] = None, 
                            column: str = 'pack_voltage (V)') -> None:
    """
    Visualize the effects of different filters on a sample column
    
    Args:
        data_file: Path to the data file. If None, use the default from config
        column: Column to visualize
    """
    # Use the default data file if none is provided
    if data_file is None:
        data_file = config.DATA_FILE
    
    # Load data
    data_loader = CSVDataLoader(data_file)
    df = data_loader.load()
    
    # Get a sample of data (first 200 rows)
    sample_df = df.head(200).copy()
    
    # Apply filters
    filter_processor = DataFilterProcessor()
    filtered_dfs = {}
    
    for filter_type in config.FILTERS_TO_COMPARE:
        filtered_dfs[filter_type] = filter_processor.process(sample_df.copy(), filter_type)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot original data
    plt.plot(sample_df.index, sample_df[column], 'k-', label='Original', alpha=0.5)
    
    # Plot filtered data
    for filter_type, filtered_df in filtered_dfs.items():
        plt.plot(filtered_df.index, filtered_df[column], label=filter_type)
    
    plt.title(f'Effect of Different Filters on {column}')
    plt.xlabel('Sample Index')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    
    # Save figure
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, f'filter_effects_{column.replace(" ", "_")}.png'))
    print(f"Filter effects visualization saved to: {os.path.join(config.RESULTS_DIR, f'filter_effects_{column.replace(' ', '_')}.png')}")
    
    # Show figure
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='AI Filtering Comparison')
    parser.add_argument('--data', type=str, help='Path to the data file')
    parser.add_argument('--filters', type=str, help='Comma-separated list of filters to compare')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    parser.add_argument('--vis-filters', action='store_true', help='Visualize filter effects only')
    parser.add_argument('--column', type=str, default='pack_voltage (V)', 
                        help='Column to visualize for filter effects')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up filters list if provided
    filters = None
    if args.filters:
        filters = [f.strip() for f in args.filters.split(',')]
    
    # Visualize filter effects if requested
    if args.vis_filters:
        visualize_filter_effects(args.data, args.column)
        return
    
    # Run experiment
    run_experiment(args.data, filters, not args.no_vis)

if __name__ == "__main__":
    main() 