#!/usr/bin/env python3
"""
AI Filtering Comparison - Main Entry Point
This script is a simple wrapper that runs the main module in the src directory.
"""

import os
import sys
import argparse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the main module from src directory
from src.main import main as run_main

if __name__ == "__main__":
    # Run the main function from src/main.py
    run_main() 