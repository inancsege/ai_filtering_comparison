from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import platform
import time

# Ensure TensorFlow can see the GPUs and configure them
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Configure each GPU for optimal performance
        for gpu in gpus:
            # Enable memory growth to avoid allocating all GPU memory at once
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optional: Set memory limit if you want to cap GPU memory usage
            # mem_limit = 1024 * 4  # 4GB
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)]
            # )
            
        print(f"Model building will use {len(gpus)} GPU(s) with memory growth enabled")
        
        # Print GPU details
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  GPU {i}: {details.get('device_name', 'Unknown')}")
            
        # Make sure TensorFlow uses GPU
        tf.config.set_visible_devices(gpus, 'GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error configuring GPUs: {e}")
        print("Models will still be built but GPU performance might be affected")
else:
    print("No GPU found. Models will be built on CPU (slower).")

# Set up mixed precision for better performance on compatible GPUs
if gpus:
    try:
        # Mixed precision uses both float16 and float32 for better performance
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision enabled (float16/float32)")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    import config
except ModuleNotFoundError:
    # If running script directly, try relative import (adjust as needed)
    import sys
    sys.path.append(os.path.dirname(__file__))
    import config

class BaseModel(ABC):
    """Abstract base class for all models"""
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build and return the model"""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        pass

class ModelConfig:
    """Configuration class for model parameters"""
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 lstm_units: Tuple[int, ...] = (64, 32),
                 dense_units: Tuple[int, ...] = (16,),
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True,
                 learning_rate: float = 0.001,
                 use_cudnn: bool = True):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        # CuDNN optimized implementation for faster training when using GPU
        self.use_cudnn = use_cudnn and len(gpus) > 0

class LSTMModel(BaseModel):
    """LSTM model implementation"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        # Configure logging directory with timestamp
        self.log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs',
            f'lstm_model_{time.strftime("%Y%m%d-%H%M%S")}'
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def build(self) -> tf.keras.Model:
        """Build and compile the LSTM model"""
        # Make sure we're building on GPU if available
        device = '/GPU:0' if gpus else '/CPU:0'
        print(f"Building model on device: {device}")
        
        with tf.device(device):
            model = tf.keras.Sequential()

            # First LSTM layer
            lstm_kwargs = {
                'units': self.config.lstm_units[0],
                'return_sequences': len(self.config.lstm_units) > 1,
                'input_shape': self.config.input_shape
            }
            
            # Use CuDNN optimized implementation if available and specified
            if self.config.use_cudnn:
                # GPU optimized LSTM
                model.add(tf.keras.layers.LSTM(
                    **lstm_kwargs, 
                    activation='tanh',
                    recurrent_activation='sigmoid'
                ))
            else:
                # Standard LSTM
                model.add(tf.keras.layers.LSTM(**lstm_kwargs))
            
            if self.config.batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

            # Additional LSTM layers
            for i, units in enumerate(self.config.lstm_units[1:]):
                return_sequences = i < len(self.config.lstm_units) - 2
                if self.config.use_cudnn:
                    model.add(tf.keras.layers.LSTM(
                        units=units, 
                        return_sequences=return_sequences,
                        activation='tanh',
                        recurrent_activation='sigmoid'
                    ))
                else:
                    model.add(tf.keras.layers.LSTM(
                        units=units, 
                        return_sequences=return_sequences
                    ))
                    
                if self.config.batch_norm:
                    model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

            # Dense layers
            for units in self.config.dense_units:
                model.add(tf.keras.layers.Dense(units=units, activation='relu'))
                if self.config.batch_norm:
                    model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(self.config.dropout_rate/2))  # Less dropout in dense layers

            # Output layer
            model.add(tf.keras.layers.Dense(units=1))  # Regression output

            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            self._model = model
            print("Model Summary:")
            model.summary()

            # Print model's total parameters
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Non-trainable parameters: {non_trainable_params:,}")

            return model
        
    def get_callbacks(self, patience_early_stop: int = 10, patience_lr: int = 5) -> list:
        """Get callbacks for model training"""
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=patience_early_stop,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when a metric has stopped improving
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience_lr,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir, 
                histogram_freq=1,
                update_freq='epoch',  # 'batch' or 'epoch'
                profile_batch=0  # Disable profiling for better performance
            )
        ]
        return callbacks

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration"""
        return {
            'input_shape': self.config.input_shape,
            'lstm_units': self.config.lstm_units,
            'dense_units': self.config.dense_units,
            'dropout_rate': self.config.dropout_rate,
            'batch_norm': self.config.batch_norm,
            'learning_rate': self.config.learning_rate,
            'use_cudnn': self.config.use_cudnn
        }
        
    def save(self, path: str) -> None:
        """Save model to disk"""
        if self._model is None:
            raise ValueError("Model has not been built yet. Call build() first.")
            
        # Save model architecture and weights
        self._model.save(path)
        print(f"Model saved to {path}")
        
    def load(self, path: str) -> tf.keras.Model:
        """Load model from disk"""
        self._model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return self._model

class ModelFactory:
    """Factory class for creating models"""
    @staticmethod
    def create_lstm_model(input_shape: Tuple[int, int], 
                         lstm_units: Tuple[int, ...] = (64, 32),
                         dense_units: Tuple[int, ...] = (16,),
                         batch_norm: bool = True,
                         dropout_rate: float = 0.2) -> BaseModel:
        """Create an LSTM model with custom configuration"""
        config = ModelConfig(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dense_units=dense_units,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate
        )
        return LSTMModel(config)

def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Legacy wrapper function for backward compatibility"""
    model = ModelFactory.create_lstm_model(input_shape)
    return model.build()

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Check TensorFlow GPU configuration
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Is GPU available: {tf.test.is_gpu_available()}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Example input shape: (sequence_length=50, num_features=8)
    example_input_shape = (config.SEQUENCE_LENGTH, len(config.FEATURE_COLUMNS))
    print(f"Building model for input shape: {example_input_shape}")
    
    # Using the new model factory
    model = ModelFactory.create_lstm_model(example_input_shape)
    model.build()
    print("\nModel built successfully.")
    print(f"Callbacks configured: {len(model.get_callbacks())}")
    
    # Test a small batch prediction for GPU throughput
    try:
        import numpy as np
        
        # Create dummy batch
        batch_size = 32
        dummy_X = np.random.rand(batch_size, example_input_shape[0], example_input_shape[1]).astype(np.float32)
        
        # Warm up
        _ = model._model.predict(dummy_X[:1])
        
        # Test GPU performance
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            start_time = time.time()
            preds = model._model.predict(dummy_X)
            duration = time.time() - start_time
            print(f"\nGPU prediction test:")
            print(f"Batch of {batch_size} processed in {duration:.4f}s")
            print(f"Throughput: {batch_size/duration:.1f} samples/second")
    except Exception as e:
        print(f"Error testing GPU throughput: {e}") 