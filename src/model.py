from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import tensorflow as tf
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
                 learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

class LSTMModel(BaseModel):
    """LSTM model implementation"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None

    def build(self) -> tf.keras.Model:
        """Build and compile the LSTM model"""
        model = tf.keras.Sequential()

        # First LSTM layer
        model.add(tf.keras.layers.LSTM(
            units=self.config.lstm_units[0],
            return_sequences=len(self.config.lstm_units) > 1,
            input_shape=self.config.input_shape
        ))
        model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

        # Additional LSTM layers
        for units in self.config.lstm_units[1:]:
            model.add(tf.keras.layers.LSTM(units=units))
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))

        # Dense layers
        for units in self.config.dense_units:
            model.add(tf.keras.layers.Dense(units=units, activation='relu'))

        # Output layer
        model.add(tf.keras.layers.Dense(units=1))  # Regression output

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse'
        )

        self._model = model
        print("Model Summary:")
        model.summary()

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration"""
        return {
            'input_shape': self.config.input_shape,
            'lstm_units': self.config.lstm_units,
            'dense_units': self.config.dense_units,
            'dropout_rate': self.config.dropout_rate,
            'learning_rate': self.config.learning_rate
        }

class ModelFactory:
    """Factory class for creating models"""
    @staticmethod
    def create_lstm_model(input_shape: Tuple[int, int]) -> BaseModel:
        """Create an LSTM model with default configuration"""
        config = ModelConfig(input_shape=input_shape)
        return LSTMModel(config)

def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Legacy wrapper function for backward compatibility"""
    model = ModelFactory.create_lstm_model(input_shape)
    return model.build()

# Example usage (optional, for testing the module directly)
if __name__ == '__main__':
    # Example input shape: (sequence_length=50, num_features=8)
    example_input_shape = (config.SEQUENCE_LENGTH, len(config.FEATURE_COLUMNS))
    print(f"Building model for input shape: {example_input_shape}")
    
    # Using the new model factory
    model = ModelFactory.create_lstm_model(example_input_shape)
    model.build()
    print("\nModel built successfully.")
    # You could add dummy data training here for further testing
    # import numpy as np
    # dummy_X = np.random.rand(100, example_input_shape[0], example_input_shape[1])
    # dummy_y = np.random.rand(100, 1)
    # model.fit(dummy_X, dummy_y, epochs=1, batch_size=16) 