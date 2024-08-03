# In src/models/lstm_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import logging
from sklearn.base import BaseEstimator, RegressorMixin
import os


class LSTMModel(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, output_size, neurons=50, num_layers=3, dropout=0.2, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_size = output_size
        self.neurons = neurons
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        for i in range(self.num_layers):
            return_sequences = i < self.num_layers - 1
            if i == 0:
                model.add(LSTM(units=self.neurons, return_sequences=return_sequences, input_shape=self.input_shape))
            else:
                model.add(LSTM(units=self.neurons, return_sequences=return_sequences))
        model.add(Dropout(self.dropout))

        model.add(Dense(units=self.output_size))

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        self.model = model
        self.logger.info("LSTM model built successfully")
        return self.model

    def fit(self, X_train, y_train, **kwargs):
        if self.model is None:
            self.build_model()

        try:
            # Check if X_train is already 3D, if not, reshape it
            if len(X_train.shape) == 2:
                X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

            # Remove any samples with NaN values
            mask = ~np.isnan(X_train).any(axis=(1, 2))
            X_train = X_train[mask]
            y_train = y_train[mask]

            if len(X_train) == 0:
                raise ValueError("All training data contains NaN values after filtering")

            history = self.model.fit(
                X_train, y_train,
                **kwargs
            )
            self.logger.info("Model training completed")
            return history
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    import numpy as np

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        try:
            # Check if X is 2D, if not, reshape it
            if len(X.shape) == 2:
                # Remove any rows with NaN values
                X = X[~np.isnan(X).any(axis=1)]
            elif len(X.shape) == 3:
                # For 3D input (samples, time steps, features)
                X = X[~np.isnan(X).any(axis=(1, 2))]
            else:
                raise ValueError(f"Unexpected input shape: {X.shape}")

            if len(X) == 0:
                raise ValueError("All rows contain NaN values after filtering")

            # Ensure X has the correct shape for LSTM input (samples, time steps, features)
            if len(X.shape) == 2:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Check if the filepath has the correct extension
            if not filepath.endswith(('.keras', '.h5')):
                filepath += '.keras'
                self.logger.warning(f"Added .keras extension to the filepath. New filepath: {filepath}")

            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_params(self, deep=True):
        return {
            "input_shape": self.input_shape,
            "output_size": self.output_size,
            "neurons": self.neurons,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self