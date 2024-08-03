import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.data.data_loader import DataLoader
from src.data.preprocessor import CryptoDataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.lstm_model import LSTMModel
from src.utils.config import load_config
import logging


class ModelTrainer:
    def __init__(self):
        self.config = load_config('training_config')
        self.data_config = load_config('data_config')
        self.data_loader = DataLoader()
        self.preprocessor = CryptoDataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        data = self.data_loader.get_latest_data(self.config['lookback_periods'])
        data = self.preprocessor.preprocess(data)
        data = self.feature_engineer.engineer_features(data)
        return data

    def create_sequences(self, data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data.iloc[i + sequence_length]['close'])
        return np.array(X), np.array(y)

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=self.config['test_size'], random_state=42)

    def train_model(self, data, neurons=50, num_layers=3, dropout=0.2, learning_rate=0.001):
        X, y = self.create_sequences(data, self.config['sequence_length'])
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = LSTMModel(input_shape, 1, neurons=neurons, num_layers=num_layers,
                               dropout=dropout, learning_rate=learning_rate)

        history = self.model.fit(X_train, y_train,
                                 epochs=self.config['epochs'],
                                 batch_size=self.config['batch_size'],
                                 validation_split=self.config['validation_split'])

        self.model.save_model(f"{self.config['model_save_path']}/{self.data_config['symbol']}/lstm_model.keras")
        return history, (X_test, y_test)  # Return both history and test data

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        self.logger.info(f"Model Evaluation - MSE: {mse}, MAE: {mae}")
        return mse, mae


class RobustModelTrainer(ModelTrainer):
    def __init__(self, n_splits=5):
        super().__init__()
        self.n_splits = n_splits

    def train_model(self, data, **kwargs):
        X = data.drop(['close', 'time_weight'], axis=1).values
        y = data['close'].values

        # Reshape X to be 3D: (samples, time steps, features)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        models = []
        histories = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = self.build_model(X.shape[1:], **kwargs)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size']
            )

            models.append(model)
            histories.append(history)

        # Use the last fold for final evaluation
        self.model = models[-1]
        return histories, (X_val, y_val)

    def build_model(self, input_shape, **kwargs):
        model = LSTMModel(
            input_shape=input_shape,
            output_size=1,
            neurons=kwargs.get('neurons', 50),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.2),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )
        return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    trainer = ModelTrainer()
    data = trainer.prepare_data()
    history, test_data = trainer.train_model(data)
    mse, mae = trainer.evaluate_model(*test_data)