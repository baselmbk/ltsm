# In src/models/ensemble_model.py
import os
import json
from .base_model import BaseModel
from typing import Dict

import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


class EnsembleModel(BaseModel):
    def __init__(self, symbol, models):
        self.symbol = symbol
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # Start with equal weights
        self.logger = logging.getLogger(__name__)

    def update_weights(self, X_val, y_val):
        errors = []
        min_length = float('inf')

        # Find the minimum length across all timeframes
        for timeframe in y_val.keys():
            min_length = min(min_length, len(y_val[timeframe]))

        for timeframe, model, _ in self.models:
            if timeframe in X_val and timeframe in y_val:
                pred = model.predict(X_val[timeframe])
                # Ensure pred and y_val have the same length
                pred = pred[:min_length]
                y = y_val[timeframe][:min_length]
                error = mean_squared_error(y, pred)
                errors.append(error)
            else:
                self.logger.warning(f"No validation data for timeframe {timeframe}")
                errors.append(float('inf'))  # Assign a high error if no data

        # Inverse error weighting
        inv_errors = 1 / np.array(errors)
        self.weights = inv_errors / np.sum(inv_errors)

        # Update the weights in self.models
        self.models = [(timeframe, model, weight) for (timeframe, model, _), weight in zip(self.models, self.weights)]

        self.logger.info(f"Updated ensemble weights: {dict(zip([m[0] for m in self.models], self.weights))}")

    def build(self):
        # Initialize each model in the ensemble
        for _, model, _ in self.models:
            if hasattr(model, 'build_model'):
                model.build_model()
            elif not hasattr(model, 'model') or model.model is None:
                raise ValueError(f"Model {type(model).__name__} has no 'build_model' method and no 'model' attribute.")
        self.logger.info("Ensemble model built successfully")

    def train(self, X_train: Dict[str, np.ndarray], y_train: Dict[str, np.ndarray], **kwargs) -> Dict[str, Dict]:
        history = {}
        for timeframe, model, _ in self.models:
            if timeframe in X_train and timeframe in y_train:
                history[timeframe] = model.fit(X_train[timeframe], y_train[timeframe], **kwargs)
            else:
                self.logger.warning(f"No training data for timeframe {timeframe}")
        return history

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        predictions = []
        weights = []

        for timeframe, model, weight in self.models:
            if timeframe in X:
                pred = model.predict(X[timeframe])
                # Ensure pred is 2D: (samples, features)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                elif pred.ndim > 2:
                    pred = pred.reshape(pred.shape[0], -1)
                predictions.append(pred)
                weights.append(weight)
            else:
                self.logger.warning(f"No input data for timeframe {timeframe}")

        if not predictions:
            raise ValueError("No valid predictions were made.")

        # Ensure all predictions have the same shape
        min_length = min(pred.shape[0] for pred in predictions)
        predictions = [pred[:min_length] for pred in predictions]

        # Stack predictions and calculate weighted average
        stacked_predictions = np.stack(predictions, axis=-1)
        weighted_predictions = np.average(stacked_predictions, axis=-1, weights=weights)

        return weighted_predictions

    def save_model(self, base_path):
        # Create a directory for the ensemble
        ensemble_path = os.path.join(base_path, self.symbol, "ensemble")
        os.makedirs(ensemble_path, exist_ok=True)

        # Save the weights of the ensemble
        weights = {timeframe: weight for timeframe, _, weight in self.models}
        with open(os.path.join(ensemble_path, "weights.json"), "w") as f:
            json.dump(weights, f)

        # Save each individual model
        for timeframe, model, _ in self.models:
            model_path = os.path.join(base_path, self.symbol, f"ensemble/{timeframe}_model.keras")
            model.save_model(model_path)
            self.logger.info(f"Saved model for {timeframe} to {model_path}")

        self.logger.info(f"Saved ensemble weights to {ensemble_path}/weights.json")

    def load_model(self, base_path):
        # Load the weights of the ensemble
        ensemble_path = os.path.join(base_path, self.symbol, "ensemble")
        with open(os.path.join(ensemble_path, "weights.json"), "r") as f:
            weights = json.load(f)

        # Load each individual model
        for timeframe, model, _ in self.models:
            model_path = os.path.join(base_path, self.symbol, f"{timeframe}_model.keras")
            model.load_model(model_path)
            self.logger.info(f"Loaded model for {timeframe} from {model_path}")

        # Update the weights
        self.models = [(timeframe, model, weights[timeframe]) for timeframe, model, _ in self.models]
        self.logger.info(f"Loaded ensemble weights from {ensemble_path}/weights.json")

    def evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray) -> dict:
        predictions = self.predict(X)

        # Ensure y and predictions have the same shape
        min_length = min(len(y), len(predictions))
        y = y[:min_length]
        predictions = predictions[:min_length]

        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

    def calculate_feature_importance(self, X: Dict[str, np.ndarray], y: np.ndarray, feature_names: List[str]) -> Dict[
        str, Dict[str, float]]:
        feature_importance = {}
        for timeframe, model, _ in self.models:
            if timeframe in X:
                importance = model.calculate_feature_importance(X[timeframe], y, feature_names)
                feature_importance[timeframe] = dict(zip(feature_names, importance))
        return feature_importance