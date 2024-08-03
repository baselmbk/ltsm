from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from src.models.lstm_model import LSTMModel
import logging


class HyperparameterTuner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def tune_hyperparameters(self, X, y, n_iter=10, cv=3):
        self.logger.info("Starting hyperparameter tuning...")

        # Ensure X and y are numpy arrays and contain only numeric data
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))

        input_shape = (X.shape[1], X.shape[2])
        output_size = 1 if len(y.shape) == 1 else y.shape[1]

        self.logger.info(f"Input shape: {input_shape}, Output size: {output_size}")

        model = LSTMModel(input_shape=input_shape, output_size=output_size)
        param_distributions = self.create_param_distributions()

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=2,
            error_score='raise'  # This will raise the error instead of just logging it
        )

        try:
            random_search.fit(X, y)

            best_params = random_search.best_params_
            best_score = -random_search.best_score_  # Convert back to MSE

            self.logger.info(f"Best parameters: {best_params}")
            self.logger.info(f"Best MSE: {best_score}")

            return best_params

        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise

    def create_param_distributions(self):
        param_distributions = {
            'neurons': [32, 64, 128],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1]
        }
        return param_distributions

    def evaluate_model(self, model, X, y):
        try:
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            return -mse  # Return negative MSE for GridSearchCV (it tries to maximize the score)
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            return np.inf  # Return infinity (worst possible score) in case of error