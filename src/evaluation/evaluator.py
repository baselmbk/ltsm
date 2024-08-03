import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
from src.utils.config import load_config
import logging


class ModelEvaluator:
    def __init__(self):
        self.config = load_config('evaluation_config')
        self.logger = logging.getLogger(__name__)

    def evaluate_predictions(self, y_true, y_pred):
        """
        Evaluate the model predictions using various metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        self.logger.info(f"MSE: {mse}")
        self.logger.info(f"RMSE: {rmse}")
        self.logger.info(f"MAE: {mae}")
        self.logger.info(f"R-squared: {r2}")
        self.logger.info(f"MAPE: {mape}%")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

    def calculate_feature_importance(self, model, X, y, feature_names):
        self.logger.info("Calculating feature importance...")

        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        elif len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        n_features = X.shape[1]
        importance = np.zeros(n_features)
        baseline_mse = mean_squared_error(y, model.predict(X))

        for i in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            mse = mean_squared_error(y, model.predict(X_permuted))
            importance[i] = mse - baseline_mse

        # Normalize importance
        importance = importance / np.sum(np.abs(importance))

        # Ensure feature_names and importance have the same length
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

        # Create a dataframe of feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        self.logger.info("Feature importance calculation completed.")
        return feature_importance

    def calculate_returns(self, prices):
        """
        Calculate returns from a series of prices.
        """
        return np.log(prices).diff()

    def sharpe_ratio(self, returns, risk_free_rate=0):
        """
        Calculate the Sharpe ratio of a strategy.
        """
        return (returns.mean() - risk_free_rate) / returns.std()

    def max_drawdown(self, prices):
        """
        Calculate the maximum drawdown of a strategy.
        """
        cumulative_returns = (1 + self.calculate_returns(prices)).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        return drawdown.min()

    def evaluate_trading_performance(self, actual_prices, predicted_prices):
        """
        Evaluate the trading performance based on actual and predicted prices.
        """
        actual_returns = self.calculate_returns(actual_prices)
        predicted_returns = self.calculate_returns(predicted_prices)

        sharpe = self.sharpe_ratio(actual_returns)
        max_dd = self.max_drawdown(actual_prices)

        self.logger.info(f"Sharpe Ratio: {sharpe}")
        self.logger.info(f"Maximum Drawdown: {max_dd}")

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    evaluator = ModelEvaluator()

    # Simulated data
    y_true = np.array([100, 101, 99, 102, 98, 103])
    y_pred = np.array([99, 100, 100, 101, 99, 102])

    prediction_metrics = evaluator.evaluate_predictions(y_true, y_pred)
    print(f"Prediction metrics: {prediction_metrics}")

    # Note: For feature importance, you'd need to pass your actual model, X, and y