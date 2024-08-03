import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from scipy import stats
from src.utils.config import load_config


class Visualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.database_config = load_config('database_config')
        self.symbol = self.database_config.get('DEFAULT_SYMBOL', 'BTCUSD')

    def save_plot(self, plt, name, timeframe, fold=None):
        date_str = datetime.now().strftime('%Y-%m-%d')
        base_path = f'plots/{self.symbol}/{date_str}/{timeframe}'

        if fold is not None:
            path = f'{base_path}/training_history'
            filename = f'{path}/fold_{fold}.png'
        else:
            path = base_path
            filename = f'{path}/{name}.png'

        os.makedirs(path, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot saved as {filename}")

    def plot_price_history(self, data, timeframe):
        plt.figure(figsize=(12, 6))

        if isinstance(data, pd.DataFrame):
            plt.plot(data.index, data['close'])
        elif isinstance(data, np.ndarray):
            plt.plot(range(len(data)), data)
        else:
            raise ValueError("Data must be either a pandas DataFrame or a numpy array")

        plt.title(f'{self.symbol} Price History ({timeframe})')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.save_plot(plt, 'price_history', timeframe)

    def plot_training_history(self, history, timeframe, fold):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.symbol} Model Training History ({timeframe})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        self.save_plot(plt, 'training_history', timeframe, fold)

    def plot_predictions_vs_actual(self, actual, predictions, timeframe):
        plt.figure(figsize=(12, 6))

        # Ensure actual and predictions have the same length
        min_length = min(len(actual), len(predictions))
        actual = actual[:min_length]
        predictions = predictions[:min_length]

        x = np.arange(min_length)
        plt.plot(x, actual, label='Actual', alpha=0.7)
        plt.plot(x, predictions, label='Predicted', alpha=0.7)
        plt.title(f'{self.symbol} Predictions vs Actual ({timeframe})')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        self.save_plot(plt, 'predictions_vs_actual', timeframe)

    def plot_feature_importance(self, importance, feature_names, timeframe):
        plt.figure(figsize=(12, 10))
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        importance_df = importance_df.sort_values('importance', ascending=True)
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title(f'{self.symbol} Feature Importance ({timeframe})')
        plt.tight_layout()
        self.save_plot(plt, 'feature_importance', timeframe)

    def plot_correlation_matrix(self, data, timeframe):
        plt.figure(figsize=(12, 10))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'{self.symbol} Correlation Matrix ({timeframe})')
        plt.tight_layout()
        self.save_plot(plt, 'correlation_matrix', timeframe)

    def plot_ensemble_predictions(self, actual, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predictions, label='Ensemble Predictions')
        plt.title(f'{self.symbol} Ensemble Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        self.save_plot(plt, 'predictions_vs_actual', 'ensemble')

    def analyze_residuals(self, y_true, y_pred, timeframe):
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals)
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot ({timeframe})')
        self.save_plot(plt, 'residual_plot', timeframe)

        # Q-Q plot
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f"Q-Q plot ({timeframe})")
        self.save_plot(plt, 'qq_plot', timeframe)

        # Histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Residuals ({timeframe})')
        self.save_plot(plt, 'residuals_histogram', timeframe)

    def plot_learning_curve(self, train_sizes, train_scores, test_scores, timeframe):
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title(f'Learning Curve ({timeframe})')
        plt.legend()
        self.save_plot(plt, 'learning_curve', timeframe)