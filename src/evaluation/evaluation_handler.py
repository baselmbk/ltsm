import numpy as np
import logging

import pandas as pd
from sklearn.model_selection import learning_curve
from src.evaluation.evaluator import ModelEvaluator
from src.utils.visualization import Visualizer
from src.evaluation.backtester import Backtester


class EvaluationHandler:
    def __init__(self):
        self.visualizer = Visualizer()
        self.logger = logging.getLogger(__name__)

    def evaluate_ensemble(self, ensemble, X_test, y_test):
        self.logger.info(f"Shape of y_test: {y_test.shape}")

        # Ensure X_test is a dictionary
        if not isinstance(X_test, dict):
            raise ValueError("X_test should be a dictionary with timeframes as keys")

        # Get predictions for each timeframe
        predictions = {}
        for timeframe in X_test.keys():
            predictions[timeframe] = ensemble.predict({timeframe: X_test[timeframe]})
            self.logger.info(f"Shape of predictions for {timeframe}: {predictions[timeframe].shape}")

        # Find the minimum length among all predictions and y_test
        min_length = min(len(y_test), *[len(pred) for pred in predictions.values()])

        # Trim all predictions and y_test to the minimum length
        y_test = y_test[:min_length]
        for timeframe in predictions:
            predictions[timeframe] = predictions[timeframe][:min_length]

        # Combine predictions from all timeframes
        ensemble_predictions = np.mean([predictions[tf] for tf in X_test.keys()], axis=0)
        self.logger.info(f"Shape of ensemble_predictions: {ensemble_predictions.shape}")

        # Recalculate metrics with adjusted data
        ensemble_metrics = ensemble.evaluate({tf: X_test[tf][:min_length] for tf in X_test.keys()}, y_test)

        self.logger.info(f"Adjusted shape of y_test: {y_test.shape}")
        self.logger.info(f"Adjusted shape of ensemble_predictions: {ensemble_predictions.shape}")

        # Plot with adjusted data
        self.visualizer.plot_predictions_vs_actual(y_test, ensemble_predictions, 'ensemble')

        return ensemble_metrics, ensemble_predictions

    def evaluate_individual_models(self, ensemble, X_test, y_test, data):
        evaluator = ModelEvaluator()
        for timeframe, model, _ in ensemble.models:
            if timeframe in X_test:
                predictions = model.predict(X_test[timeframe])

                # Flatten predictions if they're 2D
                if predictions.ndim > 1:
                    predictions = predictions.flatten()

                metrics = evaluator.evaluate_predictions(y_test[timeframe], predictions)
                self.visualizer.plot_predictions_vs_actual(y_test[timeframe], predictions, timeframe)

                # Ensure X_test[timeframe] is 2D
                X_test_2d = X_test[timeframe].reshape(X_test[timeframe].shape[0], -1)

                feature_importance = evaluator.calculate_feature_importance(model, X_test_2d, y_test[timeframe],
                                                                            data[timeframe].columns.drop(
                                                                                'close').tolist())
                self.visualizer.plot_feature_importance(feature_importance['importance'].values,
                                                        feature_importance['feature'].values, timeframe)

                # Create a DataFrame for price history plotting
                price_history_df = pd.DataFrame({'close': predictions}, index=data[timeframe].index[-len(predictions):])
                self.visualizer.plot_price_history(price_history_df, timeframe)

                # Create a DataFrame for correlation matrix plotting
                correlation_df = pd.DataFrame(X_test_2d, columns=data[timeframe].columns.drop('close'))
                correlation_df['predictions'] = predictions
                self.visualizer.plot_correlation_matrix(correlation_df, timeframe)

                # Call new visualization methods
                self.visualizer.analyze_residuals(y_test[timeframe], predictions, timeframe)

                # For learning curve, we need to use the entire dataset
                X_full = data[timeframe].drop('close', axis=1).values
                y_full = data[timeframe]['close'].values
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_full, y_full, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
                self.visualizer.plot_learning_curve(train_sizes, train_scores, test_scores, timeframe)

                yield timeframe, metrics, feature_importance
            else:
                self.logger.warning(f"No test data for timeframe {timeframe}")

    @staticmethod
    def run_backtesting(y_test, ensemble_predictions):
        backtester = Backtester()
        backtest_returns = backtester.run(y_test, ensemble_predictions)
        backtest_results = backtester.evaluate(backtest_returns)
        return backtest_results