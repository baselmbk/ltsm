from src.data.data_handler import DataHandler
from src.models.model_handler import ModelHandler
from src.evaluation.evaluation_handler import EvaluationHandler

class EnsembleTrainer:
    def __init__(self, config, data_config, logger):
        self.symbol = data_config['symbol']
        self.timeframes = config['timeframes']
        self.lookback_periods = config['lookback_periods']
        self.logger = logger
        self.evaluation_handler = EvaluationHandler()

    def run(self):
        self.logger.info("Starting the ensemble training process...")

        # Load and preprocess data
        data = {tf: DataHandler.load_and_preprocess_data(self.symbol, tf, self.lookback_periods) for tf in
                self.timeframes}
        X_train, y_train, X_test, y_test, X_val, y_val = DataHandler.prepare_data(data, self.timeframes)

        # Create and train ensemble model
        input_shapes = {tf: X_train[tf].shape[1:] if len(X_train[tf].shape) == 3 else (1, X_train[tf].shape[1]) for tf
                        in self.timeframes}
        ensemble = ModelHandler.create_ensemble_model(self.symbol, self.timeframes, input_shapes, output_size=1)

        # Update ensemble weights
        ensemble.update_weights(X_val, y_val)

        # Evaluate ensemble model
        ensemble_metrics, ensemble_predictions = self.evaluation_handler.evaluate_ensemble(ensemble, X_test,
                                                                                           y_test[self.timeframes[0]])
        self.logger.info("Ensemble Model Metrics:")
        for metric, value in ensemble_metrics.items():
            self.logger.info(f"{metric}: {value}")

        # Evaluate individual models
        for timeframe, metrics, feature_importance in self.evaluation_handler.evaluate_individual_models(ensemble, X_test,
                                                                                                   y_test, data):
            self.logger.info(f"\n{timeframe} Model Metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value}")
            self.logger.info(f"Feature Importance for {timeframe}:")
            for _, row in feature_importance.iterrows():
                self.logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Run backtesting
        backtest_results = self.evaluation_handler.run_backtesting(y_test[self.timeframes[0]], ensemble_predictions)
        self.logger.info("Backtest results for ensemble:")
        for metric, value in backtest_results.items():
            self.logger.info(f"{metric}: {value}")

        # Save the ensemble model
        ensemble.save_model("models")

        self.logger.info("Ensemble training process completed.")