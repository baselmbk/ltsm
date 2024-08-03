import logging
import sys
from datetime import datetime
import os
import numpy as np
from src.data.data_loader import DataLoader
from src.data.preprocessor import CryptoDataPreprocessor
from src.data.feature_engineering import FeatureEngineer, TimeWeightedFeatureEngineer
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.training.trainer import ModelTrainer, RobustModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.backtester import Backtester
from src.utils.config import load_config
from src.utils.visualization import Visualizer


def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/crypto_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    visualizer = Visualizer()
    logger.info("Starting the cryptocurrency prediction process with combined data...")

    try:
        data_config = load_config('data_config')
        training_config = load_config('training_config')
        feature_config = load_config('feature_config')
        # Define time frames to process
        timeframes = ['1h', '4h', '1d', '3d', '1w', '1M']

        #timeframe = '1d'  # Focus on daily timeframe
        for timeframe in timeframes:
            logger.info(f"Processing timeframe: {timeframe}")


            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            data_loader = DataLoader(symbol=data_config['symbol'], timeframe=timeframe)
            raw_data = data_loader.get_latest_data(lookback_periods=data_config['lookback_periods'])

            preprocessor = CryptoDataPreprocessor()
            processed_data = preprocessor.preprocess(raw_data)

            if processed_data is None or processed_data.empty:
                logger.error("Preprocessing failed or resulted in empty data. Exiting.")
                return

            # Print data types after preprocessing
            logger.info("Data types after preprocessing:")
            for column, dtype in processed_data.dtypes.items():
                logger.info(f"{column}: {dtype}")

            feature_engineer = TimeWeightedFeatureEngineer()
            featured_data = feature_engineer.engineer_features(processed_data, timeframe=timeframe)

            # Print data types after feature engineering
            logger.info("Data types after feature engineering:")
            for column, dtype in featured_data.dtypes.items():
                logger.info(f"{column}: {dtype}")

            logger.info(f"Data shape after feature engineering: {featured_data.shape}")
            logger.info(f"Features: {featured_data.columns.tolist()}")

            # Visualizations
            visualizer.plot_price_history(featured_data, data_config['symbol'], timeframe)
            visualizer.plot_correlation_matrix(featured_data, timeframe)

            # Prepare data for hyperparameter tuning
            X = featured_data.drop(['close'], axis=1).values
            y = featured_data['close'].values

            # Perform hyperparameter tuning
            logger.info("Performing hyperparameter tuning...")
            tuner = HyperparameterTuner()
            best_params = tuner.tune_hyperparameters(X, y, n_iter=50)

            # Train the model
            logger.info("Training the model...")
            trainer = RobustModelTrainer()
            histories, test_data = trainer.train_model(featured_data, **best_params)

            # Plot training history
            for i, history in enumerate(histories):
                visualizer.plot_training_history(history, f"{timeframe}_fold_{i}")

            # Evaluate the model
            logger.info("Evaluating the model...")
            X_test, y_test = test_data
            predictions = trainer.model.predict(X_test)

            visualizer.plot_predictions_vs_actual(y_test, predictions.flatten(), timeframe)

            evaluator = ModelEvaluator()
            prediction_metrics = evaluator.evaluate_predictions(y_test, predictions.flatten())

            logger.info(f"Prediction metrics for {timeframe}:")
            for metric, value in prediction_metrics.items():
                logger.info(f"{metric}: {value}")

            # Calculate and plot feature importance
            feature_importance = evaluator.calculate_feature_importance(trainer.model, X_test, y_test, featured_data.columns.tolist())

            # Log feature importance
            logger.info(f"Feature Importance for {timeframe}:")
            for _, row in feature_importance.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")

            # Plot feature importance
            visualizer.plot_feature_importance(feature_importance['importance'].values, feature_importance['feature'].values,
                                    timeframe)

            # Run backtesting
            logger.info("Running backtesting...")
            backtester = Backtester()
            backtest_returns = backtester.run(y_test, predictions.flatten())
            backtest_results = backtester.evaluate(backtest_returns)

            logger.info(f"Backtest results for {timeframe}:")
            for metric, value in backtest_results.items():
                logger.info(f"{metric}: {value}")

            # Save the model
            model_save_path = f"models/{data_config['symbol']}/{timeframe}_model.keras"
            trainer.model.save_model(model_save_path)
            logger.info(f"Model for {timeframe} saved to {model_save_path}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception("Exception details:")

    logger.info("Cryptocurrency prediction process completed.")

if __name__ == "__main__":
    main()