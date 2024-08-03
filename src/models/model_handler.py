from src.models.lstm_model import LSTMModel
from src.models.ensemble_model import EnsembleModel


class ModelHandler:
    @staticmethod
    def create_ensemble_model(symbol, timeframes, input_shapes, output_size):
        models = []
        for timeframe in timeframes:
            model = LSTMModel(input_shape=input_shapes[timeframe], output_size=output_size)
            try:
                model.load_model(f"models/{symbol}/{timeframe}_model.keras")
                print(f"Loaded pre-trained model for {timeframe}")
            except:
                print(f"Could not load pre-trained model for {timeframe}. Creating new model.")
                model.build_model()
            model.model.summary()
            models.append((timeframe, model, 1.0))
        return EnsembleModel(symbol, models)

    @staticmethod
    def train_models(ensemble, X_val, y_val):
        # Only update the ensemble weights, not retrain individual models
        ensemble.update_weights(X_val, y_val)