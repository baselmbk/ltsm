from src.data.data_loader import DataLoader
from src.data.preprocessor import CryptoDataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from sklearn.model_selection import train_test_split


class DataHandler:
    @staticmethod
    def load_and_preprocess_data(symbol, timeframe, lookback_periods):
        data_loader = DataLoader(symbol=symbol, timeframe=timeframe)
        raw_data = data_loader.get_latest_data(lookback_periods=lookback_periods)

        preprocessor = CryptoDataPreprocessor()
        processed_data = preprocessor.preprocess(raw_data)

        feature_engineer = FeatureEngineer()
        featured_data = feature_engineer.engineer_features(processed_data, timeframe=timeframe)

        return featured_data

    @staticmethod
    def prepare_data(data, timeframes):
        X_train, y_train, X_test, y_test, X_val, y_val = {}, {}, {}, {}, {}, {}
        for timeframe, df in data.items():
            X = df.drop('close', axis=1).values
            y = df['close'].values
            X_train[timeframe], X_test[timeframe], y_train[timeframe], y_test[timeframe] = train_test_split(X, y,
                                                                                                            test_size=0.2,
                                                                                                            random_state=42)
            X_train[timeframe], X_val[timeframe], y_train[timeframe], y_val[timeframe] = train_test_split(
                X_train[timeframe], y_train[timeframe], test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test, X_val, y_val