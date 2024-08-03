import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging


class CryptoDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)

    def preprocess(self, df):
        """
        Preprocess the cryptocurrency data.

        :param df: pandas DataFrame with OHLCV data
        :return: preprocessed pandas DataFrame
        """
        try:
            # Remove '_id' column if it exists
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                self.logger.info("Removed '_id' column from the DataFrame")

            # Handle missing values
            df = self.handle_missing_values(df)

            # Add basic technical indicators
            df = self.add_technical_indicators(df)

            # Scale the features
            df = self.scale_features(df)

            # Final check for NaN values
            if df.isnull().any().any():
                self.logger.warning("NaN values still present after preprocessing")
                self.logger.info("Columns with NaN values: " + ", ".join(df.columns[df.isnull().any()]))
                df = df.fillna(0.0)
                self.logger.warning("Filled remaining NaN values with 0.0")

            self.logger.info("Data preprocessing completed successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return None

    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Identify columns with NaN values
        columns_with_nan = df.columns[df.isnull().any()].tolist()
        if columns_with_nan:
            self.logger.warning(f"Columns with NaN values: {columns_with_nan}")

        # Fill NaN values with 0.0
        df = df.fillna(0.0)
        self.logger.warning("Filled NaN values with 0.0")

        return df

    def add_technical_indicators(self, df):
        """Add basic technical indicators to the dataset."""
        # Ensure the column names match your data
        close_column = 'close' if 'close' in df.columns else 'Close'
        high_column = 'high' if 'high' in df.columns else 'High'
        low_column = 'low' if 'low' in df.columns else 'Low'

        # Simple Moving Average
        df['SMA_20'] = df[close_column].rolling(window=20).mean()

        # Exponential Moving Average
        df['EMA_20'] = df[close_column].ewm(span=20, adjust=False).mean()

        # Relative Strength Index
        delta = df[close_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_middle'] = df[close_column].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df[close_column].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df[close_column].rolling(window=20).std()


        return df

    def scale_features(self, df):
        """Scale the features using MinMaxScaler."""
        cols_to_scale = ['open', 'high', 'low', 'close', 'volume'] if 'open' in df.columns else ['Open', 'High', 'Low',
                                                                                                 'Close', 'Volume']
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        return df


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Example usage
    from data_loader import DataLoader

    loader = DataLoader("BTC", timeframe="1d")
    data = loader.get_latest_data(days=100)

    if data is not None:
        preprocessor = CryptoDataPreprocessor()
        processed_data = preprocessor.preprocess(data)
        if processed_data is not None:
            print(processed_data.head())
            print(processed_data.columns)