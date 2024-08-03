import pandas as pd
import numpy as np
import talib
from src.utils.config import load_config
import logging


class FeatureEngineer:
    def __init__(self):
        self.config = load_config('feature_config')
        self.logger = logging.getLogger(__name__)

    def engineer_features(self, df, timeframe='1h'):
        """
        Add engineered features to the dataframe for a specific timeframe.
        """
        original_columns = df.columns.tolist()
        df = self.add_technical_indicators(df, timeframe)
        df = self.add_time_features(df)
        df = self.handle_nan_values(df, original_columns)

        df['SMA_4'] = df['close'].rolling(window=4).mean()  # 4-week SMA
        df['EMA_4'] = df['close'].ewm(span=4, adjust=False).mean()  # 4-week EMA

        # Recalculate MACD for weekly data
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26,
                                                                  signalperiod=9)

        # Recalculate ATR for weekly data
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

        # Final NaN check and handling
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].isnull().sum() / len(df) > 0.5:  # If more than 50% are NaN, drop the column
                    df = df.drop(column, axis=1)
                else:
                    df[column] = df[column].fillna(df[column].median())

        return df
    def add_technical_indicators(self, df, timeframe):
        """
        Add technical indicators as features, adjusted for the given timeframe.
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Adjust periods based on timeframe
        period_multiplier = self.get_period_multiplier(timeframe)

        # Moving Averages
        for period in [10, 20, 30]:
            adjusted_period = max(2, int(period * period_multiplier))
            df[f'SMA_{period}'] = talib.SMA(close, timeperiod=adjusted_period)
            df[f'EMA_{period}'] = talib.EMA(close, timeperiod=adjusted_period)

        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close, timeperiod=max(2, int(20 * period_multiplier)))

        # RSI
        df['RSI'] = talib.RSI(close, timeperiod=max(2, int(14 * period_multiplier)))

        # MACD
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close,
                                                                  fastperiod=max(2, int(12 * period_multiplier)),
                                                                  slowperiod=max(2, int(26 * period_multiplier)),
                                                                  signalperiod=max(2, int(9 * period_multiplier)))

        # Stochastic Oscillator
        df['slowk'], df['slowd'] = talib.STOCH(high, low, close)

        # Average True Range
        df['ATR'] = talib.ATR(high, low, close, timeperiod=max(2, int(14 * period_multiplier)))

        # On-Balance Volume
        df['OBV'] = talib.OBV(close, volume)

        # Add new indicators for longer timeframes
        if timeframe in ['3d', '1w', '1M']:
            df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['ROC'] = talib.ROC(df['close'], timeperiod=10)
            df['Williams_R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

        if timeframe in ['1w', '1M']:
            # Add even longer-term indicators
            ichimoku_df = self.ichimoku_cloud(df)
            df = pd.concat([df, ichimoku_df], axis=1)
            df['ADL'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])


        return df

    def add_time_features(self, df):
        """
        Add time-based features.
        """
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def handle_nan_values(self, df, original_columns):
        """
        Handle NaN values in the dataframe.
        """
        nan_columns = df.columns[df.isna().any()].tolist()

        if nan_columns:
            self.logger.warning(f"NaN values found in columns: {nan_columns}")

            for col in nan_columns:
                nan_count = df[col].isna().sum()
                total_count = len(df)
                nan_percentage = (nan_count / total_count) * 100

                self.logger.info(f"Column {col}: {nan_count} NaN values ({nan_percentage:.2f}% of total)")

                if col in original_columns:
                    # For original features, use forward fill then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    self.logger.info(f"Filled NaN values in {col} using forward fill then backward fill")
                else:
                    # For engineered features, use a more sophisticated approach
                    if nan_percentage > 50:
                        # If more than 50% are NaN, drop the column
                        df = df.drop(columns=[col])
                        self.logger.warning(f"Dropped column {col} due to high NaN percentage")
                    else:
                        # Otherwise, fill NaNs with the median of non-NaN values
                        median_value = df[col].median()
                        df[col] = df[col].fillna(median_value)
                        self.logger.info(f"Filled NaN values in {col} with median value: {median_value}")

        return df

    def get_period_multiplier(self, timeframe):
        """
        Get a multiplier for adjusting indicator periods based on the timeframe.
        """
        timeframe_multipliers = {
            '1h': 1,
            '4h': 4,
            '1d': 24,
            '3d': 72,
            '1w': 168,
            '1M': 504  # Assuming an average of 21 trading days per month
        }
        return timeframe_multipliers.get(timeframe, 1)

    def ichimoku_cloud(self, df, high_col='high', low_col='low', close_col='close'):
        """
        Calculate Ichimoku Cloud indicator
        """
        ichimoku_df = pd.DataFrame(index=df.index)

        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df[high_col].rolling(window=9).max()
        period9_low = df[low_col].rolling(window=9).min()
        ichimoku_df['tenkan_sen'] = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df[high_col].rolling(window=26).max()
        period26_low = df[low_col].rolling(window=26).min()
        ichimoku_df['kijun_sen'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        ichimoku_df['senkou_span_a'] = ((ichimoku_df['tenkan_sen'] + ichimoku_df['kijun_sen']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df[high_col].rolling(window=52).max()
        period52_low = df[low_col].rolling(window=52).min()
        ichimoku_df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

        # Chikou Span (Lagging Span): Close plotted 26 periods in the past
        ichimoku_df['chikou_span'] = df[close_col].shift(-26)

        ichimoku_df['ichimoku_trend'] = np.where(df[close_col] > ichimoku_df['senkou_span_a'], 1,
                                                 np.where(df[close_col] < ichimoku_df['senkou_span_b'], -1, 0))

        return ichimoku_df


class TimeWeightedFeatureEngineer(FeatureEngineer):
    def __init__(self, half_life=365):
        super().__init__()
        self.half_life = half_life

    def engineer_features(self, df, timeframe='1d'):
        df = super().engineer_features(df, timeframe)
        df = self.add_time_weights(df)
        return df

    def add_time_weights(self, df):
        now = df.index[-1]
        days_old = (now - df.index).days
        df['time_weight'] = np.exp(-np.log(2) * days_old / self.half_life)
        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader

    loader = DataLoader()
    data = loader.get_latest_data(lookback_periods=1000)

    if data is not None:
        fe = FeatureEngineer()
        processed_data = fe.engineer_features(data, timeframe='1d')
        print(processed_data.head())
        print(processed_data.columns)