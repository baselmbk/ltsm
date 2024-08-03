from pymongo import MongoClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from src.utils.config import load_config


class DataLoader:
    def __init__(self, symbol=None, timeframe=None):
        self.database_config = load_config('database_config')
        self.symbol = symbol or self.database_config.get('DEFAULT_SYMBOL', 'BTCUSD')
        self.timeframe = timeframe or self.database_config.get('DEFAULT_TIMEFRAME', '1h')
        self.mongo_client = MongoClient(self.database_config['MONGO_URL'])
        self.logger = logging.getLogger(__name__)

    def get_latest_data(self, lookback_periods=None):
        self.logger.info(f"Loading combined {self.symbol} {self.timeframe} OHLCV data...")

        end_date = datetime.now()
        start_date = self.calculate_start_date(end_date, lookback_periods)

        # Load data from both databases
        bitstamp_data = self.get_data_from_db('BITSTAMP_OHLCV_DB', start_date, end_date)
        binance_data = self.get_data_from_db('BINANCE_OHLCV_DB', start_date, end_date)

        # Combine and process the data
        combined_data = self.combine_and_process_data(bitstamp_data, binance_data)

        self.check_data_continuity(combined_data)

        self.logger.info(
            f"Loaded {len(combined_data)} records. Date range: {combined_data.index.min()} to {combined_data.index.max()}")
        return combined_data

    def get_data_from_db(self, db_name, start_date, end_date):
        db_name = self.database_config[db_name]
        db = self.mongo_client[db_name]
        collection = db[f"{self.symbol.lower()}_data"]

        pipeline = self.get_aggregation_pipeline()
        pipeline.insert(0, {"$match": {"timestamp": {"$gte": start_date, "$lte": end_date}}})

        data = list(collection.aggregate(pipeline))
        self.logger.info(f"Retrieved {len(data)} records from {db_name}")
        return data

    def combine_and_process_data(self, bitstamp_data, binance_data):
        combined_data = pd.DataFrame(bitstamp_data + binance_data)
        combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
        combined_data.set_index('timestamp', inplace=True)
        combined_data.sort_index(inplace=True)

        # Remove duplicates, reconciling conflicting data points
        combined_data = self.reconcile_duplicates(combined_data)

        return combined_data

    def reconcile_duplicates(self, df):
        if df.index.duplicated().any():
            self.logger.warning(f"Found {df.index.duplicated().sum()} duplicate timestamps")
            # Group by index and aggregate
            df = df.groupby(df.index).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        return df

    def check_data_continuity(self, df):
        expected_freq = self.get_expected_frequency()
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
        missing_dates = date_range.difference(df.index)
        if len(missing_dates) > 0:
            self.logger.warning(f"Found {len(missing_dates)} missing data points")
            self.logger.debug(f"Missing dates: {missing_dates}")

    def get_expected_frequency(self):
        freq_map = {'1h': 'H', '4h': '4H', '1d': 'D', '3d': '3D', '1w': 'W', '1M': 'MS'}
        return freq_map.get(self.timeframe, 'D')

    def get_aggregation_pipeline(self):
        if self.timeframe == '1h':
            return [{"$sort": {"timestamp": 1}}]

        group_id = self.get_group_id()
        return [
            {"$sort": {"timestamp": 1}},
            {"$group": {
                "_id": group_id,
                "timestamp": {"$first": "$timestamp"},
                "open": {"$first": "$open"},
                "high": {"$max": "$high"},
                "low": {"$min": "$low"},
                "close": {"$last": "$close"},
                "volume": {"$sum": "$volume"}
            }},
            {"$sort": {"timestamp": 1}}
        ]

    def get_group_id(self):
        if self.timeframe == '4h':
            return {
                "year": {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"},
                "day": {"$dayOfMonth": "$timestamp"},
                "hour": {"$subtract": [{"$hour": "$timestamp"}, {"$mod": [{"$hour": "$timestamp"}, 4]}]}
            }
        elif self.timeframe == '1d':
            return {
                "year": {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"},
                "day": {"$dayOfMonth": "$timestamp"}
            }
        elif self.timeframe == '3d':
            return {
                "threeDay": {
                    "$subtract": [
                        {"$dayOfYear": "$timestamp"},
                        {"$mod": [{"$subtract": [{"$dayOfYear": "$timestamp"}, 1]}, 3]}
                    ]
                },
                "year": {"$year": "$timestamp"}
            }
        elif self.timeframe == '1w':
            return {
                "year": {"$year": "$timestamp"},
                "week": {"$week": "$timestamp"}
            }
        elif self.timeframe == '1M':
            return {
                "year": {"$year": "$timestamp"},
                "month": {"$month": "$timestamp"}
            }
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

    def calculate_start_date(self, end_date, lookback_periods):
        if self.timeframe == '1h':
            return end_date - timedelta(hours=lookback_periods)
        elif self.timeframe == '4h':
            return end_date - timedelta(hours=4 * lookback_periods)
        elif self.timeframe == '1d':
            return end_date - timedelta(days=lookback_periods)
        elif self.timeframe == '3d':
            return end_date - timedelta(days=3 * lookback_periods)
        elif self.timeframe == '1w':
            return end_date - timedelta(weeks=lookback_periods)
        elif self.timeframe == '1M':
            return end_date - timedelta(days=30 * lookback_periods)
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example usage
    loader = DataLoader(symbol='BTCUSD', timeframe='1d')
    data = loader.get_latest_data(lookback_periods=100)
    print(data.head())
    print(f"Shape: {data.shape}")