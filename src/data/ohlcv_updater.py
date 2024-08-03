import os
import sys
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure
import ccxt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.utils.logger import setup_logger


class OHLCVUpdater:
    def __init__(self):
        self.database_config = load_config('database_config')
        self.logger = setup_logger("ohlcv_updater")
        self.mongo_client = MongoClient(self.database_config['MONGO_URL'])
        self.exchanges = {
            'binance': self.init_exchange('binance'),
            'bitstamp': self.init_exchange('bitstamp')
        }
        self.symbol_mapping = {
            'binance': {
                'BTCUSD': 'BTCUSDT',
            },
            'bitstamp': {
                'BTCUSD': 'BTC/USD',
            }
        }
        self.collection_mapping = {
            'BTCUSD': 'btcusd_data',
        }

    def init_exchange(self, exchange_name):
        try:
            exchange_class = getattr(ccxt, exchange_name)
            return exchange_class({
                'apiKey': os.getenv(f'{exchange_name.upper()}_API_KEY'),
                'secret': os.getenv(f'{exchange_name.upper()}_SECRET'),
                'enableRateLimit': True,
            })
        except Exception as e:
            self.logger.error(f"Failed to initialize {exchange_name} exchange: {str(e)}")
            return None

    def get_exchange_symbol(self, exchange_name, generic_symbol):
        return self.symbol_mapping[exchange_name].get(generic_symbol, generic_symbol)

    def get_collection_name(self, generic_symbol):
        return self.collection_mapping.get(generic_symbol, f"{generic_symbol.lower()}_data")

    def get_last_timestamp(self, collection):
        try:
            last_record = collection.find_one(sort=[("timestamp", -1)])
            return last_record["timestamp"] if last_record else None
        except Exception as e:
            self.logger.error(f"Error getting last timestamp: {str(e)}")
            return None

    def fetch_ohlcv(self, exchange, symbol, timeframe, since):
        if exchange is None:
            raise ValueError("Exchange object is None")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
        return [
            {
                "timestamp": datetime.fromtimestamp(candle[0] / 1000),
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            }
            for candle in ohlcv
        ]

    def ensure_collection_and_index(self, db, collection_name):
        try:
            collection = db[collection_name]

            if collection.count_documents({}) == 0:
                self.logger.info(f"Collection {collection_name} is empty. Creating it with an index.")
                collection.create_index([('timestamp', ASCENDING)], unique=True)
            else:
                index_info = collection.index_information()
                if 'timestamp_1' not in index_info:
                    self.logger.info(f"Creating index on timestamp for collection {collection_name}")
                    collection.create_index([('timestamp', ASCENDING)], unique=True)
                else:
                    self.logger.info(f"Index on timestamp already exists for collection {collection_name}")

            return collection
        except OperationFailure as e:
            self.logger.error(f"Failed to ensure collection and index: {str(e)}")
            return None

    def update_exchange_data(self, exchange_name, generic_symbol, timeframe):
        db_name = self.database_config[f'{exchange_name.upper()}_OHLCV_DB']
        db = self.mongo_client[db_name]
        collection_name = self.get_collection_name(generic_symbol)
        exchange_symbol = self.get_exchange_symbol(exchange_name, generic_symbol)

        collection = self.ensure_collection_and_index(db, collection_name)
        if collection is None:
            return

        last_timestamp = self.get_last_timestamp(collection)
        if last_timestamp is None:
            # If no data exists, start from 7 days ago
            since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        else:
            # Start from the last timestamp in the database
            since = int(last_timestamp.timestamp() * 1000)

        exchange = self.exchanges[exchange_name]
        if exchange is None:
            self.logger.error(f"Exchange {exchange_name} is not initialized")
            return

        try:
            total_records_inserted = 0
            while True:
                new_data = self.fetch_ohlcv(exchange, exchange_symbol, timeframe, since)

                if not new_data:
                    break

                # Filter out any records that already exist in the database
                new_data = [record for record in new_data if record['timestamp'] > last_timestamp]

                if new_data:
                    result = collection.insert_many(new_data)
                    inserted_count = len(result.inserted_ids)
                    total_records_inserted += inserted_count
                    self.logger.info(f"Inserted {inserted_count} new records for {exchange_name} {generic_symbol}")

                    # Update the 'since' timestamp for the next iteration
                    since = int(new_data[-1]['timestamp'].timestamp() * 1000) + 1
                    last_timestamp = new_data[-1]['timestamp']
                else:
                    break

                # Check if we've reached the current time
                if since > int(datetime.now().timestamp() * 1000):
                    break

            if total_records_inserted > 0:
                self.logger.info(
                    f"Total of {total_records_inserted} new records inserted for {exchange_name} {generic_symbol}")
            else:
                self.logger.info(f"No new data to insert for {exchange_name} {generic_symbol}")

        except ccxt.NetworkError as e:
            self.logger.error(f"Network error when fetching data from {exchange_name}: {str(e)}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error for {exchange_name}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error for {exchange_name}: {str(e)}")

    def update_all(self, generic_symbol, timeframe):
        self.logger.info(f"Starting OHLCV data update process for {generic_symbol}")
        for exchange_name, exchange in self.exchanges.items():
            if exchange is not None:
                self.update_exchange_data(exchange_name, generic_symbol, timeframe)
            else:
                self.logger.warning(f"Skipping update for {exchange_name} as it was not properly initialized")
        self.logger.info(f"OHLCV data update process completed for {generic_symbol}")



def main():
    updater = OHLCVUpdater()
    try:
        generic_symbol = 'BTCUSD'  # Use a generic symbol
        timeframe = '1h'
        updater.update_all(generic_symbol, timeframe)
    finally:
        updater.mongo_client.close()


if __name__ == "__main__":
    main()