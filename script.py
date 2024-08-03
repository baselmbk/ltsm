from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer


def verify_timeframe_data(timeframe):
    loader = DataLoader(timeframe=timeframe)
    data = loader.get_latest_data(lookback_periods=100)

    print(f"\nTimeframe: {timeframe}")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Number of unique dates: {data.index.nunique()}")

    # Check for any large price jumps
    price_changes = data['close'].pct_change()
    large_changes = price_changes[abs(price_changes) > 0.1]
    if not large_changes.empty:
        print("Large price changes detected:")
        print(large_changes)

    return data


 # Verify data for 3d and 1M timeframes

data_3d = verify_timeframe_data('3d')
data_1M = verify_timeframe_data('1M')

# Now, apply feature engineering and check the results
fe = FeatureEngineer()
processed_3d = fe.engineer_features(data_3d, timeframe='3d')
processed_1M = fe.engineer_features(data_1M, timeframe='1M')

print("\nProcessed 3d data:")
print(processed_3d.head())
print(processed_3d.columns)

print("\nProcessed 1M data:")
print(processed_1M.head())
print(processed_1M.columns)
