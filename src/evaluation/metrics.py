import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def directional_accuracy(y_true, y_pred):
    correct_direction = np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])
    return np.mean(correct_direction) * 100


def sharpe_ratio(returns, risk_free_rate=0):
    return (np.mean(returns) - risk_free_rate) / np.std(returns)


def max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return np.max(drawdown)


# Example usage
if __name__ == "__main__":
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])

    print(f"RMSE: {root_mean_squared_error(y_true, y_pred)}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred)}")
    print(f"Directional Accuracy: {directional_accuracy(y_true, y_pred)}")

    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    print(f"Sharpe Ratio: {sharpe_ratio(returns)}")
    print(f"Max Drawdown: {max_drawdown(returns)}")