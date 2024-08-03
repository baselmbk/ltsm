import pandas as pd
import numpy as np
from src.evaluation.metrics import sharpe_ratio, max_drawdown


class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0
        self.trades = []

    def run(self, prices, predictions):
        returns = []
        min_length = min(len(prices), len(predictions))

        for i in range(1, min_length):
            if predictions[i - 1] > prices[i - 1]:  # Buy signal
                self.buy(prices[i - 1], 1)
            elif predictions[i - 1] < prices[i - 1]:  # Sell signal
                self.sell(prices[i - 1], 1)

            # Calculate returns
            current_value = self.capital + self.positions * prices[i]
            returns.append((current_value - self.initial_capital) / self.initial_capital)

        # Close all positions at the end
        if self.positions > 0:
            self.sell(prices[-1], self.positions)
        elif self.positions < 0:
            self.buy(prices[-1], -self.positions)

        return pd.DataFrame({
            'returns': returns,
            'cumulative_returns': (1 + pd.Series(returns)).cumprod() - 1
        })

    def buy(self, price, amount):
        cost = price * amount
        if cost <= self.capital:
            self.capital -= cost
            self.positions += amount
            self.trades.append(('buy', price, amount))

    def sell(self, price, amount):
        if self.positions >= amount:
            self.capital += price * amount
            self.positions -= amount
            self.trades.append(('sell', price, amount))

    def evaluate(self, returns):
        total_return = returns['cumulative_returns'].iloc[-1]
        sharp_ratio = sharpe_ratio(returns['returns'])
        max_draw = max_drawdown(returns['returns'])

        return {
            'total_return': total_return,
            'sharpe_ratio': sharp_ratio,
            'max_drawdown': max_draw
        }


# Example usage
if __name__ == "__main__":
    # Generate some dummy data
    prices = np.random.rand(100) * 100 + 50
    predictions = prices + np.random.randn(100) * 5

    backtester = Backtester()
    returns = backtester.run(prices, predictions)
    results = backtester.evaluate(returns)

    print("Backtest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")