import numpy as np


class BacktestMetrics:

    @staticmethod
    def cumulative_returns(
        strategy_returns
    ):

        return np.cumprod(
            1 + strategy_returns
        )

    @staticmethod
    def sharpe_ratio(
        strategy_returns,
        risk_free_rate=0.0
    ):

        excess_returns = (
            strategy_returns
            - risk_free_rate
        )

        return (
            np.mean(excess_returns)
            /
            (
                np.std(excess_returns)
                + 1e-8
            )
        ) * np.sqrt(252)

    @staticmethod
    def max_drawdown(
        cumulative_returns
    ):

        peak = np.maximum.accumulate(
            cumulative_returns
        )

        drawdown = (
            cumulative_returns - peak
        ) / peak

        return np.min(drawdown)

    @staticmethod
    def win_rate(
        strategy_returns
    ):

        wins = (
            strategy_returns > 0
        ).sum()

        total = (
            strategy_returns != 0
        ).sum()

        if total == 0:

            return 0

        return wins / total