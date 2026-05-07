import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from preprocessing.dataset_builder import (
    DatasetBuilder
)

from app.models.transformer_model import (
    TransformerPredictor
)

from backtesting.strategy import (
    TradingStrategy
)

from backtesting.metrics import (
    BacktestMetrics
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class Backtester:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Using device: {self.device}"
        )

        # Dataset
        builder = DatasetBuilder()

        self.test_dataset = (
            builder.build_dataset(
                "test_scaled.csv"
            )
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False
        )

        # Model
        self.model = (
            TransformerPredictor()
            .to(self.device)
        )

        self.model.load_state_dict(
            torch.load(
                "models/saved/best_transformer.pth",
                map_location=self.device
            )
        )

        self.model.eval()

        self.strategy = (
            TradingStrategy()
        )

    def run(self):

        logger.info(
            "Running backtest..."
        )

        predicted_returns = []

        actual_returns = []

        with torch.no_grad():

            for (
                X,
                y_return,
                _
            ) in self.test_loader:

                X = X.to(self.device)

                pred_return = (
                    self.model(X)
                )

                predicted_returns.extend(
                    pred_return
                    .cpu()
                    .numpy()
                )

                actual_returns.extend(
                    y_return.numpy()
                )

        predicted_returns = np.array(
            predicted_returns
        )

        actual_returns = np.array(
            actual_returns
        )

        # Generate trading signals
        signals = (
            self.strategy
            .generate_signals(
                predicted_returns
            )
        )

        # Strategy returns
        strategy_returns = (
            signals * actual_returns
        )

        # Metrics
        cumulative = (
            BacktestMetrics
            .cumulative_returns(
                strategy_returns
            )
        )

        sharpe = (
            BacktestMetrics
            .sharpe_ratio(
                strategy_returns
            )
        )

        max_dd = (
            BacktestMetrics
            .max_drawdown(
                cumulative
            )
        )

        win_rate = (
            BacktestMetrics
            .win_rate(
                strategy_returns
            )
        )

        logger.info(
            "\n===== BACKTEST RESULTS ====="
        )

        logger.info(
            f"Total Return: "
            f"{(cumulative[-1]-1)*100:.2f}%"
        )

        logger.info(
            f"Sharpe Ratio: "
            f"{sharpe:.4f}"
        )

        logger.info(
            f"Max Drawdown: "
            f"{max_dd:.4f}"
        )

        logger.info(
            f"Win Rate: "
            f"{win_rate:.4f}"
        )

        logger.info(
            f"Total Trades: "
            f"{(signals != 0).sum()}"
        )

        # Plot equity curve
        plt.figure(figsize=(14, 7))

        plt.plot(
            cumulative,
            label="Strategy Equity Curve"
        )

        plt.title(
            "Transformer Trading Strategy"
        )

        plt.xlabel("Trades")

        plt.ylabel("Portfolio Value")

        plt.legend()

        plt.grid(True)

        plt.show()


if __name__ == "__main__":

    backtester = Backtester()

    backtester.run()