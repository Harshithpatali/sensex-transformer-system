import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class WindowGenerator:

    def __init__(self):

        # Sequence length
        self.window_size = 60

        # EXACT SAME feature list as scaler.py
        self.feature_columns = [

            # SENSEX OHLCV
            "sensex_open",
            "sensex_high",
            "sensex_low",
            "sensex_close",
            "sensex_volume",

            # Global market closes
            "nifty_close",
            "india_vix_close",
            "nasdaq_close",
            "dow_jones_close",
            "crude_oil_close",
            "usdinr_close",

            # Returns
            "sensex_close_return",
            "nifty_close_return",
            "nasdaq_close_return",
            "dow_jones_close_return",
            "crude_oil_close_return",
            "usdinr_close_return",

            # Technical indicators
            "sensex_rsi",
            "sensex_sma_20",
            "bb_high",
            "bb_low",
            "rolling_volatility",
            "momentum_10",

            # Wavelet features
            "wavelet_trend",
            "wavelet_detail_1",
            "wavelet_detail_2"
        ]

    def create_sequences(
        self,
        df: pd.DataFrame
    ):

        logger.info(
            "Creating sliding window sequences..."
        )

        X = []

        y_return = []

        y_direction = []

        # Features
        features = (
            df[self.feature_columns]
            .values
        )

        # Targets
        target_return = (
            df["target_return"]
            .values
        )

        target_direction = (
            df["target_direction"]
            .values
        )

        # Sliding windows
        for i in range(
            self.window_size,
            len(df)
        ):

            X.append(
                features[
                    i - self.window_size:i
                ]
            )

            y_return.append(
                target_return[i]
            )

            y_direction.append(
                target_direction[i]
            )

        # Convert to arrays
        X = np.array(X)

        y_return = np.array(y_return)

        y_direction = np.array(y_direction)

        logger.info(
            f"X shape: {X.shape}"
        )

        logger.info(
            f"y_return shape: "
            f"{y_return.shape}"
        )

        logger.info(
            f"y_direction shape: "
            f"{y_direction.shape}"
        )

        return (
            X,
            y_return,
            y_direction
        )