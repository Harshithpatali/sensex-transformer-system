import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:

    def __init__(self):

        self.input_path = Path(
            "data/raw/global_market_data.csv"
        )

        self.output_path = Path(
            "data/features"
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def load_data(self):

        logger.info("Loading data...")

        df = pd.read_csv(self.input_path)

        return df

    def create_returns(
        self,
        df
    ):

        logger.info(
            "Creating log returns..."
        )

        close_cols = [
            col for col in df.columns
            if "close" in col.lower()
        ]

        for col in close_cols:

            df[f"{col}_return"] = np.log(
                df[col] / df[col].shift(1)
            )

        return df

    def add_market_features(
        self,
        df
    ):

        logger.info(
            "Adding advanced indicators..."
        )

        # SENSEX indicators
        close = df["sensex_close"]

        df["sensex_rsi"] = (
            RSIIndicator(
                close=close,
                window=14
            ).rsi()
        )

        df["sensex_sma_20"] = (
            SMAIndicator(
                close=close,
                window=20
            ).sma_indicator()
        )

        bb = BollingerBands(close)

        df["bb_high"] = (
            bb.bollinger_hband()
        )

        df["bb_low"] = (
            bb.bollinger_lband()
        )

        # Volatility
        df["rolling_volatility"] = (
            df[
                "sensex_close_return"
            ]
            .rolling(20)
            .std()
        )

        # Momentum
        df["momentum_10"] = (
            close / close.shift(10)
        ) - 1

        return df

    def add_wavelet_features(
        self,
        df
    ):

        logger.info(
            "Adding Haar wavelet features..."
        )

        close_prices = (
            df["sensex_close"]
            .ffill()
            .values
        )

        # Haar decomposition
        coeffs = pywt.wavedec(
            close_prices,
            "haar",
            level=2
        )

        cA2, cD2, cD1 = coeffs

        # Reconstruct approximation
        approx_signal = pywt.waverec(
            [
                cA2,
                np.zeros_like(cD2),
                np.zeros_like(cD1)
            ],
            "haar"
        )

        # Reconstruct detail level 1
        detail_signal_1 = pywt.waverec(
            [
                np.zeros_like(cA2),
                np.zeros_like(cD2),
                cD1
            ],
            "haar"
        )

        # Reconstruct detail level 2
        detail_signal_2 = pywt.waverec(
            [
                np.zeros_like(cA2),
                cD2,
                np.zeros_like(cD1)
            ],
            "haar"
        )

        # Match dataframe length
        approx_signal = approx_signal[
            :len(df)
        ]

        detail_signal_1 = detail_signal_1[
            :len(df)
        ]

        detail_signal_2 = detail_signal_2[
            :len(df)
        ]

        # Add wavelet features
        df["wavelet_trend"] = (
            approx_signal
        )

        df["wavelet_detail_1"] = (
            detail_signal_1
        )

        df["wavelet_detail_2"] = (
            detail_signal_2
        )

        return df

    def create_targets(
        self,
        df
    ):

        logger.info(
            "Creating targets..."
        )

        # 5-day future return
        df["target_return"] = (
            np.log(
                df["sensex_close"]
                .shift(-5)
                / df["sensex_close"]
            )
        )

        # Ignore tiny noisy moves
        df["target_direction"] = (
            (
                df["target_return"] > 0.003
            )
            .astype(int)
        )

        return df

    def clean_data(
        self,
        df
    ):

        logger.info("Cleaning data...")

        df.dropna(inplace=True)

        df.reset_index(
            drop=True,
            inplace=True
        )

        return df

    def save_data(
        self,
        df
    ):

        output_file = (
            self.output_path
            / "advanced_features.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        logger.info(
            f"Saved features to "
            f"{output_file}"
        )

    def run(self):

        df = self.load_data()

        df = self.create_returns(df)

        df = self.add_market_features(df)

        df = self.add_wavelet_features(df)

        df = self.create_targets(df)

        df = self.clean_data(df)

        print(df.head())

        self.save_data(df)


if __name__ == "__main__":

    engineer = (
        AdvancedFeatureEngineer()
    )

    engineer.run()