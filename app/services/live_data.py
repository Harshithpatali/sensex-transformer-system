import numpy as np
import pandas as pd
import pywt
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands


class LiveMarketData:

    def __init__(self):

        self.symbols = {
            "sensex": "^BSESN",
            "nifty": "^NSEI",
            "india_vix": "^INDIAVIX",
            "nasdaq": "^IXIC",
            "dow_jones": "^DJI",
            "crude_oil": "CL=F",
            "usdinr": "INR=X"
        }

    def download_asset(
        self,
        name,
        symbol
    ):

        df = yf.download(
            symbol,
            period="6mo",
            progress=False
        )

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns.get_level_values(0)
            )

        df.reset_index(inplace=True)

        if "Adj Close" in df.columns:

            df.drop(
                columns=["Adj Close"],
                inplace=True
            )

        rename_cols = {}

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:

            if col in df.columns:

                rename_cols[col] = (
                    f"{name}_{col.lower()}"
                )

        df.rename(
            columns=rename_cols,
            inplace=True
        )

        return df

    def merge_data(self):

        merged_df = None

        for name, symbol in (
            self.symbols.items()
        ):

            df = self.download_asset(
                name,
                symbol
            )

            if merged_df is None:

                merged_df = df

            else:

                merged_df = pd.merge(
                    merged_df,
                    df,
                    on="Date",
                    how="inner"
                )

        return merged_df

    def add_features(
        self,
        df
    ):

        # Returns
        close_cols = [
            col for col in df.columns
            if "close" in col.lower()
        ]

        for col in close_cols:

            df[f"{col}_return"] = np.log(
                df[col] / df[col].shift(1)
            )

        # Indicators
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

        df["rolling_volatility"] = (
            df[
                "sensex_close_return"
            ]
            .rolling(20)
            .std()
        )

        df["momentum_10"] = (
            close / close.shift(10)
        ) - 1

        # Wavelets
        prices = (
            df["sensex_close"]
            .ffill()
            .values
        )

        coeffs = pywt.wavedec(
            prices,
            "haar",
            level=2
        )

        cA2, cD2, cD1 = coeffs

        approx_signal = pywt.waverec(
            [
                cA2,
                np.zeros_like(cD2),
                np.zeros_like(cD1)
            ],
            "haar"
        )

        detail_signal_1 = pywt.waverec(
            [
                np.zeros_like(cA2),
                np.zeros_like(cD2),
                cD1
            ],
            "haar"
        )

        detail_signal_2 = pywt.waverec(
            [
                np.zeros_like(cA2),
                cD2,
                np.zeros_like(cD1)
            ],
            "haar"
        )

        df["wavelet_trend"] = (
            approx_signal[:len(df)]
        )

        df["wavelet_detail_1"] = (
            detail_signal_1[:len(df)]
        )

        df["wavelet_detail_2"] = (
            detail_signal_2[:len(df)]
        )

        df.dropna(inplace=True)

        return df