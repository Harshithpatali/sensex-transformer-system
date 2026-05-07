import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class MultiMarketDataLoader:

    def __init__(self):

        self.start_date = "2000-01-01"

        self.symbols = {
            "sensex": "^BSESN",
            "nifty": "^NSEI",
            "india_vix": "^INDIAVIX",
            "nasdaq": "^IXIC",
            "dow_jones": "^DJI",
            "crude_oil": "CL=F",
            "usdinr": "INR=X"
        }

        self.output_path = Path(
            "data/raw"
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def download_single_asset(
        self,
        name,
        symbol
    ):

        logger.info(
            f"Downloading {name}..."
        )

        df = yf.download(
            symbol,
            start=self.start_date,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            raise ValueError(
                f"{name} dataset empty."
            )

        # Fix MultiIndex columns
        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns.get_level_values(0)
            )

        # Reset index
        df.reset_index(inplace=True)

        # Remove Adj Close
        if "Adj Close" in df.columns:

            df.drop(
                columns=["Adj Close"],
                inplace=True
            )

        # Rename columns
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

            df = self.download_single_asset(
                name,
                symbol
            )

            logger.info(
                f"{name} shape: {df.shape}"
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

                logger.info(
                    f"Merged shape: "
                    f"{merged_df.shape}"
                )

        # Sort by date
        merged_df.sort_values(
            "Date",
            inplace=True
        )

        # Reset index
        merged_df.reset_index(
            drop=True,
            inplace=True
        )

        return merged_df

    def save_data(
        self,
        df
    ):

        output_file = (
            self.output_path
            / "global_market_data.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        logger.info(
            f"Saved merged dataset to "
            f"{output_file}"
        )

    def run(self):

        df = self.merge_data()

        print("\nDataset Preview:\n")

        print(df.head())

        print("\nDataset Shape:\n")

        print(df.shape)

        print("\nColumns:\n")

        print(df.columns.tolist())

        self.save_data(df)


if __name__ == "__main__":

    loader = MultiMarketDataLoader()

    loader.run()