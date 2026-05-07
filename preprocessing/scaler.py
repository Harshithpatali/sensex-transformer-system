import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DataScaler:

    def __init__(self):

        self.input_path = Path(
            "data/features/advanced_features.csv"
        )

        self.output_path = Path(
            "data/processed"
        )

        self.scaler_path = Path(
            "data/scaler"
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.scaler_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Final curated feature set
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

    def load_data(self) -> pd.DataFrame:

        logger.info(
            "Loading feature dataset..."
        )

        df = pd.read_csv(
            self.input_path
        )

        logger.info(
            f"Dataset shape: {df.shape}"
        )

        return df

    def validate_features(
        self,
        df: pd.DataFrame
    ):

        logger.info(
            "Validating feature columns..."
        )

        missing_features = []

        for feature in self.feature_columns:

            if feature not in df.columns:

                missing_features.append(
                    feature
                )

        if len(missing_features) > 0:

            raise ValueError(
                f"Missing features: "
                f"{missing_features}"
            )

        logger.info(
            "All required features found."
        )

    def split_data(
        self,
        df: pd.DataFrame
    ):

        logger.info(
            "Performing time-series split..."
        )

        train_size = int(
            len(df) * 0.7
        )

        val_size = int(
            len(df) * 0.15
        )

        train_df = df.iloc[
            :train_size
        ]

        val_df = df.iloc[
            train_size:
            train_size + val_size
        ]

        test_df = df.iloc[
            train_size + val_size:
        ]

        logger.info(
            f"Train size: "
            f"{len(train_df)}"
        )

        logger.info(
            f"Validation size: "
            f"{len(val_df)}"
        )

        logger.info(
            f"Test size: "
            f"{len(test_df)}"
        )

        return (
            train_df,
            val_df,
            test_df
        )

    def scale_data(
        self,
        train_df,
        val_df,
        test_df
    ):

        logger.info(
            "Scaling datasets..."
        )

        scaler = StandardScaler()

        # Train fit
        train_scaled = (
            scaler.fit_transform(
                train_df[
                    self.feature_columns
                ]
            )
        )

        # Validation transform
        val_scaled = (
            scaler.transform(
                val_df[
                    self.feature_columns
                ]
            )
        )

        # Test transform
        test_scaled = (
            scaler.transform(
                test_df[
                    self.feature_columns
                ]
            )
        )

        # Save scaler
        joblib.dump(
            scaler,
            self.scaler_path
            / "feature_scaler.pkl"
        )

        logger.info(
            "Scaler saved."
        )

        # Convert to DataFrames
        train_scaled_df = pd.DataFrame(
            train_scaled,
            columns=self.feature_columns
        )

        val_scaled_df = pd.DataFrame(
            val_scaled,
            columns=self.feature_columns
        )

        test_scaled_df = pd.DataFrame(
            test_scaled,
            columns=self.feature_columns
        )

        # Targets
        train_scaled_df[
            "target_return"
        ] = train_df[
            "target_return"
        ].values

        val_scaled_df[
            "target_return"
        ] = val_df[
            "target_return"
        ].values

        test_scaled_df[
            "target_return"
        ] = test_df[
            "target_return"
        ].values

        train_scaled_df[
            "target_direction"
        ] = train_df[
            "target_direction"
        ].values

        val_scaled_df[
            "target_direction"
        ] = val_df[
            "target_direction"
        ].values

        test_scaled_df[
            "target_direction"
        ] = test_df[
            "target_direction"
        ].values

        # Drop NaNs
        train_scaled_df.dropna(
            inplace=True
        )

        val_scaled_df.dropna(
            inplace=True
        )

        test_scaled_df.dropna(
            inplace=True
        )

        return (
            train_scaled_df,
            val_scaled_df,
            test_scaled_df
        )

    def save_processed_data(
        self,
        train_df,
        val_df,
        test_df
    ):

        train_df.to_csv(
            self.output_path
            / "train_scaled.csv",
            index=False
        )

        val_df.to_csv(
            self.output_path
            / "val_scaled.csv",
            index=False
        )

        test_df.to_csv(
            self.output_path
            / "test_scaled.csv",
            index=False
        )

        logger.info(
            "Processed datasets saved."
        )

    def run(self):

        df = self.load_data()

        self.validate_features(df)

        (
            train_df,
            val_df,
            test_df
        ) = self.split_data(df)

        (
            train_scaled_df,
            val_scaled_df,
            test_scaled_df
        ) = self.scale_data(
            train_df,
            val_df,
            test_df
        )

        self.save_processed_data(
            train_scaled_df,
            val_scaled_df,
            test_scaled_df
        )

        print(
            "\nProcessed Train Dataset:\n"
        )

        print(
            train_scaled_df.head()
        )

        print(
            f"\nFeature Count: "
            f"{len(self.feature_columns)}"
        )


if __name__ == "__main__":

    scaler = DataScaler()

    scaler.run()