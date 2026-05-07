import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing.window_generator import (
    WindowGenerator
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class FinancialDataset(Dataset):

    def __init__(
        self,
        X,
        y_return,
        y_direction
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y_return = torch.tensor(
            y_return,
            dtype=torch.float32
        )

        self.y_direction = torch.tensor(
            y_direction,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y_return[idx],
            self.y_direction[idx]
        )


class DatasetBuilder:

    def __init__(self):

        self.processed_path = Path(
            "data/processed"
        )

        self.output_path = Path(
            "data/processed"
        )

        self.window_generator = (
            WindowGenerator()
        )

    def load_dataset(
        self,
        filename
    ):

        logger.info(f"Loading {filename}")

        df = pd.read_csv(
            self.processed_path / filename
        )

        return df

    def build_dataset(
        self,
        filename
    ):

        df = self.load_dataset(filename)

        X, y_return, y_direction = (
            self.window_generator
            .create_sequences(df)
        )

        dataset = FinancialDataset(
            X,
            y_return,
            y_direction
        )

        logger.info(
            f"Dataset size: {len(dataset)}"
        )

        return dataset


if __name__ == "__main__":

    builder = DatasetBuilder()

    train_dataset = (
        builder.build_dataset(
            "train_scaled.csv"
        )
    )

    print("\nSample Tensor Shapes:\n")

    sample = train_dataset[0]

    print(f"Input Shape: {sample[0].shape}")

    print(f"Return Target: {sample[1]}")

    print(f"Direction Target: {sample[2]}")