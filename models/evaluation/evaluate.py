import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import DataLoader

from preprocessing.dataset_builder import (
    DatasetBuilder
)

from app.models.transformer_model import (
    TransformerPredictor
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class Evaluator:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Using device: {self.device}"
        )

        # Load dataset
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

        # Load model
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

        self.pred_returns = []
        self.true_returns = []

        self.pred_directions = []
        self.true_directions = []

    def evaluate(self):

        logger.info("Running evaluation...")

        with torch.no_grad():

            for (
                X,
                y_return,
                y_direction
            ) in self.test_loader:

                X = X.to(self.device)

                pred_return = self.model(X)

                # Regression predictions
                self.pred_returns.extend(
                    pred_return.cpu().numpy()
                )

                self.true_returns.extend(
                    y_return.numpy()
                )

                # Direction derived from predicted returns
                threshold = 0.001

                pred_classes = (
                    pred_return > threshold
                ).int()

                self.pred_directions.extend(
                    pred_classes.cpu().numpy()
                )

                self.true_directions.extend(
                    y_direction.numpy()
                )

        self.calculate_metrics()

        self.plot_predictions()

    def calculate_metrics(self):

        rmse = np.sqrt(
            mean_squared_error(
                self.true_returns,
                self.pred_returns
            )
        )

        mae = mean_absolute_error(
            self.true_returns,
            self.pred_returns
        )

        r2 = r2_score(
            self.true_returns,
            self.pred_returns
        )

        direction_acc = accuracy_score(
            self.true_directions,
            self.pred_directions
        )

        cm = confusion_matrix(
            self.true_directions,
            self.pred_directions
        )

        logger.info("\n===== METRICS =====")

        logger.info(f"RMSE: {rmse:.6f}")

        logger.info(f"MAE: {mae:.6f}")

        logger.info(f"R2 Score: {r2:.6f}")

        logger.info(
            f"Directional Accuracy: "
            f"{direction_acc:.4f}"
        )

        logger.info(
            f"Mean Predicted Return: "
            f"{np.mean(self.pred_returns):.6f}"
        )

        logger.info(
            f"Std Predicted Return: "
            f"{np.std(self.pred_returns):.6f}"
        )

        logger.info("\nConfusion Matrix:")

        logger.info(f"\n{cm}")

        logger.info(
            "\nClassification Report:"
        )

        logger.info(
            "\n"
            + classification_report(
                self.true_directions,
                self.pred_directions
            )
        )

    def plot_predictions(self):

        plt.figure(figsize=(14, 7))

        plt.plot(
            self.true_returns[:200],
            label="True Returns"
        )

        plt.plot(
            self.pred_returns[:200],
            label="Predicted Returns"
        )

        plt.title(
            "True vs Predicted Log Returns"
        )

        plt.xlabel("Time")

        plt.ylabel("Log Return")

        plt.legend()

        plt.grid(True)

        plt.show()


if __name__ == "__main__":

    evaluator = Evaluator()

    evaluator.evaluate()