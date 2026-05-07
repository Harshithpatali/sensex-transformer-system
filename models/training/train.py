import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import (
    autocast,
    GradScaler
)

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


class Trainer:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Using device: {self.device}"
        )

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 40

        # Early stopping patience
        self.patience = 5

        self.model_path = Path(
            "models/saved"
        )

        self.model_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # Dataset
        builder = DatasetBuilder()

        self.train_dataset = (
            builder.build_dataset(
                "train_scaled.csv"
            )
        )

        self.val_dataset = (
            builder.build_dataset(
                "val_scaled.csv"
            )
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Model
        self.model = (
            TransformerPredictor()
            .to(self.device)
        )

        # Regression loss
        self.regression_loss = (
            nn.MSELoss()
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Learning rate scheduler
        self.scheduler = (
            torch.optim.lr_scheduler
            .ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=3
            )
        )

        # Mixed precision scaler
        self.scaler = GradScaler()

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0

        for (
            X,
            y_return,
            _
        ) in self.train_loader:

            X = X.to(self.device)

            y_return = (
                y_return.to(self.device)
            )

            self.optimizer.zero_grad()

            with autocast():

                pred_return = self.model(X)

                reg_loss = (
                    self.regression_loss(
                        pred_return,
                        y_return
                    )
                )

                loss = reg_loss

            # Backpropagation
            self.scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # Optimizer step
            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            total_loss += loss.item()

        return (
            total_loss
            / len(self.train_loader)
        )

    def validate(self):

        self.model.eval()

        total_loss = 0

        with torch.no_grad():

            for (
                X,
                y_return,
                _
            ) in self.val_loader:

                X = X.to(self.device)

                y_return = (
                    y_return.to(self.device)
                )

                pred_return = self.model(X)

                reg_loss = (
                    self.regression_loss(
                        pred_return,
                        y_return
                    )
                )

                loss = reg_loss

                total_loss += loss.item()

        return (
            total_loss
            / len(self.val_loader)
        )

    def train(self):

        best_val_loss = float("inf")

        # Early stopping counter
        patience_counter = 0

        for epoch in range(self.epochs):

            train_loss = (
                self.train_one_epoch()
            )

            val_loss = (
                self.validate()
            )

            # Scheduler step
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}"
            )

            logger.info(
                f"Train Loss: {train_loss:.6f}"
            )

            logger.info(
                f"Validation Loss: {val_loss:.6f}"
            )

            logger.info(
                f"Learning Rate: "
                f"{self.optimizer.param_groups[0]['lr']}"
            )

            # Save best model
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    self.model_path
                    / "best_transformer.pth"
                )

                logger.info(
                    "Best model saved."
                )

                # Reset patience counter
                patience_counter = 0

            else:

                patience_counter += 1

                logger.info(
                    f"Patience Counter: "
                    f"{patience_counter}"
                )

                if patience_counter >= self.patience:

                    logger.info(
                        "Early stopping triggered."
                    )

                    break

        logger.info(
            "Training completed."
        )


if __name__ == "__main__":

    trainer = Trainer()

    trainer.train()