import joblib
import numpy as np
import torch

from app.models.transformer_model import (
    TransformerPredictor
)


class PredictorService:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
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

        # Load scaler
        self.scaler = joblib.load(
            "data/scaler/feature_scaler.pkl"
        )

    def predict(
        self,
        features
    ):

        features = np.array(
            features
        )

        # Scale
        features = (
            self.scaler.transform(
                features
            )
        )

        # Shape:
        # (1, 60, 26)
        features = np.expand_dims(
            features,
            axis=0
        )

        tensor = torch.tensor(
            features,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():

            prediction = (
                self.model(tensor)
            )

        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        return float(prediction)