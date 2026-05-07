import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional information to Transformer inputs.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000
    ):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x


class TransformerPredictor(nn.Module):
    """
    Transformer model for stock market prediction.
    Predicts next-day return (regression only)
    """

    def __init__(self):

        super().__init__()

        self.input_dim = 26
        self.d_model = 128
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.4
        self.sequence_length = 60

        # Feature projection
        self.input_projection = nn.Linear(
            self.input_dim,
            self.d_model
        )

        # Positional encoding
        self.positional_encoding = (
            PositionalEncoding(
                d_model=self.d_model,
                max_len=self.sequence_length
            )
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            dim_feedforward=512,
            activation="gelu"
        )

        # Transformer encoder
        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.num_layers
            )
        )

        # Shared representation
        self.shared_fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Regression head
        self.return_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last timestep representation
        x = x[:, -1, :]

        # Shared representation
        x = self.shared_fc(x)

        predicted_return = (
            self.return_head(x)
        )

        return predicted_return.squeeze(-1)


if __name__ == "__main__":

    # Example batch
    batch_size = 32
    sequence_length = 60
    features = 23

    # Dummy input
    x = torch.randn(
        batch_size,
        sequence_length,
        features
    )

    # Model
    model = TransformerPredictor()

    # Forward pass
    predicted_return = model(x)

    print("Predicted Return Shape:")
    print(predicted_return.shape)