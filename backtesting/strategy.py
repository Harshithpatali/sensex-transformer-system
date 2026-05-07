import numpy as np


class TradingStrategy:

    def __init__(self):

        # Trading thresholds
        self.buy_threshold = 0.01

        self.sell_threshold = -0.01

    def generate_signals(
        self,
        predicted_returns
    ):

        signals = []

        for pred in predicted_returns:

            if pred > self.buy_threshold:

                signals.append(1)

            elif pred < self.sell_threshold:

                signals.append(-1)

            else:

                signals.append(0)

        return np.array(signals)