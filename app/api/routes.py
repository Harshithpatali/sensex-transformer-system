from fastapi import APIRouter

from app.schemas.request_schema import (
    PredictionRequest
)

from app.services.predictor import (
    PredictorService
)

from app.services.live_data import (
    LiveMarketData
)

router = APIRouter()

# Services
predictor = PredictorService()

live_data_service = (
    LiveMarketData()
)


@router.get("/")
def home():

    return {
        "message":
        "Transformer Quant API Running"
    }


@router.post("/predict")
def predict(
    request: PredictionRequest
):

    prediction = predictor.predict(
        request.features
    )

    signal = "HOLD"

    if prediction > 0.03:

        signal = "BUY"

    elif prediction < -0.03:

        signal = "SELL"

    return {

        "predicted_return":
        prediction,

        "signal":
        signal
    }


@router.post("/live-predict")
def live_predict():

    # Download live market data
    df = live_data_service.merge_data()

    # Feature engineering
    df = live_data_service.add_features(df)

    # EXACT SAME features used in training
    feature_columns = [

        # SENSEX OHLCV
        "sensex_open",
        "sensex_high",
        "sensex_low",
        "sensex_close",
        "sensex_volume",

        # Global markets
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

        # Wavelets
        "wavelet_trend",
        "wavelet_detail_1",
        "wavelet_detail_2"
    ]

    # Last 60 trading days
    features = (
        df[feature_columns]
        .tail(60)
        .values
    )

    prediction = predictor.predict(
        features
    )

    signal = "HOLD"

    if prediction > 0.03:

        signal = "BUY"

    elif prediction < -0.03:

        signal = "SELL"

    return {

        "predicted_return":
        prediction,

        "signal":
        signal
    }