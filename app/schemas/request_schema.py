from typing import List

from pydantic import BaseModel


class PredictionRequest(BaseModel):

    features: List[List[float]]