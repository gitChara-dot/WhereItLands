from pydantic import BaseModel, Field
from typing import Any
class PredictionRequest(BaseModel):
    home_team : str = Field(..., description="Home Team", examples=["Argentina"])
    away_team : str = Field(..., description="Away Team", examples=["France"])
    neutral : bool = Field(default=True, description="True if the match is played on even ground.")

class PredictionResponse(BaseModel):
    results : dict[str, Any]