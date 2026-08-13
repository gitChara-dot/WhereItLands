from pydantic import BaseModel, Field
from typing import Any

class PredictionRequest(BaseModel):
    home_team : str = Field(..., description="Home Team", examples=["Argentina"])
    away_team : str = Field(..., description="Away Team", examples=["France"])
    neutral : bool = Field(default=True, description="True if the match is played on even ground.")
    iterations : int = Field(default=1, description="Enter the number of possible outcomes. For each iteration, result is less likely. Max 6.", examples=["4"])

class PredictionResponse(BaseModel):
    results : dict[str, Any]

class TeamsResponse(BaseModel):
    teams : list[str]