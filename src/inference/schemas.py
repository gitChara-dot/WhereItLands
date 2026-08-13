from pydantic import BaseModel, Field
from typing import List, Tuple


class PredictionRequest(BaseModel):
    """Request payload schema for single match outcome prediction."""
    home_team: str = Field(..., description="Name of the home team.", examples=["Argentina"])
    away_team: str = Field(..., description="Name of the away team.", examples=["France"])
    neutral: bool = Field(default=True, description="Whether the match is played on neutral ground.")
    iterations: int = Field(default=3, ge=1, le=6, description="Number of most probable scorelines to return (1-6).", examples=[3])


class PredictionResponse(BaseModel):
    """Response payload schema containing match outcome probabilities and top scorelines."""
    home_win_chance: float = Field(..., description="Probability of home team victory.")
    draw_chance: float = Field(..., description="Probability of a draw.")
    away_win_chance: float = Field(..., description="Probability of away team victory.")
    top_results: List[Tuple[int, int, float]] = Field(..., description="List of most probable scorelines formatted as (home_goals, away_goals, percentage).")


class TeamsResponse(BaseModel):
    """Response payload schema for the list of available historical teams."""
    teams: List[str] = Field(..., description="Alphabetically sorted list of available team names.")