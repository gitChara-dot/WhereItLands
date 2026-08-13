from pydantic import BaseModel, Field
from typing import List, Tuple

class PredictionRequest(BaseModel):
    home_team: str = Field(..., description="Nombre del equipo local.", examples=["Argentina"])
    away_team: str = Field(..., description="Nombre del equipo visitante.", examples=["France"])
    neutral: bool = Field(default=True, description="Indica si el partido se juega en campo neutral.")
    iterations: int = Field(default=3, ge=1, le=6, description="Cantidad de resultados mas probables a calcular (entre 1 y 6).", examples=[3])

class PredictionResponse(BaseModel):
    home_win_chance: float = Field(..., description="Probabilidad de victoria del equipo local.")
    draw_chance: float = Field(..., description="Probabilidad de empate.")
    away_win_chance: float = Field(..., description="Probabilidad de victoria del equipo visitante.")
    top_results: List[Tuple[int, int, float]] = Field(..., description="Lista de resultados mas probables en formato (goles_local, goles_visitante, probabilidad).")

class TeamsResponse(BaseModel):
    teams: List[str] = Field(..., description="Lista de nombres de equipos disponibles.")