import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import StackingRegressor

from src.config import load_config
from src.data_processing.pipeline import get_available_teams
from src.inference.results_calc import get_predictions
from src.inference.schemas import PredictionRequest, PredictionResponse, TeamsResponse
from src.math_utils.EloSystem import EloSystem

home_stack: Optional[StackingRegressor] = None
away_stack: Optional[StackingRegressor] = None
elo_system: Optional[EloSystem] = None
team_history: Optional[pd.DataFrame] = None
config: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Administra el ciclo de vida de la aplicacion cargando configuracion, modelos y artefactos."""
    global config, home_stack, away_stack, elo_system, team_history
    print("Iniciando servidor. Cargando modelos y artefactos.")
    config = load_config()
    load_models()
    load_artifacts()
    try:
        yield
    finally:
        print("Deteniendo servidor.")
        home_stack = None
        away_stack = None
        team_history = None
        elo_system = None
        config = None

def load_models() -> None:
    """Carga los modelos entrenados desde la boveda (artifacts) al iniciar la API."""
    global home_stack, away_stack
    artifacts_dir: str = "artifacts"
    home_path: str = os.path.join(artifacts_dir, "home_stack.joblib")
    away_path: str = os.path.join(artifacts_dir, "away_stack.joblib")

    if os.path.exists(home_path) and os.path.exists(away_path):
        home_stack = joblib.load(home_path)
        away_stack = joblib.load(away_path)
        print("Modelos cargados exitosamente.")
    else:
        print("Advertencia: No se encontraron los modelos en artifacts/. Entrena el modelo primero.")

def load_artifacts() -> None:
    """Carga el dataframe historico y el sistema ELO desde la boveda al iniciar la API."""
    global elo_system, team_history
    artifacts_dir: str = "artifacts"
    elo_path: str = os.path.join(artifacts_dir, "elo_system.joblib")
    dataframe_path: str = os.path.join(artifacts_dir, "team_history_dataframe.parquet")

    if os.path.exists(elo_path) and os.path.exists(dataframe_path):
        elo_system = joblib.load(elo_path)
        team_history = pd.read_parquet(dataframe_path)
        print("Sistema ELO y dataframe historico cargados correctamente.")
    else:
        print("Advertencia: No se encontro el sistema ELO o el registro historico de partidos.")

app = FastAPI(
    title="WhereItLands Predictor", 
    description="API de Inferencia de Futbol", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root() -> Dict[str, str]:
    """Retorna el mensaje de bienvenida de la API."""
    return {"message": "Bienvenido a la API de WhereItLands Predictor."}

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Verifica el estado de disponibilidad de los modelos y artefactos."""
    status: str = "models_missing"
    if home_stack is not None and away_stack is not None and elo_system is not None and team_history is not None:
        status = "ready"
    return {"status": status}

@app.get("/teams", response_model=TeamsResponse)
def get_teams_list() -> TeamsResponse:
    """Retorna la lista de equipos disponibles en el registro historico."""
    if team_history is None:
        raise HTTPException(
            status_code=500,
            detail="La informacion de los equipos no se ha inicializado de forma correcta."
        )
    return TeamsResponse(teams=get_available_teams(team_history))

@app.post("/prediction", response_model=PredictionResponse)
def get_prediction(data: PredictionRequest) -> PredictionResponse:
    """Calcula y retorna la probabilidad de resultado para un partido especificado."""
    if home_stack is None or away_stack is None:
        raise HTTPException(
            status_code=500,
            detail="Los modelos de prediccion no se han inicializado de forma correcta."
        )

    if team_history is None or elo_system is None or config is None:
        raise HTTPException(
            status_code=500,
            detail="La informacion de los equipos o la configuracion no se ha inicializado de forma correcta."
        )

    teams: List[str] = get_available_teams(team_history)
    home_team: str = data.home_team
    away_team: str = data.away_team
    neutral: bool = data.neutral
    iterations: int = data.iterations

    if home_team == away_team:
        raise HTTPException(
            status_code=400,
            detail="El equipo local y el equipo visitante no pueden ser el mismo."
        )

    if home_team not in teams or away_team not in teams:
        raise HTTPException(
            status_code=400,
            detail="Uno o ambos equipos no se encuentran en la lista de equipos validos."
        )

    prediction_result: Dict[str, Any] = get_predictions(
        home_team=home_team,
        away_team=away_team,
        home_stack=home_stack,
        away_stack=away_stack,
        neutral=neutral,
        elo_sys=elo_system,
        team_history=team_history,
        config=config,
        iterations=iterations
    )

    return PredictionResponse(**prediction_result)