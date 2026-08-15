import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    """Manage application startup and shutdown lifecycle by loading configuration and artifacts."""
    global config, home_stack, away_stack, elo_system, team_history
    print("Starting server. Loading models and artifacts...")
    config = load_config()
    load_models()
    load_artifacts()
    try:
        yield
    finally:
        print("Shutting down server. Cleaning memory state...")
        home_stack = None
        away_stack = None
        team_history = None
        elo_system = None
        config = None


def load_models() -> None:
    """Load serialized regression models from the artifacts directory."""
    global home_stack, away_stack
    artifacts_dir: str = "artifacts"
    home_path: str = os.path.join(artifacts_dir, "home_stack.joblib")
    away_path: str = os.path.join(artifacts_dir, "away_stack.joblib")

    if os.path.exists(home_path) and os.path.exists(away_path):
        home_stack = joblib.load(home_path)
        away_stack = joblib.load(away_path)
        print("Models loaded successfully.")
    else:
        print("Warning: Model artifacts not found. Train the models first.")


def load_artifacts() -> None:
    """Load the historical dataframe and EloSystem instance from the artifacts directory."""
    global elo_system, team_history
    artifacts_dir: str = "artifacts"
    elo_path: str = os.path.join(artifacts_dir, "elo_system.joblib")
    dataframe_path: str = os.path.join(artifacts_dir, "team_history_dataframe.parquet")

    if os.path.exists(elo_path) and os.path.exists(dataframe_path):
        elo_system = joblib.load(elo_path)
        team_history = pd.read_parquet(dataframe_path)
        print("Elo system and historical records loaded successfully.")
    else:
        print("Warning: Elo system or historical parquet artifact not found.")


app = FastAPI(
    title="WhereItLands Predictor", 
    description="International Football Match Outcome Prediction API", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Check readiness status of loaded models and artifacts."""
    status: str = "models_missing"
    if home_stack is not None and away_stack is not None and elo_system is not None and team_history is not None:
        status = "ready"
    return {"status": status}


@app.get("/teams", response_model=TeamsResponse)
def get_teams_list() -> TeamsResponse:
    """Return the list of available historical teams."""
    if team_history is None:
        raise HTTPException(
            status_code=500,
            detail="Team history records have not been properly initialized."
        )
    return TeamsResponse(teams=get_available_teams(team_history))


@app.post("/prediction", response_model=PredictionResponse)
def get_prediction(data: PredictionRequest) -> PredictionResponse:
    """Calculate and return match outcome probabilities for the requested teams."""
    if home_stack is None or away_stack is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction models have not been properly initialized."
        )

    if team_history is None or elo_system is None or config is None:
        raise HTTPException(
            status_code=500,
            detail="Team history or configuration has not been properly initialized."
        )

    teams: List[str] = get_available_teams(team_history)
    home_team: str = data.home_team
    away_team: str = data.away_team
    neutral: bool = data.neutral
    iterations: int = data.iterations
    date: Optional[str] = data.date

    if home_team == away_team:
        raise HTTPException(
            status_code=400,
            detail="Home team and away team cannot be the same."
        )

    if home_team not in teams or away_team not in teams:
        raise HTTPException(
            status_code=400,
            detail="One or both teams were not found in the historical records."
        )

    if date:
        min_date = str(team_history["date"].min())
        if date < min_date:
            raise HTTPException(
                status_code=400,
                detail="Date is before historical data."
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
        iterations=iterations, 
        date=date
    )

    return PredictionResponse(**prediction_result)


# Mount static frontend directory at root to serve UI automatically
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")