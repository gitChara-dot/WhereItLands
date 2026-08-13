from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import os
import pandas as pd
from src.inference.schemas import TeamsResponse, PredictionRequest, PredictionResponse
from src.data_processing import get_available_teams
from src.inference.results_calc import get_predictions
from src.config import load_config


# Variables globales para los modelos
home_stack = None
away_stack = None
elo_system = None
team_history = None
config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print("Iniciando servidor. Cargando modelos.")
    config = load_config()
    load_models()
    load_artifacts()
    try:
        yield
    finally:
        print("Deteniendo servidor.")
        global home_stack, away_stack, team_history, elo_system
        home_stack = None
        away_stack = None
        team_history = None
        elo_system = None
        config = None

def load_models():
    """Carga los modelos entrenados desde la bóveda (artifacts) al iniciar la API."""
    global home_stack, away_stack
    
    # Ruta relativa a la bóveda (asegurarse de correr la API desde el directorio raíz)
    artifacts_dir = "artifacts"
    home_path = os.path.join(artifacts_dir, "home_stack.joblib")
    away_path = os.path.join(artifacts_dir, "away_stack.joblib")
    
    if os.path.exists(home_path) and os.path.exists(away_path):
        home_stack = joblib.load(home_path)
        away_stack = joblib.load(away_path)
        print("Modelos cargados exitosamente.")
    else:
        print("Advertencia: No se encontraron los modelos en artifacts/. Entrena el modelo primero.")

def load_artifacts():
    """Carga el dataframe historico de los partidos por equipo y el sistema de ELO actualizado al iniciar la API."""
    global elo_system, team_history

    artifacts_dir = "artifacts"
    
    elo_path = os.path.join(artifacts_dir, "elo_system.joblib")
    dataframe_path = os.path.join(artifacts_dir, "team_history_dataframe.parquet")

    if(os.path.exists(elo_path) and os.path.exists(dataframe_path)):
        elo_system = joblib.load(elo_path)
        team_history = pd.read_parquet(dataframe_path)
        print("Sistema y dataframe cargados correctamente.")
    else:
        print("Advertencia: No se encontró el sistema de elo o el registro historico de partidos.")
        
app = FastAPI(title="WhereItLands Predictor", description="API de Inferencia de Fútbol", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de WhereItLands Predictor. (En construcción)"}

@app.get("/health")
def health_check():
    status = "models_missing"
    if home_stack is not None and away_stack is not None:
        status = "ready"

    return {"status": status}

@app.get("/teams", response_model=TeamsResponse)
def get_teams_list():
    if team_history is None:
        raise HTTPException(
            status_code=500,
            detail='La información de los equipos no se ha inicializado de forma correcta. Intente luego.'
        )
    return get_available_teams(team_history)

@app.post("/prediction", response_model=PredictionResponse)
def get_prediction(data: PredictionRequest):
    global team_history, home_stack, away_stack, elo_system

    if home_stack is None or away_stack is None:
        raise HTTPException(
            status_code=500,
            detail='Los modelos de prediccion no se han inicializado de forma correcta. Intente luego.'
        )

    if team_history is None or elo_system is None:
        raise HTTPException(
            status_code=500,
            detail='La información de los equipos no se ha inicializado de forma correcta. Intente luego.'
        )
    
    teams = get_available_teams(team_history)
    home_team : str = data.home_team
    away_team : str = data.away_team
    neutral : bool = data.neutral
    iterations : int = data.iterations

    if iterations <= 0 or iterations > 6:
        raise HTTPException(
            status_code=400,
            detail='La cantidad de iteraciones no es valida. Intenta con un numero mayor a 0 y menor o igual a 6.'
        )
    
    if home_team not in teams or away_team not in teams:
        raise HTTPException(
            status_code=400,
            detail='El equipo no está en la lista de equipos válidos.'
        )
    
    return get_predictions(
        home_team, 
        away_team,
        home_stack, 
        away_stack, 
        neutral, 
        elo_system, 
        team_history,
        config=load_config()
    )
    
    