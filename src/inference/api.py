from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import os
import pandas as pd
from schemas import TeamsResponse, PredictionResponse
from data_processing import get_available_teams

# Variables globales para los modelos
home_stack = None
away_stack = None
elo_system = None
team_history = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print("Iniciando servidor. Cargando modelos.")
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
    if(team_history is None):
        raise HTTPException(
            status_code=503,
            detail='La información de los equipos no se ha inicializado de forma correcta. Intente luego.'
        )
    return get_available_teams(team_history)