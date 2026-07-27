from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import os

app = FastAPI(title="WhereItLands Predictor", description="API de Inferencia de Fútbol")

# Variables globales para los modelos
home_stack = None
away_stack = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando servidor. Cargando modelos.")
    load_models()

    yield

    print("Deteniendo servidor.")
    global home_stack, away_stack
    home_stack = None
    away_stack = None

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

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de WhereItLands Predictor. (En construcción)"}

@app.get("/health")
def health_check():
    status = "models_missing"
    if home_stack is not None and away_stack is not None:
        status = "ready"

    return {"status": status}
