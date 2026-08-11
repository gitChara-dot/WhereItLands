import os
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform, loguniform

from src.math_utils.EloSystem import EloSystem
from src.data_processing.pipeline import full_pipeline, split_data


class ModelTrainer:
    """
    Clase responsable de configurar, optimizar y entrenar los modelos de regresión 
    (StackingRegressor con XGBoost, LightGBM y ExtraTrees) para los goles de local y visitante.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.seed: int = config['training']['seed']
        self.n_iter_base: int = config['training']['n_iter_base']
        self.tscv = TimeSeriesSplit(n_splits=5)

    def get_param_distributions() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Devuelve las distribuciones de hiperparámetros para RandomizedSearchCV."""
        xgb_param_dist = {
            'n_estimators': randint(50, 250),
            'learning_rate': uniform(0.01, 0.15),
            'max_depth': randint(3, 8),
            'subsample': uniform(0.6, 0.4)
        }

        lgbm_param_dist = {
            'n_estimators': randint(50, 250),
            'learning_rate': uniform(0.01, 0.15),
            'max_depth': randint(3, 8),
            'subsample': uniform(0.6, 0.4),
            'subsample_freq': [1]
        }

        extra_trees_param_dist = {
            'n_estimators': randint(50, 250),
            'max_depth': [3, 5, 8, 12, None],
            'min_samples_split': randint(5, 25)
        }

        alphas_final_reg = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 500.0])

        return xgb_param_dist, lgbm_param_dist, extra_trees_param_dist, alphas_final_reg

    def build_and_train_stack(
        self, 
        x_train: pd.DataFrame, 
        y_train: pd.Series, 
        is_home: bool
    ) -> StackingRegressor:
        """Construye y entrena el StackingRegressor para el objetivo especificado."""
        xgb_dist, lgbm_dist, et_dist, alphas = ModelTrainer.get_param_distributions() # type: ignore
        final_optimized_regressor = RidgeCV(alphas=alphas, cv=self.tscv)

        # 1. XGBoost Regressor
        search_xgb = RandomizedSearchCV(
            estimator=XGBRegressor(objective='count:poisson', eval_metric='poisson-nloglik'),
            param_distributions=xgb_dist,
            n_iter=self.n_iter_base,
            n_jobs=-1,
            cv=self.tscv,
            random_state=self.seed
        )
        search_xgb.fit(x_train, y_train)

        # 2. LightGBM Regressor
        search_lgbm = RandomizedSearchCV(
            estimator=LGBMRegressor(objective='poisson', verbose=-1), # type: ignore
            param_distributions=lgbm_dist,
            n_iter=self.n_iter_base,
            n_jobs=-1,
            cv=self.tscv,
            random_state=self.seed
        )
        search_lgbm.fit(x_train, y_train)

        # 3. ExtraTrees Regressor con TransformedTargetRegressor (log1p / expm1)
        criterion = 'squared_error' if is_home else 'absolute_error'
        search_et = RandomizedSearchCV(
            estimator=ExtraTreesRegressor(criterion=criterion),
            param_distributions=et_dist,
            n_iter=self.n_iter_base,
            n_jobs=-1,
            cv=self.tscv,
            random_state=self.seed
        )
        search_et.fit(x_train, y_train)

        et_wrapped = TransformedTargetRegressor(
            regressor=search_et.best_estimator_,
            func=np.log1p,
            inverse_func=np.expm1
        )

        # Stacking Regressor
        estimators = [
            ('xgb', search_xgb.best_estimator_),
            ('et', et_wrapped),
            ('lgb', search_lgbm.best_estimator_)
        ]

        stack = StackingRegressor(
            estimators=estimators, 
            final_estimator=final_optimized_regressor
        )
        stack.fit(x_train, y_train)

        return stack

    def train(
        self, 
        x_train: pd.DataFrame, 
        y_reg_home_train: pd.Series, 
        y_reg_away_train: pd.Series
    ) -> Tuple[StackingRegressor, StackingRegressor]:
        """Entrena ambos stacks (Local y Visitante)."""
        print("Iniciando entrenamiento del modelo Local (Home)...")
        home_stack = self.build_and_train_stack(x_train, y_reg_home_train, is_home=True)
        print("Entrenamiento Home finalizado.")

        print("Iniciando entrenamiento del modelo Visitante (Away)...")
        away_stack = self.build_and_train_stack(x_train, y_reg_away_train, is_home=False)
        print("Entrenamiento Away finalizado.")

        return home_stack, away_stack


def train_models(
    x_train: pd.DataFrame, 
    y_reg_home_train: pd.Series, 
    y_reg_away_train: pd.Series, 
    config: Dict[str, Any]
) -> Tuple[StackingRegressor, StackingRegressor]:
    """Función de conveniencia para entrenar modelos desde main.py u otros scripts."""
    trainer = ModelTrainer(config)
    return trainer.train(x_train, y_reg_home_train, y_reg_away_train)


def save_models(home_stack: StackingRegressor, away_stack: StackingRegressor, artifacts_dir: str) -> None:
    """Guarda los modelos entrenados en archivos .joblib dentro de la bóveda."""
    os.makedirs(artifacts_dir, exist_ok=True)
    home_path = os.path.join(artifacts_dir, "home_stack.joblib")
    away_path = os.path.join(artifacts_dir, "away_stack.joblib")

    joblib.dump(home_stack, home_path)
    joblib.dump(away_stack, away_path)
    print(f"Modelos guardados exitosamente en '{artifacts_dir}'.")

def save_artifacts(elo_sys : EloSystem, team_history : pd.DataFrame, artifacts_dir : str) -> None:
    """Guarda el sistema de ELO, como .joblib, y el dataframe de los partidos historicos, como .parquet, dentro de la bóveda"""
    os.makedirs(artifacts_dir, exist_ok=True)

    elo_path = os.path.join(artifacts_dir, "elo_system.joblib")
    dataframe_path = os.path.join(artifacts_dir, "team_history_dataframe.parquet")

    joblib.dump(elo_sys, elo_path)
    team_history.to_parquet(dataframe_path)

def run_training_pipeline(config_path: str = "config/config.yaml") -> Tuple[StackingRegressor, StackingRegressor]:
    """
    Función de ejecución completa del capataz (train_pipeline.py):
    Carga configuración, procesa datos con el pipeline, entrena los modelos y los guarda en la bóveda.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("--- [PIPELINE] Iniciando ejecucion del Pipeline de Entrenamiento ---")
    
    # 1. Instanciar Calculadora
    elo_system = EloSystem(
        k_values=config['k_values'],
        initial_elo=config['constants']['INITIAL_ELO']
    )

    # 2. Llamar a los operarios (Pipeline de Datos)
    data_path = config['paths']['unprocessed_data']
    df_final, _ = full_pipeline(data_path, elo_system)

    training_cols = [
        "diff_goals_5_matches", 
        "diff_vsgoals_5_matches", 
        "diff_streak_5_matches", 
        "elo_diff", 
        "home_advantage"
    ]

    x_train, _,  y_reg_home_train, y_reg_away_train, _, _ = split_data(df_final, training_cols)

    # 3. Entrenar a XGBoost / LightGBM / ExtraTrees en Stacking
    home_stack, away_stack = train_models(x_train, y_reg_home_train, y_reg_away_train, config)

    # 4. Guardar en la bóveda
    save_models(home_stack, away_stack, config['paths']['artifacts_dir'])
    print("--- [PIPELINE] Pipeline finalizado con exito ---")

    return home_stack, away_stack


if __name__ == "__main__":
    run_training_pipeline()
