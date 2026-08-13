import os
import yaml
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
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
    """Clase responsable de configurar, optimizar y entrenar los modelos de regresion para los goles de local y visitante."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config
        self.seed: int = config['training']['seed']
        self.n_iter_base: int = config['training']['n_iter_base']
        self.tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=5)

    @staticmethod
    def get_param_distributions() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Devuelve las distribuciones de hiperparametros para RandomizedSearchCV."""
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
        xgb_dist, lgbm_dist, et_dist, alphas = ModelTrainer.get_param_distributions()
        final_optimized_regressor = RidgeCV(alphas=alphas, cv=self.tscv)

        search_xgb = RandomizedSearchCV(
            estimator=XGBRegressor(objective='count:poisson', eval_metric='poisson-nloglik'),
            param_distributions=xgb_dist,
            n_iter=self.n_iter_base,
            n_jobs=-1,
            cv=self.tscv,
            random_state=self.seed
        )
        search_xgb.fit(x_train, y_train)

        search_lgbm = RandomizedSearchCV(
            estimator=LGBMRegressor(objective='poisson', verbose=-1), #type: ignore
            param_distributions=lgbm_dist,
            n_iter=self.n_iter_base,
            n_jobs=-1,
            cv=self.tscv,
            random_state=self.seed
        )
        search_lgbm.fit(x_train, y_train)

        criterion: str = 'squared_error' if is_home else 'absolute_error'
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
    """Funcion auxiliar para instanciar el entrenador y ejecutar el entrenamiento de modelos."""
    trainer = ModelTrainer(config)
    return trainer.train(x_train, y_reg_home_train, y_reg_away_train)


def save_models(home_stack: StackingRegressor, away_stack: StackingRegressor, artifacts_dir: str) -> None:
    """Guarda los modelos entrenados en archivos .joblib dentro del directorio de artefactos."""
    os.makedirs(artifacts_dir, exist_ok=True)
    home_path: str = os.path.join(artifacts_dir, "home_stack.joblib")
    away_path: str = os.path.join(artifacts_dir, "away_stack.joblib")

    joblib.dump(home_stack, home_path)
    joblib.dump(away_stack, away_path)
    print(f"Modelos guardados exitosamente en '{artifacts_dir}'.")


def save_artifacts(elo_sys: EloSystem, team_history: pd.DataFrame, artifacts_dir: str) -> None:
    """Guarda el sistema ELO en .joblib y el dataframe historico en .parquet dentro de los artefactos."""
    os.makedirs(artifacts_dir, exist_ok=True)
    elo_path: str = os.path.join(artifacts_dir, "elo_system.joblib")
    dataframe_path: str = os.path.join(artifacts_dir, "team_history_dataframe.parquet")

    joblib.dump(elo_sys, elo_path)
    team_history.to_parquet(dataframe_path, index=False)
    print(f"Sistema ELO e historial guardados exitosamente en '{artifacts_dir}'.")


def run_training_pipeline(
    config_path: str = "config/config.yaml", 
    skip_training: bool = False
) -> Tuple[Optional[StackingRegressor], Optional[StackingRegressor]]:
    """Ejecuta el pipeline completo de procesamiento de datos, calculo de ELO y entrenamiento opcional de modelos."""
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    print("--- [PIPELINE] Iniciando ejecucion del Pipeline de Datos y ELO ---")
    
    elo_system = EloSystem(
        k_values=config['k_values'],
        initial_elo=config['constants']['INITIAL_ELO']
    )

    data_path: str = config['paths']['unprocessed_data']
    df_final, team_history = full_pipeline(data_path, elo_system)

    save_artifacts(elo_system, team_history, artifacts_dir=config['paths']['artifacts_dir'])

    if skip_training:
        print("Opcion skip_training activada. Se omitio el entrenamiento de modelos.")
        print("--- [PIPELINE] Pipeline finalizado exitosamente (Solo artefactos) ---")
        return None, None

    training_cols: List[str] = [
        "diff_goals_5_matches", 
        "diff_vsgoals_5_matches", 
        "diff_streak_5_matches", 
        "elo_diff", 
        "home_advantage"
    ]

    x_train, _, y_reg_home_train, y_reg_away_train, _, _ = split_data(df_final, training_cols)

    home_stack, away_stack = train_models(x_train, y_reg_home_train, y_reg_away_train, config)
    save_models(home_stack, away_stack, config['paths']['artifacts_dir'])
    print("--- [PIPELINE] Pipeline finalizado con exito ---")

    return home_stack, away_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento y generacion de artefactos.")
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Procesa los datos y guarda unicamente el sistema ELO y el historial Parquet sin entrenar modelos."
    )
    args = parser.parse_args()
    run_training_pipeline(skip_training=args.skip_training)
