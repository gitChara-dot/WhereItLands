import yaml
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from src.math_utils import EloSystem, process_match, get_chances
from src.data_processing import full_pipeline, split_data
from src.training import train_models, save_models
from src.config import load_config


def evaluate_models(home_stack, away_stack, x_test, y_reg_test_home, y_reg_test_away, rho: float):
    print("Evaluando modelos con datos de test...")
    
    # Predecir lambdas
    all_match_lambda_home = home_stack.predict(X=x_test)
    all_match_lambda_away = away_stack.predict(X=x_test)
    
    results = x_test.copy()
    results['home_lambda'] = all_match_lambda_home 
    results['away_lambda'] = all_match_lambda_away

    # Función interna para evaluar una fila
    def evaluate_match_row(row_match):
        matrix = process_match(row_match.home_lambda, row_match.away_lambda, rho)
        chances = get_chances(matrix)
        return pd.Series([np.argmax(chances)])

    # Clases reales
    real_conditions = [
        y_reg_test_home > y_reg_test_away,
        y_reg_test_home == y_reg_test_away,
        y_reg_test_home < y_reg_test_away
    ]
    classes = [0, 1, 2] # 0: local gana, 1: empate, 2: visitante gana
    y_real = np.select(real_conditions, classes)
    
    # Predicción
    y_pred = results.apply(evaluate_match_row, axis=1)
    
    final_acc = accuracy_score(y_real, y_pred)
    print(f"Final average prediction for win/draw/lose: {final_acc * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="WhereItLands - Predicción de Fútbol")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Modo de ejecución")
    args = parser.parse_args()

    # 1. Cargar configuración
    print("Cargando configuración...")
    config = load_config()
    
    # 2. Inicializar sistema ELO
    elo_system = EloSystem(
        k_values=config['k_values'], 
        initial_elo=config['constants']['INITIAL_ELO']
    )

    if args.mode == "train":
        print("Iniciando procesamiento de datos...")
        data_path = config['paths']['unprocessed_data']
        
        # 3. Ejecutar pipeline de datos
        df_final, team_history = full_pipeline(data_path, elo_system)
        
        # 4. Dividir datos
        training_cols = ["diff_goals_5_matches", "diff_vsgoals_5_matches", "diff_streak_5_matches", "elo_diff", "home_advantage"]
        x_train, x_test, y_reg_home_train, y_reg_away_train, y_reg_test_home, y_reg_test_away = split_data(df_final, training_cols)
        
        print(f"Entrenamiento con {len(x_train)} partidos.")
        
        # 5. Entrenar modelos
        home_stack, away_stack = train_models(x_train, y_reg_home_train, y_reg_away_train, config)
        
        # 6. Evaluar
        evaluate_models(home_stack, away_stack, x_test, y_reg_test_home, y_reg_test_away, config['constants']['RHO'])
        
        # 7. Guardar
        save_models(home_stack, away_stack, config['paths']['artifacts_dir'])
        print("Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
