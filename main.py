import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from typing import List

from src.math_utils import EloSystem, process_match, get_chances
from src.data_processing import full_pipeline, split_data
from src.training import train_models, save_models, save_artifacts
from src.config import load_config


def evaluate_models(
    home_stack, 
    away_stack, 
    x_test: pd.DataFrame, 
    y_reg_test_home: pd.Series, 
    y_reg_test_away: pd.Series, 
    rho: float
) -> None:
    """Evaluate model accuracy on test holdout set."""
    print("Evaluating models on test dataset...")
    
    all_match_lambda_home = home_stack.predict(X=x_test)
    all_match_lambda_away = away_stack.predict(X=x_test)
    
    results = x_test.copy()
    results['home_lambda'] = all_match_lambda_home 
    results['away_lambda'] = all_match_lambda_away

    def evaluate_match_row(row_match):
        matrix = process_match(row_match.home_lambda, row_match.away_lambda, rho)
        chances = get_chances(matrix)
        return pd.Series([np.argmax(chances)])

    real_conditions = [
        y_reg_test_home > y_reg_test_away,
        y_reg_test_home == y_reg_test_away,
        y_reg_test_home < y_reg_test_away
    ]
    classes = [0, 1, 2]
    y_real = np.select(real_conditions, classes)
    
    y_pred = results.apply(evaluate_match_row, axis=1)
    
    final_acc = accuracy_score(y_real, y_pred)
    print(f"Final average prediction for win/draw/loss: {final_acc * 100:.2f}%")


def main() -> None:
    """Primary command-line entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description="WhereItLands - Football Match Outcome Predictor")
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Execution mode.")
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Compute and save Elo ratings and Parquet history without fitting models."
    )
    args = parser.parse_args()

    print("Loading configuration...")
    config = load_config()
    
    elo_system = EloSystem(
        k_values=config['k_values'], 
        initial_elo=config['constants']['INITIAL_ELO']
    )

    if args.mode == "train":
        print("Starting data processing and Elo computation...")
        data_path = config['paths']['unprocessed_data']
        
        df_final, team_history = full_pipeline(data_path, elo_system)
        
        save_artifacts(elo_system, team_history, config['paths']['artifacts_dir'])
        
        if args.skip_training:
            print("Option --skip-training enabled. Elo system and history artifacts saved successfully.")
            return

        training_cols: List[str] = [
            "diff_goals_5_matches", 
            "diff_vsgoals_5_matches", 
            "diff_streak_5_matches", 
            "elo_diff", 
            "home_advantage"
        ]
        x_train, x_test, y_reg_home_train, y_reg_away_train, y_reg_test_home, y_reg_test_away = split_data(df_final, training_cols)
        
        print(f"Fitting models with {len(x_train)} historical matches...")
        
        home_stack, away_stack = train_models(x_train, y_reg_home_train, y_reg_away_train, config)
        
        evaluate_models(home_stack, away_stack, x_test, y_reg_test_home, y_reg_test_away, config['constants']['RHO'])
        
        save_models(home_stack, away_stack, config['paths']['artifacts_dir'])
        print("Pipeline execution complete.")


if __name__ == "__main__":
    main()
